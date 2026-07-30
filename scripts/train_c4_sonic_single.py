import os
import torch
import math
import argparse
from transformers import (
    LlamaTokenizerFast,
    MixtralForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from datasets import load_from_disk



# H100 Tuning
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["WANDB_PROJECT"] = "C4_sonic"
# Force Dynamo off so Accelerator never compiles (avoids _grouped_mm BF16-only meta error with DeepSeek V3 / MoE)
os.environ["ACCELERATE_DYNAMO_BACKEND"] = "no"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="configs/example/more-deepseek_v3-114m")
    # This repo ships no dataset. Supply your own in HuggingFace
    # save_to_disk format -- see README, "Data", for the required columns.
    parser.add_argument("--train_data", type=str, required=True,
                        help="path to a datasets.save_to_disk training set")
    parser.add_argument("--val_data", type=str, required=True,
                        help="path to a datasets.save_to_disk validation set")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--eval_steps", type=int, default=10_000)
    parser.add_argument("--stop_at_step", type=int, default=None, help="Stop training at this step (max_steps still the ceiling)")
    parser.add_argument("--append_to_run_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir (default: ./c4-sonic/<run_name>)")
    parser.add_argument("--model_type", type=str, default="mixtral", choices=["mixtral", "moeut"], help="Model type: mixtral (from_pretrained) or moeut (wrapper)")
    # Only profiles defined in the cloned MoEUT repo's profiles.py work. The
    # paper's larger MoEUT baselines used profiles appended to that file by
    # hand, which are not upstream and are not shipped here.
    parser.add_argument("--moeut_profile", type=str, default="MoEUT_126M",
                        help="MoEUT profile name from the cloned repo's "
                             "profiles.py (default: MoEUT_126M)")
    args = parser.parse_args()
    return args


class StopAtStepCallback(TrainerCallback):
    """Stop training when global_step >= stop_at_step (e.g. to cap at 10k while max_steps=100k)."""

    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.stop_at_step:
            control.should_training_stop = True  # Trainer checks this, not should_stop
        return control

class MoEPerplexityTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nll_sum = 0.0
        self.aux_loss_sum = 0.0
        self.nll_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        total_loss = outputs.loss
        aux_loss = getattr(outputs, "aux_loss", None)  # MoE has it; Mistral (dense) does not
        
        actual_model = model.module if hasattr(model, "module") else model
        aux_coef = getattr(actual_model, "router_aux_loss_coef", 0.0)

        # Track NLL for perplexity
        # Some models (e.g. Qwen3 MoE) may return aux_loss=0 (int) when there is no aux term.
        if aux_loss is not None and hasattr(aux_loss, "detach"):
            nll = total_loss.detach() - (aux_coef * aux_loss.detach()) if aux_coef > 0 else total_loss.detach()
            self.aux_loss_sum += aux_loss.detach().item()
        else:
            nll = total_loss.detach()

        self.nll_sum += nll.item()
        self.nll_steps += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Save and reset counters so eval doesn't mix with training accumulators
        saved_nll, saved_aux, saved_steps = self.nll_sum, self.aux_loss_sum, self.nll_steps
        self.nll_sum = 0.0
        self.aux_loss_sum = 0.0
        self.nll_steps = 0
        self._evaluating = True

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None if self.compute_metrics is not None else True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.nll_steps > 0:
            avg_nll = self.nll_sum / self.nll_steps
            avg_aux = self.aux_loss_sum / self.nll_steps
            try:
                perplexity = math.exp(avg_nll)
            except OverflowError:
                perplexity = float("inf")
            output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity
            output.metrics[f"{metric_key_prefix}_aux_loss"] = avg_aux

        # Log eval metrics directly (bypass the log() override which is for training)
        super().log(output.metrics)
        self._evaluating = False

        # Restore training counters so no training steps are lost
        self.nll_sum = saved_nll
        self.aux_loss_sum = saved_aux
        self.nll_steps = saved_steps

        return output.metrics

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        # Only inject training perplexity/aux_loss during training, not eval
        if self.nll_steps > 0 and not getattr(self, "_evaluating", False):
            avg_nll = self.nll_sum / self.nll_steps
            avg_aux = self.aux_loss_sum / self.nll_steps
            try:
                logs["perplexity"] = math.exp(avg_nll)
            except OverflowError:
                logs["perplexity"] = float("inf")
            logs["aux_loss"] = avg_aux
            self.nll_sum = 0.0
            self.aux_loss_sum = 0.0
            self.nll_steps = 0
        super().log(logs, *args, **kwargs)

def count_parameters(model):
    """Return (total, unique).

    remove_duplicate=False is load-bearing: expert tying assigns the SAME
    nn.Parameter object to several layers, and named_parameters() de-duplicates
    by object identity by default. Without it this reported total == unique for
    every model, so the "Total Parameters" line below was really the unique
    count. Matches count_parameters in scripts/gen_conf_*.py.
    """
    unique_params = set()
    total_params = 0
    unique_params_count = 0
    for name, p in model.named_parameters(remove_duplicate=False):
        ptr = p.data_ptr()
        total_params += p.numel()
        if ptr not in unique_params:
            unique_params.add(ptr)
            unique_params_count += p.numel()
    return total_params, unique_params_count


args = get_args()

# The paper's runs are all on NVIDIA hardware. A few TrainingArguments fields
# below hard-require it, so gate them on this rather than assume it.
_CUDA = torch.cuda.is_available()

# Resolved from this file, not the CWD, so the script works from any directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tokenizer = LlamaTokenizerFast.from_pretrained(
    os.path.join(REPO_ROOT, "c4_tokenizer")
)

vocab_size = len(tokenizer)
pad_multiple = 128
padded_vocab_size = ((vocab_size + pad_multiple - 1) // pad_multiple) * pad_multiple

if getattr(args, "model_type", "mixtral") == "moeut":
    # Repo root, not the script's own directory: `more` is a top-level package
    # and this script lives in scripts/. get_moeut_profile raises MoEUTNotFound
    # with clone instructions when MOEUT_PATH is unset, so no default here.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from more.moeut_hf import MoEUTLMForCausalLM, get_moeut_profile
    profile = get_moeut_profile(args.moeut_profile)
    model = MoEUTLMForCausalLM(vocab_size=padded_vocab_size, router_aux_loss_coef=0.001, **profile)
else:
    # configs/example/* holds config.json only -- no weights. from_pretrained
    # requires a weights file, so training from an architecture spec (the
    # normal case here, since no checkpoints ship) needs from_config, which
    # builds a randomly-initialized model. from_pretrained is kept for
    # resuming from a real checkpoint directory.
    _weight_files = ("model.safetensors", "pytorch_model.bin")
    _has_weights = any(
        os.path.exists(os.path.join(args.model_name_or_path, f))
        for f in _weight_files
    )
    if _has_weights:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,
            use_cache=False,
            attn_implementation="sdpa"
        )
    else:
        _cfg = AutoConfig.from_pretrained(args.model_name_or_path)
        _cfg.use_cache = False
        model = AutoModelForCausalLM.from_config(
            _cfg, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
    if padded_vocab_size != vocab_size:
        model.resize_token_embeddings(padded_vocab_size)

total, unique = count_parameters(model)
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    # Both numbers, labelled. They differ whenever experts are tied, and the
    # paper's scale labels track `unique`. Note the vocab is padded to a
    # multiple of 128 above, so these run slightly above the config's counts.
    print(f"Parameters: {total:,} total, {unique:,} unique "
          f"(vocab padded {vocab_size} -> {padded_vocab_size})")

train_dataset = load_from_disk(args.train_data)
val_dataset = load_from_disk(args.val_data)

if getattr(args, "model_type", "mixtral") == "moeut":
    run_name = f"moeut-{args.moeut_profile}"
else:
    if args.model_name_or_path.endswith('/'):
        run_name = f"{args.model_name_or_path.split('/')[-2]}"
    else:
        run_name = f"{args.model_name_or_path.split('/')[-1]}"

run_name += args.append_to_run_name

output_dir = args.output_dir if getattr(args, "output_dir", None) else f"./c4-sonic/{run_name}"
training_args = TrainingArguments(
    output_dir=output_dir,
    
    per_device_train_batch_size=args.batch_size, 
    per_device_eval_batch_size=args.batch_size,
    
    gradient_accumulation_steps=args.grad_accum_steps,
    gradient_checkpointing=False, 
    
    # 1. DISABLE COMPILE for MoE (avoids _grouped_mm BF16-only meta error with DeepSeek V3 / grouped MoE)
    # Do not set torch_compile_mode or torch_compile_backend when torch_compile=False, or TrainingArguments will enable compile.
    torch_compile=False, 
    
    
    
    # Dataloader tuning for small models
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2, 
    
    # Gated on CUDA rather than hardcoded: tf32 requires an Ampere-or-newer
    # NVIDIA GPU and TrainingArguments raises outright without one, so the
    # script could not even start on a CPU machine. On GPU this is unchanged
    # from the paper's runs; on CPU it falls back to fp32, which is the only
    # option there anyway.
    bf16=_CUDA,
    tf32=_CUDA,

    # Without CUDA, force CPU rather than letting Trainer pick Apple MPS.
    # transformers' grouped-MoE path calls torch.histc, which Metal does not
    # implement for integer tensors, so an MPS run dies partway through the
    # first step. CPU is slow but correct, which is what a smoke run needs.
    use_cpu=not _CUDA,

    optim="adamw_torch_fused" if _CUDA else "adamw_torch",
    learning_rate=2.5e-4,
    logging_steps=50,
    save_total_limit=2,
    do_eval=True,
    max_steps=args.max_steps,
    save_steps=args.eval_steps,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    max_grad_norm= 0.25,
    warmup_steps=int(0.04 * args.max_steps),
    weight_decay=0.1,
    report_to=["wandb"] if args.use_wandb else ["none"],
    run_name=f"{run_name}",
)

callbacks = []
if getattr(args, "stop_at_step", None) is not None:
    callbacks.append(StopAtStepCallback(args.stop_at_step))

trainer = MoEPerplexityTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=None,
    preprocess_logits_for_metrics=None,
    callbacks=callbacks if callbacks else None,
)

print("Starting training...")
trainer.train()