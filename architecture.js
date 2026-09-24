(() => {
  'use strict';

  const story = document.getElementById('architecture-story');
  const controls = document.getElementById('architecture-controls');
  const play = document.getElementById('architecture-play');
  const status = document.getElementById('architecture-status');
  if (!story || !controls || !play) return;

  const steps = Array.from(story.querySelectorAll('[data-architecture-step]'));
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const duration = 10500;
  const timings = { separate: 0, shared: 3000, architecture: 7500 };
  const descriptions = {
    separate: 'Each layer has its own router and experts.',
    shared: 'Each layer keeps its own router and gains a depth embedding before routing to the shared expert pool.',
    architecture: 'The full architecture shows MoRE alongside standard MoE.',
  };

  let elapsed = preference.matches ? duration : 0;
  let requestedPlaying = !preference.matches;
  let inView = false;
  let pageActive = true;
  let frame = 0;
  let lastTime = null;
  let active = false;
  const pausedTransitions = new Set();

  function globalPaused() {
    return document.documentElement.classList.contains('motion-paused');
  }

  function phaseAt(time) {
    if (time >= timings.architecture) return 'architecture';
    return time >= timings.shared ? 'shared' : 'separate';
  }

  function canPlay() {
    return requestedPlaying && elapsed < duration && inView && pageActive &&
      !document.hidden && !globalPaused() &&
      !document.documentElement.classList.contains('intro-active');
  }

  function render() {
    const phase = phaseAt(elapsed);
    story.dataset.phase = phase;
    story.dataset.playing = String(active);
    for (const step of steps) {
      step.setAttribute('aria-pressed', String(step.dataset.architectureStep === phase));
    }

    const globallyPaused = globalPaused();
    play.disabled = false;
    play.title = globallyPaused ? 'Play this animation and resume page animations.' : '';
    play.textContent = elapsed >= duration
      ? 'Replay animation'
      : active ? 'Pause animation' : 'Play animation';
    // aria-pressed represents active playback, including automatic visibility pauses.
    play.setAttribute('aria-pressed', String(active));

    if (status) {
      const message = globallyPaused
        ? `${descriptions[phase]} Animations are paused; use the playback button to resume.`
        : descriptions[phase];
      if (status.textContent !== message) status.textContent = message;
    }
  }

  function pauseTransitions() {
    if (!story.getAnimations) return;
    for (const animation of story.getAnimations({ subtree: true })) {
      // CSS dotted paths are paused by data-playing. Preserve in-flight fades and moves too.
      if ('transitionProperty' in animation && animation.playState === 'running') {
        animation.pause();
        pausedTransitions.add(animation);
      }
    }
  }

  function resumeTransitions() {
    for (const animation of pausedTransitions) {
      if (animation.playState === 'paused') animation.play();
    }
    pausedTransitions.clear();
  }

  function advance(now) {
    if (lastTime !== null) elapsed = Math.min(duration, elapsed + Math.max(0, now - lastTime));
    lastTime = now;
    if (elapsed >= duration) requestedPlaying = false;
  }

  function stopClock() {
    if (frame) window.cancelAnimationFrame(frame);
    frame = 0;
    if (active) advance(performance.now());
    lastTime = null;
    active = false;
  }

  function reconcile() {
    if (canPlay()) {
      if (!active) {
        active = true;
        lastTime = performance.now();
        render();
        resumeTransitions();
      }
      if (!frame) frame = window.requestAnimationFrame(tick);
    } else {
      const wasActive = active;
      stopClock();
      render();
      if (wasActive) pauseTransitions();
    }
    render();
  }

  function tick(now) {
    frame = 0;
    if (!canPlay()) {
      reconcile();
      return;
    }
    advance(now);
    render();
    if (elapsed >= duration) {
      active = false;
      lastTime = null;
      render();
      return;
    }
    frame = window.requestAnimationFrame(tick);
  }

  play.addEventListener('click', () => {
    if (globalPaused()) {
      // Explicit playback remains available after the floating intro is dismissed.
      // Let the existing control also update the fireworks' internal pause state.
      document.getElementById('motion-toggle')?.click();
      if (globalPaused()) return;
    }
    if (active) {
      requestedPlaying = false;
    } else {
      if (elapsed >= duration) elapsed = 0;
      if (preference.matches) {
        story.classList.add('story-motion-enabled');
      }
      requestedPlaying = true;
    }
    reconcile();
  });

  for (const step of steps) {
    step.addEventListener('click', () => {
      const phase = step.dataset.architectureStep;
      if (!Object.prototype.hasOwnProperty.call(timings, phase)) return;
      stopClock();
      requestedPlaying = false;
      // Selecting a stage must not resume a transition from a previous stage.
      for (const animation of pausedTransitions) animation.cancel();
      pausedTransitions.clear();
      elapsed = phase === 'architecture' ? duration : timings[phase];
      render();
    });
  }

  const preferenceChanged = () => {
    stopClock();
    story.classList.remove('story-motion-enabled');
    if (preference.matches) {
      requestedPlaying = false;
      elapsed = duration;
      for (const animation of pausedTransitions) animation.cancel();
      pausedTransitions.clear();
    }
    // A preference change never restarts a completed or manually paused story.
    reconcile();
  };
  if (preference.addEventListener) preference.addEventListener('change', preferenceChanged);
  else if (preference.addListener) preference.addListener(preferenceChanged);

  document.addEventListener('visibilitychange', reconcile);
  window.addEventListener('pagehide', () => {
    pageActive = false;
    reconcile();
  });
  window.addEventListener('pageshow', () => {
    pageActive = true;
    reconcile();
  });

  if ('MutationObserver' in window) {
    new MutationObserver(reconcile).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });
  }

  controls.hidden = false;
  play.hidden = false;
  render();

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting && entry.intersectionRatio >= 0.25;
      reconcile();
    }, { threshold: [0, 0.25] }).observe(story);
  } else {
    const checkVisibility = () => {
      const bounds = story.getBoundingClientRect();
      const visibleHeight = Math.max(0, Math.min(bounds.bottom, window.innerHeight) - Math.max(bounds.top, 0));
      inView = bounds.height > 0 && visibleHeight / bounds.height >= 0.25;
      reconcile();
    };
    window.addEventListener('scroll', checkVisibility, { passive: true });
    window.addEventListener('resize', checkVisibility, { passive: true });
    checkVisibility();
  }
})();
