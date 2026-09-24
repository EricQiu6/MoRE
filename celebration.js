(() => {
  'use strict';

  const canvas = document.getElementById('celebration-canvas');
  const hero = document.getElementById('opportunities');
  const celebrate = document.getElementById('celebrate');
  const motionToggle = document.getElementById('motion-toggle');
  if (!canvas || !hero) return;

  let context;
  try {
    context = canvas.getContext('2d', { alpha: true });
  } catch (_) {
    return;
  }
  if (!context) return;

  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const colors = ['#e6399b', '#7c3aed', '#2563eb', '#f97316', '#0891b2', '#e6a008'];
  const openingBursts = [
    { time: 0.25, x: 0.14, y: 0.25 },
    { time: 1.8, x: 0.87, y: 0.37 },
    { time: 3.3, x: 0.2, y: 0.7 },
    { time: 4.85, x: 0.8, y: 0.18 },
  ];
  let width = 1;
  let height = 1;
  let particleLimit = 180;
  let particles = [];
  let frame = 0;
  let lastTime = 0;
  let activeTime = 0;
  let nextOpening = 0;
  let nextSpark = 0;
  let paused = motionPreference.matches;
  let heroVisible = true;
  let manualUntil = 0;

  function syncControls() {
    document.documentElement.classList.toggle('motion-paused', paused);
    document.documentElement.classList.toggle('motion-enabled', !paused);
    if (motionToggle) {
      motionToggle.setAttribute('aria-pressed', String(paused));
      motionToggle.textContent = paused ? 'Play animations' : 'Pause animations';
      motionToggle.hidden = false;
    }
    if (celebrate) celebrate.hidden = false;
  }

  function resize() {
    const bounds = hero.getBoundingClientRect();
    width = Math.max(1, bounds.width);
    height = Math.max(1, bounds.height);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    particleLimit = width < 640 ? 90 : 180;
    particles = [];
  }

  function addParticle(particle) {
    if (particles.length >= particleLimit) particles.shift();
    particles.push(particle);
  }

  function burst(x, y, strength = 1) {
    const count = Math.round((width < 640 ? 38 : 62) * strength);
    const spread = Math.min(width, height) * (width < 640 ? 0.27 : 0.32);
    const hueOffset = Math.floor(Math.random() * colors.length);
    for (let index = 0; index < count; index += 1) {
      const angle = (Math.PI * 2 * index) / count + Math.random() * 0.15;
      const speed = spread * (0.38 + Math.random() * 0.62);
      addParticle({
        x,
        y,
        oldX: x,
        oldY: y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0,
        duration: 1.25 + Math.random() * 1.25,
        size: 1.4 + Math.random() * 2.3,
        angle,
        spin: (Math.random() - 0.5) * 5,
        color: colors[(index + hueOffset) % colors.length],
        confetti: index % 4 === 0,
        spark: false,
      });
    }
  }

  function driftingSpark() {
    const edge = Math.random() < 0.5;
    addParticle({
      x: width * (edge ? Math.random() * 0.27 : 0.73 + Math.random() * 0.27),
      y: height * (0.2 + Math.random() * 0.8),
      vx: (Math.random() - 0.5) * 9,
      vy: -(8 + Math.random() * 12),
      life: 0,
      duration: 3 + Math.random() * 3,
      size: 1.3 + Math.random() * 1.5,
      angle: Math.random() * Math.PI,
      spin: 0.3,
      color: colors[Math.floor(Math.random() * colors.length)],
      confetti: false,
      spark: true,
    });
  }

  function shouldAnimate(now) {
    return !document.hidden && heroVisible && (!paused || now < manualUntil);
  }

  function stop() {
    if (frame) window.cancelAnimationFrame(frame);
    frame = 0;
    lastTime = 0;
  }

  function start() {
    if (!frame && shouldAnimate(performance.now())) {
      lastTime = 0;
      frame = window.requestAnimationFrame(draw);
    }
  }

  function draw(now) {
    frame = 0;
    if (!shouldAnimate(now)) {
      lastTime = 0;
      if (paused && manualUntil > 0 && now >= manualUntil) {
        manualUntil = 0;
        particles = [];
        context.clearRect(0, 0, width, height);
      }
      return;
    }
    const delta = lastTime ? Math.min((now - lastTime) / 1000, 0.035) : 1 / 60;
    lastTime = now;
    context.clearRect(0, 0, width, height);

    if (!paused) {
      activeTime += delta;
      if (!motionPreference.matches && nextOpening < openingBursts.length) {
        const opening = openingBursts[nextOpening];
        if (activeTime >= opening.time) {
          burst(width * opening.x, height * opening.y);
          nextOpening += 1;
        }
      }
      nextSpark -= delta;
      if (nextSpark <= 0) {
        driftingSpark();
        nextSpark = width < 640 ? 0.42 : 0.24;
      }
    }

    particles = particles.filter((particle) => particle.life < particle.duration);
    for (const particle of particles) {
      particle.life += delta;
      const progress = Math.min(particle.life / particle.duration, 1);
      particle.oldX = particle.x;
      particle.oldY = particle.y;
      if (!particle.spark) {
        const drag = Math.exp(-0.9 * delta);
        particle.vx *= drag;
        particle.vy = particle.vy * drag + 28 * delta;
      }
      particle.x += particle.vx * delta;
      particle.y += particle.vy * delta;
      particle.angle += particle.spin * delta;
      context.globalAlpha = particle.spark
        ? Math.sin(progress * Math.PI) * 0.45
        : (1 - progress) * 0.86;
      context.fillStyle = particle.color;
      context.strokeStyle = particle.color;
      context.lineWidth = particle.size * 0.7;

      if (particle.confetti) {
        context.save();
        context.translate(particle.x, particle.y);
        context.rotate(particle.angle);
        context.fillRect(-particle.size, -particle.size * 0.5, particle.size * 2.8, particle.size);
        context.restore();
      } else if (particle.spark) {
        const radius = particle.size * 1.9;
        context.beginPath();
        context.moveTo(particle.x - radius, particle.y);
        context.lineTo(particle.x + radius, particle.y);
        context.moveTo(particle.x, particle.y - radius);
        context.lineTo(particle.x, particle.y + radius);
        context.stroke();
      } else {
        context.beginPath();
        context.moveTo(particle.x - particle.vx * 0.045, particle.y - particle.vy * 0.045);
        context.lineTo(particle.x, particle.y);
        context.stroke();
        context.beginPath();
        context.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        context.fill();
      }
    }
    context.globalAlpha = 1;
    frame = window.requestAnimationFrame(draw);
  }

  celebrate?.addEventListener('click', () => {
    // A paused or reduced-motion page permits only this explicitly requested,
    // finite burst; it never turns continuous animation back on implicitly.
    if (paused) manualUntil = performance.now() + 1800;
    burst(width * 0.22, height * 0.35, 1.2);
    burst(width * 0.78, height * 0.27, 1.2);
    start();
  });

  motionToggle?.addEventListener('click', () => {
    paused = !paused;
    manualUntil = 0;
    syncControls();
    if (paused) stop();
    else start();
  });

  const preferenceChanged = () => {
    paused = motionPreference.matches;
    manualUntil = 0;
    syncControls();
    if (paused) {
      stop();
      particles = [];
      context.clearRect(0, 0, width, height);
    } else start();
  };
  if (motionPreference.addEventListener) {
    motionPreference.addEventListener('change', preferenceChanged);
  } else if (motionPreference.addListener) {
    motionPreference.addListener(preferenceChanged);
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
    else start();
  });
  window.addEventListener('pagehide', stop);
  window.addEventListener('pageshow', start);

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      heroVisible = entry.isIntersecting;
      if (heroVisible) start();
      else stop();
    }, { threshold: 0 }).observe(hero);
  }
  if ('ResizeObserver' in window) {
    new ResizeObserver(resize).observe(hero);
  } else {
    window.addEventListener('resize', resize, { passive: true });
  }

  resize();
  syncControls();
  start();
})();
