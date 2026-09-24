(() => {
  'use strict';

  const root = document.documentElement;
  const hero = document.getElementById('opportunities');
  const research = document.getElementById('research');
  const main = research?.parentElement;
  const close = hero?.querySelector('.intro-close');
  if (!hero || !research || !main || !close) return;

  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const navigation = performance.getEntriesByType('navigation')[0];
  const runway = document.createElement('div');
  runway.className = 'intro-runway';
  runway.setAttribute('aria-hidden', 'true');
  main.append(runway);
  root.classList.add('intro-floating');
  hero.hidden = false;
  hero.inert = true;

  let distance = 0;
  let speed = preference.matches ? 1 : 0.7;
  let ready = false;
  let frame = 0;
  let entranceTimer = 0;

  function scrollInstantly(top) {
    const previousBehavior = root.style.scrollBehavior;
    root.style.scrollBehavior = 'auto';
    window.scrollTo(0, top);
    root.style.scrollBehavior = previousBehavior;
  }

  function reveal() {
    if (ready) return;
    ready = true;
    window.clearTimeout(entranceTimer);
    root.classList.add('intro-ready');
    render();
  }

  function render() {
    frame = 0;
    const position = Math.max(0, window.scrollY);
    const active = position < distance;
    root.classList.toggle('intro-active', active);
    hero.style.setProperty('--intro-shift', `${-Math.min(position, distance) * speed}px`);
    const visible = ready && active;
    if (!visible && hero.contains(document.activeElement)) {
      research.focus({ preventScroll: true });
    }
    hero.inert = !visible;
    hero.setAttribute('aria-hidden', String(!visible));
  }

  function measure() {
    const previousDistance = distance;
    const position = Math.max(0, window.scrollY);
    speed = preference.matches ? 1 : 0.7;
    const height = hero.offsetHeight;
    const viewportHeight = window.innerHeight;
    const top = Math.max(16, Math.min(220, viewportHeight * 0.24, viewportHeight - height - 24));
    hero.style.setProperty('--intro-top', `${top}px`);
    distance = Math.ceil((top + height + 48) / speed);
    // Keep this space on every visit, including refreshes. The scroll position
    // alone determines whether the introduction is on screen, in both directions.
    runway.style.height = `${distance}px`;
    if (previousDistance && previousDistance !== distance && position > 0) {
      const nextPosition = position >= previousDistance
        ? distance + position - previousDistance
        : distance * position / previousDistance;
      scrollInstantly(nextPosition);
    }
    render();
  }

  function onScroll() {
    if (!frame) frame = window.requestAnimationFrame(render);
  }

  function skipIntro() {
    // Closing advances past the card; scrolling back still brings it back.
    reveal();
    scrollInstantly(distance);
    render();
  }

  function anchorTarget(hash) {
    try {
      return document.getElementById(decodeURIComponent(hash.slice(1)));
    } catch (_) {
      return null;
    }
  }

  function goToAnchor(hash) {
    const target = anchorTarget(hash);
    if (!target) return;
    reveal();
    if (target === hero || hero.contains(target)) {
      scrollInstantly(0);
    } else if (target === research || research.contains(target)) {
      // Native anchor positioning alone misses the research's sticky runway.
      const offset = target.getBoundingClientRect().top - research.getBoundingClientRect().top;
      const padding = parseFloat(getComputedStyle(root).scrollPaddingTop) || 0;
      scrollInstantly(distance + Math.max(0, offset - padding));
    }
    render();
    if (target === research) research.focus({ preventScroll: true });
  }

  close.addEventListener('click', skipIntro);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && root.classList.contains('intro-active')) skipIntro();
  });
  document.addEventListener('click', (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const anchor = event.target.closest?.('a[href^="#"]');
    if (!anchor) return;
    const hash = anchor.getAttribute('href');
    if (!anchorTarget(hash)) return;
    event.preventDefault();
    if (location.hash !== hash) history.pushState(history.state, '', hash);
    goToAnchor(hash);
  });
  window.addEventListener('hashchange', () => {
    if (location.hash) goToAnchor(location.hash);
    else onScroll();
  });
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', measure, { passive: true });
  window.addEventListener('pageshow', (event) => {
    measure();
    // Fresh links go to their section; reload and history keep restored scroll.
    if (!event.persisted && navigation?.type === 'navigate' && location.hash) {
      goToAnchor(location.hash);
    }
    if (window.scrollY > 0) reveal();
  });
  if (preference.addEventListener) preference.addEventListener('change', measure);
  else if (preference.addListener) preference.addListener(measure);
  if ('ResizeObserver' in window) new ResizeObserver(measure).observe(hero);
  measure();
  entranceTimer = window.setTimeout(reveal, preference.matches ? 0 : 650);
})();
