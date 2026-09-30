// 背景の水面：等高線のように揺れる細線。カーソル位置で波紋が立つ。
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function startWater(canvas) {
  const ctx = canvas.getContext("2d");
  let w = 0, h = 0, dpr = 1, lines = 0, raf = 0, t0 = performance.now();
  const mouse = { x: -9999, y: -9999, tx: -9999, ty: -9999, energy: 0 };
  const ripples = [];

  function color() {
    const s = getComputedStyle(document.documentElement);
    return {
      a: s.getPropertyValue("--water-a").trim() || "124,245,208",
      b: s.getPropertyValue("--water-b").trim() || "63,214,255",
      alpha: parseFloat(s.getPropertyValue("--water-alpha")) || 0.13,
    };
  }
  let pal = color();

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    lines = Math.round(Math.min(46, Math.max(22, h / 22)));
  }

  function frame(now) {
    const t = (now - t0) / 1000;
    mouse.x += (mouse.tx - mouse.x) * 0.08;
    mouse.y += (mouse.ty - mouse.y) * 0.08;
    mouse.energy *= 0.96;
    ctx.clearRect(0, 0, w, h);
    const step = w < 700 ? 14 : 10;
    for (let i = 0; i < lines; i++) {
      const p = i / (lines - 1);
      const baseY = h * (0.08 + p * 0.98);
      const depth = 0.35 + 0.65 * Math.sin(Math.PI * p);
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      const a = pal.alpha * depth;
      grad.addColorStop(0, `rgba(${pal.b},${a * 0.2})`);
      grad.addColorStop(0.45, `rgba(${pal.a},${a})`);
      grad.addColorStop(1, `rgba(${pal.b},${a * 0.35})`);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = -step; x <= w + step; x += step) {
        let y = baseY
          + Math.sin(x * 0.0042 + t * 0.35 + i * 0.42) * 10 * depth
          + Math.sin(x * 0.0113 - t * 0.52 + i * 0.9) * 4.5
          + Math.cos(x * 0.0021 + t * 0.18 - i * 0.27) * 16 * depth;
        const dx = x - mouse.x, dy = baseY - mouse.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 90000) {
          const d = Math.sqrt(d2);
          y += Math.sin(d * 0.05 - t * 5) * (1 - d / 300) * 10 * (0.25 + mouse.energy);
        }
        for (const r of ripples) {
          const rd = Math.hypot(x - r.x, baseY - r.y);
          const band = rd - r.age * 260;
          if (band > -60 && band < 60) y += Math.sin(band * 0.12) * 9 * (1 - r.age / 2.2) * (1 - Math.abs(band) / 60);
        }
        x === -step ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    for (let i = ripples.length - 1; i >= 0; i--) {
      ripples[i].age += 1 / 60;
      if (ripples[i].age > 2.2) ripples.splice(i, 1);
    }
    raf = requestAnimationFrame(frame);
  }

  resize();
  addEventListener("resize", resize);
  addEventListener("pointermove", (e) => {
    mouse.tx = e.clientX; mouse.ty = e.clientY;
    if (mouse.x < -5000) { mouse.x = e.clientX; mouse.y = e.clientY; }
    mouse.energy = Math.min(1, mouse.energy + 0.04);
  }, { passive: true });
  addEventListener("pointerdown", (e) => {
    if (ripples.length < 4) ripples.push({ x: e.clientX, y: e.clientY, age: 0 });
  }, { passive: true });
  new MutationObserver(() => { pal = color(); if (reduce) frame(performance.now()), cancelAnimationFrame(raf); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => { pal = color(); });

  if (reduce) { frame(performance.now()); cancelAnimationFrame(raf); return; }
  raf = requestAnimationFrame(frame);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cancelAnimationFrame(raf);
    else raf = requestAnimationFrame(frame);
  });
}
