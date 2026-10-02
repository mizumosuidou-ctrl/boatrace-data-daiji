// 1周1マークの展開シミュレーション。
// 予想の着順（本命の3連単＋残りは1着率順）どおりに、スタートからターンまでを描く。
const COLORS = { 1: "#f4f6f8", 2: "#20242b", 3: "#e8333a", 4: "#2172e0", 5: "#f6d21c", 6: "#1ba35a" };
const INK = { 1: "#0b0d10", 2: "#ffffff", 3: "#ffffff", 4: "#ffffff", 5: "#1b1700", 6: "#ffffff" };
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const ease = (x) => (x < 0 ? 0 : x > 1 ? 1 : x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * t;

export function mountSim(canvas, race, opts = {}) {
  let onDone = opts.onDone;
  const ctx = canvas.getContext("2d");
  const pred = race.prediction;
  const boats = pred.boats.map((b) => b.boat);
  const byBoat = Object.fromEntries(pred.boats.map((b) => [b.boat, b]));
  const entries = Object.fromEntries(race.entries.map((e) => [e.boat, e]));
  const top = (race.result && race.result.order && race.result.order.length >= 3 && race.__showResult)
    ? race.result.order.slice(0, 3)
    : ((race.ai && race.ai.picks && race.ai.picks[0]) ? race.ai.picks[0].combo : pred.trifecta[0].combo).split("-").map(Number);
  const rest = boats.filter((b) => !top.includes(b)).sort((a, b) => byBoat[b].win - byBoat[a].win);
  const order = [...top, ...rest];
  const rank = Object.fromEntries(order.map((b, i) => [b, i]));

  let w = 0, h = 0, dpr = 1, raf = 0, start = 0, trails = {};
  const DURATION = 6200;

  function css(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  function resize() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    w = r.width; h = r.height;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // レイアウト（キャンバス座標）
  function geo() {
    const narrow = w < 620;
    const laneGap = (h * (narrow ? 0.5 : 0.52)) / 6;
    const mark = { x: w * (narrow ? 0.82 : 0.8), y: h * 0.38 };
    const laneTop = mark.y + laneGap * 0.55;
    const startX = w * (narrow ? 0.24 : 0.3);
    return { mark, laneTop, laneGap, startX, r0: laneGap * 0.8, narrow };
  }

  // u: 0 = スタートライン, 0.45 = マーク手前, 0.72 = ターン出口, 以降はバックストレッチ
  const A_END = 0.45, B_END = 0.72;
  function lanePos(course, u, g) {
    const yLane = g.laneTop + (course - 0.5) * g.laneGap;
    const rIn = g.r0 + (course - 1) * g.laneGap * 0.42;
    const rOut = g.r0 + (course - 1) * g.laneGap * 0.1;
    if (u <= A_END) {
      const k = u / A_END;
      const x = g.startX + k * (g.mark.x - g.startX);
      const y = u < 0 ? yLane : lerp(yLane, g.mark.y + rIn, Math.pow(k, 2.4));
      const dy = u < 0 ? 0 : (g.mark.y + rIn - yLane) * 2.4 * Math.pow(Math.max(k, 1e-3), 1.4) / A_END / (g.mark.x - g.startX) * A_END;
      return { x, y, a: Math.atan(dy) };
    }
    if (u <= B_END) {
      const k = (u - A_END) / (B_END - A_END);
      const th = Math.PI / 2 - k * Math.PI;
      const r = lerp(rIn, rOut, ease(k));
      return { x: g.mark.x + Math.cos(th) * r, y: g.mark.y + Math.sin(th) * r, a: th - Math.PI / 2 };
    }
    const k = (u - B_END) / (1 - B_END);
    return { x: g.mark.x - k * (g.mark.x - w * 0.34), y: g.mark.y - rOut, a: Math.PI };
  }

  function progress(b, t) {
    const e = entries[b] || {};
    // 予想スタート順があればそれで隊形を描く（1番手ほど前）
    const so = byBoat[b].start_order;
    const st = so != null ? 0.09 + (so - 1) * 0.022 : Math.abs(e.ex_st ?? e.avg_st ?? 0.16);
    const r = rank[b];
    // 助走 → スタート（STが速いほど前）→ 着順どおりに差が開く
    const launch = -0.22 - st * 0.9;
    const finish = 1.02 - r * 0.075;
    const p = ease(t);
    const mid = 0.42 - st * 0.35 - r * 0.012;
    if (t < 0.45) return lerp(launch, mid, ease(t / 0.45));
    return lerp(mid, finish, ease((t - 0.45) / 0.55) * 0.5 + p * 0.5);
  }

  function drawBoat(b, pos, g) {
    ctx.save();
    ctx.translate(pos.x, pos.y);
    ctx.rotate(pos.a);
    const L = Math.max(22, g.laneGap * 0.9), H = Math.max(12, g.laneGap * 0.5);
    ctx.fillStyle = COLORS[b];
    ctx.strokeStyle = b === 2 ? "rgba(255,255,255,.55)" : "rgba(0,0,0,.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(L / 2, 0);
    ctx.quadraticCurveTo(L / 2 - 4, -H / 2, 0, -H / 2);
    ctx.lineTo(-L / 2, -H / 2);
    ctx.lineTo(-L / 2, H / 2);
    ctx.lineTo(0, H / 2);
    ctx.quadraticCurveTo(L / 2 - 4, H / 2, L / 2, 0);
    ctx.fill();
    ctx.stroke();
    ctx.rotate(-pos.a);
    ctx.fillStyle = INK[b];
    ctx.font = `800 ${Math.round(H * 0.95)}px "Big Shoulders Display", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(b), 0, 1);
    ctx.restore();
  }

  function frame(now) {
    if (!start) start = now;
    const t = Math.min(1, (now - start) / DURATION);
    const g = geo();
    const line = css("--line-2", "rgba(190,215,255,.18)");
    const muted = css("--muted", "#7d8aa0");
    const accent = css("--accent", "#7cf5d0");
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    ctx.lineCap = "butt";

    // コース線
    ctx.strokeStyle = line;
    ctx.setLineDash([2, 8]);
    for (let c = 0; c <= 6; c++) {
      const y = g.laneTop + c * g.laneGap;
      ctx.beginPath(); ctx.moveTo(w * 0.04, y); ctx.lineTo(g.mark.x - g.laneGap, y); ctx.stroke();
    }
    ctx.setLineDash([]);
    // スタートライン
    ctx.strokeStyle = accent;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(g.startX, g.laneTop); ctx.lineTo(g.startX, g.laneTop + g.laneGap * 6); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = muted;
    ctx.font = `500 10px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.fillText("スタート", g.startX, g.laneTop + g.laneGap * 6.2 + 14);
    // ターンマーク
    ctx.fillStyle = "#ff7a2f";
    ctx.beginPath(); ctx.arc(g.mark.x, g.mark.y, Math.max(6, g.laneGap * 0.22), 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(255,122,47,.35)";
    ctx.lineWidth = 1;
    for (let k = 1; k <= 2; k++) {
      ctx.beginPath(); ctx.arc(g.mark.x, g.mark.y, g.laneGap * (0.4 + k * 0.35 + ((now / 1600) % 1) * 0.3), 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = muted;
    ctx.fillText("1M", g.mark.x, g.mark.y - g.laneGap * 0.6);
    // ST表示
    ctx.textAlign = "right";
    for (const b of boats) {
      const c = byBoat[b].course;
      const e = entries[b] || {};
      const st = e.ex_st != null ? e.ex_st : e.avg_st;
      const y = g.laneTop + (c - 0.5) * g.laneGap;
      ctx.fillStyle = muted;
      ctx.fillText(`${g.narrow ? "" : c + "C  "}${st == null ? "--" : (st < 0 ? "F" : "") + Math.abs(st).toFixed(2).replace(/^0/, "")}`, g.startX - 10, y + 3);
    }

    // 艇と航跡
    const drawn = [];
    for (const b of boats) {
      const u = progress(b, t);
      const pos = lanePos(byBoat[b].course, u, g);
      const tr = (trails[b] ||= []);
      tr.push({ x: pos.x, y: pos.y });
      if (tr.length > 46) tr.shift();
      drawn.push({ b, pos, u });
    }
    for (const { b } of drawn) {
      const tr = trails[b];
      for (let i = 1; i < tr.length; i++) {
        ctx.strokeStyle = b === 2 ? `rgba(160,170,190,${(i / tr.length) * 0.5})` : hexA(COLORS[b], (i / tr.length) * 0.55);
        ctx.lineWidth = (i / tr.length) * Math.max(3, g.laneGap * 0.28);
        ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(tr[i - 1].x, tr[i - 1].y); ctx.lineTo(tr[i].x, tr[i].y); ctx.stroke();
      }
    }
    drawn.sort((a, b) => a.u - b.u).forEach(({ b, pos }) => drawBoat(b, pos, g));

    // 着順表示
    if (t > 0.86) {
      const alpha = Math.min(1, (t - 0.86) / 0.1);
      ctx.globalAlpha = alpha;
      ctx.textAlign = "left";
      ctx.font = `800 ${Math.max(16, h * 0.07)}px "Big Shoulders Display", sans-serif`;
      ctx.fillStyle = css("--text", "#eaf0f7");
      ctx.fillText(order.slice(0, 3).join(" - "), w * 0.04, h * 0.14);
      ctx.font = `500 10px "JetBrains Mono", monospace`;
      ctx.fillStyle = muted;
      ctx.fillText(race.__showResult ? "RESULT" : "PREDICTED ORDER AT 1ST TURN", w * 0.04, h * 0.14 - Math.max(16, h * 0.07) - 2);
      ctx.globalAlpha = 1;
    }

    if (t < 1) raf = requestAnimationFrame(frame);
    else onDone && onDone();
  }

  let done = false;
  function renderFinal() {
    trails = {};
    // 航跡を描くため終盤の数コマを早回しで描く
    for (let i = 40; i >= 0; i--) {
      start = performance.now() - DURATION * (1 - i * 0.004);
      cancelAnimationFrame(raf);
      frame(performance.now());
    }
    cancelAnimationFrame(raf);
  }
  function play() {
    cancelAnimationFrame(raf);
    trails = {};
    start = 0;
    done = false;
    resize();
    if (reduce) { renderFinal(); done = true; return; }
    raf = requestAnimationFrame(frame);
  }
  onDone = ((cb) => () => { done = true; cb && cb(); })(onDone);

  // 画面に入る前もスタート前の隊形を描いておく
  resize();
  start = performance.now();
  frame(start);
  cancelAnimationFrame(raf);
  trails = {};
  let started = false;
  const ro = new ResizeObserver(() => { resize(); if (done) renderFinal(); });
  ro.observe(canvas);
  const io = new IntersectionObserver((es) => {
    if (es[0].isIntersecting && !started) { started = true; play(); io.disconnect(); }
  }, { threshold: 0.3 });
  io.observe(canvas);
  return { play, destroy() { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); } };
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}
