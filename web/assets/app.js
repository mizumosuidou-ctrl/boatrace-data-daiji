import { startWater } from "./water.js";
import { mountSim } from "./sim.js";

/* ------------------------------------------------------------ utils */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => (p == null ? "--" : (p * 100).toFixed(d));
const yen = (n) => (n == null ? "--" : "¥" + Number(n).toLocaleString("ja-JP"));
const boat = (b, size = "") => `<span class="boat ${size}" data-b="${b}" aria-label="${b}号艇">${b}</span>`;
const combo = (c, size = "sm") => c ? `<span class="combo" aria-label="3連単 ${esc(c)}">${c.split("-").map((b) => boat(b, size)).join("<i></i>")}</span>` : "";
const arrow = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 8h11M9 4l4 4-4 4"/></svg>`;
const BOAT_VARS = { 1: ["--b1", "--b1-ink"], 2: ["--b2", "--b2-ink"], 3: ["--b3", "--b3-ink"], 4: ["--b4", "--b4-ink"], 5: ["--b5", "--b5-ink"], 6: ["--b6", "--b6-ink"] };
const cVar = (b) => `--c: var(${BOAT_VARS[b][0]}); --ink: var(${BOAT_VARS[b][1]});`;

const deadlineMs = (date, hhmm) => {
  if (!hhmm) return NaN;
  return Date.parse(`${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)}T${hhmm}:00+09:00`);
};
// デモデータでは生成時刻を「いま」とみなし、読み込み後の経過時間だけ進める
let clockOffset = 0;
const nowMs = () => Date.now() + clockOffset;
const jstParts = (ms = nowMs()) => {
  const d = new Date(ms + 9 * 3600e3);
  return { y: d.getUTCFullYear(), m: d.getUTCMonth() + 1, d: d.getUTCDate(), hh: d.getUTCHours(), mm: d.getUTCMinutes(), ss: d.getUTCSeconds() };
};
const pad = (n) => String(n).padStart(2, "0");
const todayJst = () => { const p = jstParts(); return `${p.y}${pad(p.m)}${pad(p.d)}`; };
const fmtDate = (d) => `${d.slice(0, 4)}.${d.slice(4, 6)}.${d.slice(6, 8)}`;
const weekday = (d) => "日月火水木金土"[new Date(`${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}T12:00:00+09:00`).getDay()];
function fmtCountdown(ms) {
  if (!isFinite(ms)) return "--:--";
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
}
function statusOf(date, r, now = nowMs()) {
  if (r.cancelled) return "cancelled";
  if (r.result) return r.hit ? "finished hit" : "finished miss";
  const dl = deadlineMs(date, r.deadline);
  const left = dl - now;
  if (left <= 0) return "closed";
  if (left <= 5 * 60e3) return "imminent";
  if (left <= 30 * 60e3) return "soon";
  return "upcoming";
}

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 2600);
}

/* ------------------------------------------------------------ state */
const state = { latest: null, date: null, day: null, record: null, route: null, race: null, sim: null, firstPaint: {} };

async function loadLatest() {
  state.latest = await getJSON("data/latest.json");
  const dates = state.latest.dates || [state.latest.date];
  const today = todayJst();
  if (!state.date) state.date = dates.includes(today) ? today : state.latest.date;
  const sel = $("#dateSelect");
  sel.innerHTML = [...dates].reverse().map((d) => `<option value="${d}" ${d === state.date ? "selected" : ""}>${fmtDate(d)} (${weekday(d)})${d === today ? " TODAY" : ""}</option>`).join("");
  $("#demoBadge").hidden = !state.latest.demo;
  if (state.latest.demo && state.latest.generated_at) clockOffset = Date.parse(state.latest.generated_at) - Date.now();
}

async function loadDay(date = state.date) {
  state.day = await getJSON(`data/${date}/day.json`);
  $("#footerMeta").textContent = `UPDATED ${state.day.generated_at.replace("T", " ").slice(0, 16)} JST`;
  return state.day;
}

function allRaces(day = state.day) {
  const out = [];
  for (const v of day.venues) for (const r of v.races) out.push({ ...r, v });
  return out;
}

/* ------------------------------------------------------------ router */
function parseRoute() {
  const h = location.hash.replace(/^#\/?/, "");
  const parts = h.split("/").filter(Boolean);
  if (parts[0] === "race" && parts.length === 4) return { name: "race", date: parts[1], jcd: parts[2], rno: Number(parts[3]) };
  if (parts[0] === "record") return { name: "record" };
  if (parts[0] === "about") return { name: "about" };
  return { name: "home" };
}

async function route() {
  const r = parseRoute();
  const same = state.route && state.route.name === r.name && JSON.stringify(state.route) === JSON.stringify(r);
  state.route = r;
  $$(".nav a").forEach((a) => (a.dataset.nav === r.name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  if (state.sim) { state.sim.destroy(); state.sim = null; }
  if (!same) {
    $("#main").innerHTML = `<div class="loading"><div class="wave"><i></i><i></i><i></i><i></i><i></i></div>LOADING WATER SURFACE</div>`;
    window.scrollTo({ top: 0 });
  }
  try {
    if (r.name === "race") await renderRace(r);
    else if (r.name === "record") await renderRecord();
    else if (r.name === "about") renderAbout();
    else await renderHome();
    $("#main").focus({ preventScroll: true });
  } catch (err) {
    console.error(err);
    $("#main").innerHTML = `<div class="wrap"><div class="loading">データを読み込めませんでした<br><small>${esc(err.message)}</small></div></div>`;
  }
}

/* ------------------------------------------------------------ home */
function pickHero(races, now) {
  const upcoming = races.filter((r) => !r.result && deadlineMs(state.date, r.deadline) > now).sort((a, b) => deadlineMs(state.date, a.deadline) - deadlineMs(state.date, b.deadline));
  if (upcoming.length) return { mode: "next", race: upcoming[0], upcoming };
  const hits = races.filter((r) => r.hit).sort((a, b) => (b.payout || 0) - (a.payout || 0));
  return { mode: "recap", race: hits[0] || races[0], upcoming: [] };
}

function ribbon(win) {
  const segs = win.map((p, i) => {
    const b = i + 1;
    const wide = p > 0.14 ? "wide" : p < 0.085 ? "tiny" : "";  // 細い区画は％を隠して艇番だけ見せる
    return `<div class="ribbon-seg ${wide}" data-b="${b}" style="${cVar(b)} flex-grow:${Math.max(p, 0.02)}" title="${b}号艇 1着率 ${pct(p, 1)}%"><span class="boatno">${b}</span><span class="pct">${pct(p)}%</span></div>`;
  }).join("");
  return `<div class="ribbon"><div class="ribbon-bar" role="img" aria-label="各艇の1着確率">${segs}</div><div class="ribbon-legend"><span>Win probability</span><span>Model × Claude</span></div></div>`;
}

function ringSvg(value, size = 132) {
  const r = 54, c = 2 * Math.PI * r;
  const off = c * (1 - Math.max(0, Math.min(100, value)) / 100);
  return `<svg viewBox="0 0 132 132" aria-hidden="true">
    <defs><linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--accent-2)"/><stop offset="1" stop-color="var(--accent)"/></linearGradient></defs>
    <circle class="track" cx="66" cy="66" r="${r}" fill="none" stroke-width="8"/>
    <circle class="val" cx="66" cy="66" r="${r}" fill="none" stroke-width="8" stroke-dasharray="${c}" stroke-dashoffset="${c}" data-off="${off}"/>
  </svg>`;
}
function animateRings(root = document, instant = false) {
  if (instant) { $$(".ring .val", root).forEach((el) => { el.style.transition = "none"; el.style.strokeDashoffset = el.dataset.off; }); return; }
  requestAnimationFrame(() => requestAnimationFrame(() => $$(".ring .val", root).forEach((el) => (el.style.strokeDashoffset = el.dataset.off))));
}

const TIER_NOTE = { 鉄板: "データが一方向に揃った本命レース", 本線: "軸は明確、相手選びが鍵", 混戦: "上位拮抗、点数を絞りにくい", 波乱: "荒れる要素が多い一戦" };
const tierOf = (c) => (c >= 72 ? "鉄板" : c >= 55 ? "本線" : c >= 40 ? "混戦" : "波乱");

function heroHtml(h, now) {
  const r = h.race;
  if (!r) return "";
  const v = r.v;
  const link = `#/race/${state.date}/${v.jcd}/${r.rno}`;
  const dl = deadlineMs(state.date, r.deadline);
  const conf = r.confidence ?? 0;
  const tier = tierOf(conf);
  const label = h.mode === "next" ? `<span class="live-dot"></span> Next to close` : `Best hit of the day`;
  return `
  <section class="hero wrap">
    <div class="hero-grid">
      <article class="panel hero-main rv" style="--i:0">
        <div class="hero-top">
          <span class="eyebrow" style="gap:8px">${label}</span>
          ${v.grade && v.grade !== "一般" ? `<span class="chip grade-${esc(v.grade)}">${esc(v.grade)}</span>` : ""}
          ${v.is_nighter ? `<span class="chip nighter">NIGHTER</span>` : ""}
          <span class="chip">${esc(v.day_label)}</span>
        </div>
        <a href="${link}" class="hero-venue" aria-label="${esc(v.name)} ${r.rno}R の予想を見る">
          <span class="jp">${esc(v.name)}</span>
          <span class="rno">${r.rno}<small>R</small></span>
          <span class="hero-roman">${esc(v.roman)}</span>
        </a>
        <p class="hero-headline">${esc(r.headline)}</p>
        <div class="hero-meta"><span>${esc(v.title)}</span><span>${esc(r.race_name)}</span><span class="num">締切 ${esc(r.deadline)}</span></div>
        ${ribbon(r.win || [])}
      </article>
      <div class="hero-side">
        <div class="panel countdown-card rv" style="--i:1">
          <div class="countdown-label"><span class="eyebrow">${h.mode === "next" ? "Deadline in" : "Payout"}</span><span class="chip num">${esc(r.deadline)} JST</span></div>
          ${h.mode === "next"
            ? `<div class="countdown" data-deadline="${dl}" data-fmt="big" data-hero>${fmtCountdown(dl - now)}</div>
               <div class="countdown-sub"><span>本命 <b>${r.honmei ?? "-"}号艇</b></span><span>推奨 <b>${esc(r.top_pick || "-")}</b></span></div>
               <div class="progress"><i data-window="${dl}" style="transform:scaleX(0)"></i></div>`
            : `<div class="countdown" style="color:var(--hit)">${yen(r.payout)}</div>
               <div class="countdown-sub"><span>結果 <b>${esc(r.result || "-")}</b></span><span>推奨 <b>${esc(r.top_pick || "-")}</b></span></div>`}
          <div style="margin-top:22px"><a class="btn" href="${link}">Claudeの予想を見る ${arrow}</a></div>
        </div>
        <div class="panel confidence-card rv" style="--i:2">
          <div class="ring">${ringSvg(conf)}<div class="ring-center"><div><b>${conf}</b><span>CONFIDENCE</span></div></div></div>
          <div>
            <div class="eyebrow">Race type</div>
            <div class="tier" style="margin-top:8px">${tier}</div>
            <div class="tier-note">${TIER_NOTE[tier]}</div>
          </div>
        </div>
      </div>
    </div>
  </section>`;
}

function upcomingHtml(list, now) {
  if (!list.length) return "";
  const cards = list.slice(0, 10).map((r, i) => {
    const dl = deadlineMs(state.date, r.deadline);
    const bars = (r.win || []).map((p, j) => `<i style="${cVar(j + 1)} flex-grow:${p}"></i>`).join("");
    return `<a class="up-card rv" style="--i:${i + 3}" href="#/race/${state.date}/${r.v.jcd}/${r.rno}">
      <div class="top"><span class="v">${esc(r.v.name)}<small>${r.rno}R</small></span><span class="cd num" data-deadline="${dl}">${fmtCountdown(dl - now)}</span></div>
      <div class="hl">${esc(r.headline)}</div>
      <div class="bars">${bars}</div>
      <div class="foot">${combo(r.top_pick)}<span class="chip num">${r.confidence ?? "-"}%</span></div>
    </a>`;
  }).join("");
  return `<section class="section wrap">
    <div class="section-head"><div><span class="eyebrow">Up next</span><h2 class="section-title">Upcoming<small>締切が近い順に、Claudeの見出しと推奨買い目</small></h2></div></div>
    <div class="strip">${cards}</div>
  </section>`;
}

function cellHtml(v, r, now, isNext) {
  const st = statusOf(state.date, r, now);
  const dl = deadlineMs(state.date, r.deadline);
  const link = `#/race/${state.date}/${v.jcd}/${r.rno}`;
  const cls = `cell ${st}${isNext ? " next" : ""}`;
  const conf = r.confidence ?? 0;
  let body;
  if (st.startsWith("finished")) {
    body = `<div class="t"><span>${r.rno}R</span><b>${esc(r.deadline)}</b></div>
      <div class="res">${esc(r.result || "")}</div>
      <div class="pay">${yen(r.payout)}</div>${r.hit ? `<span class="stamp">HIT</span>` : ""}`;
  } else if (st === "cancelled") {
    body = `<div class="t"><span>${r.rno}R</span><b>中止</b></div>`;
  } else {
    const t = st === "soon" || st === "imminent" ? `<b data-deadline="${dl}">${fmtCountdown(dl - now)}</b>` : `<b>${esc(r.deadline)}</b>`;
    body = `<div class="t"><span>${r.rno}R</span>${t}</div>
      <div class="mid">${r.honmei ? boat(r.honmei, "sm") : ""}<span class="pick">${esc(r.top_pick || "")}</span></div>
      <div class="cbar"><i style="width:${conf}%"></i></div>`;
  }
  const label = `${v.name}${r.rno}R 締切${r.deadline} ${r.result ? "結果" + r.result : "本命" + (r.honmei || "")}`;
  return `<a class="${cls}" href="${link}" aria-label="${esc(label)}">${body}</a>`;
}

const FILTERS = [
  ["all", "ALL"],
  ["graded", "SG / G"],
  ["nighter", "NIGHTER"],
  ["hits", "HITS"],
];

function monitorHtml(day, now, filter) {
  let venues = day.venues;
  if (filter === "graded") venues = venues.filter((v) => v.grade && v.grade !== "一般");
  if (filter === "nighter") venues = venues.filter((v) => v.is_nighter);
  if (filter === "hits") venues = venues.filter((v) => v.races.some((r) => r.hit));
  const head = `<div class="mon-head" role="row"><div>VENUE</div>${Array.from({ length: 12 }, (_, i) => `<div>${i + 1}R</div>`).join("")}</div>`;
  const rows = venues.map((v, i) => {
    const byR = Object.fromEntries(v.races.map((r) => [r.rno, r]));
    const next = v.races.find((r) => !r.result && deadlineMs(state.date, r.deadline) > now);
    const cells = Array.from({ length: 12 }, (_, k) => {
      const r = byR[k + 1];
      return r ? cellHtml(v, r, now, next && next.rno === r.rno) : `<div class="cell empty"></div>`;
    }).join("");
    return `<div class="mon-row ${state.firstPaint.monitor ? "" : "rv"}" style="--i:${Math.min(i, 12) + 4}">
      <div class="mon-venue">
        <div class="name"><b>${esc(v.name)}</b><span>${esc(v.roman)}</span></div>
        <div class="tags">${v.grade && v.grade !== "一般" ? `<span class="chip grade-${esc(v.grade)}">${esc(v.grade)}</span>` : ""}${v.is_nighter ? `<span class="chip nighter">N</span>` : ""}<span class="chip">${esc(v.day_label)}</span></div>
        <div class="title">${esc(v.title)}</div>
      </div>
      <div class="mon-cells">${cells}</div>
    </div>`;
  }).join("");
  return `<div class="monitor" role="table" aria-label="全場レースモニター">${head}${rows || `<div class="mon-empty">該当する場はありません</div>`}</div>`;
}

function kpisHtml(t) {
  const hitRate = t.settled ? (t.hits / t.settled) * 100 : null;
  const honmei = t.settled ? (t.honmei_hits / t.settled) * 100 : null;
  const roi = t.stake ? (t.return / t.stake) * 100 : null;
  const f = (x) => (x == null ? "--" : x.toFixed(1));
  return `<div class="kpis">
    <div class="kpi"><b class="num">${t.settled}<span class="muted" style="font-size:.5em">/${t.races}</span></b><span>Settled</span></div>
    <div class="kpi hit"><b>${f(hitRate)}<small style="font-size:.5em">%</small></b><span>3連単 Hit</span></div>
    <div class="kpi"><b>${f(honmei)}<small style="font-size:.5em">%</small></b><span>本命 1着</span></div>
    <div class="kpi"><b>${f(roi)}<small style="font-size:.5em">%</small></b><span>回収率</span></div>
  </div>`;
}

async function renderHome(refresh = false) {
  if (!state.day || state.day.date !== state.date || refresh) await loadDay();
  const now = nowMs();
  const races = allRaces();
  const hero = pickHero(races, now);
  const filter = state.filter || "all";
  const main = $("#main");
  const html = `
    ${heroHtml(hero, now)}
    ${upcomingHtml(hero.upcoming.slice(1), now)}
    <section class="section wrap" id="monitor">
      <div class="section-head">
        <div><span class="eyebrow">${fmtDate(state.date)} · ${state.day.venues.length} venues</span><h2 class="section-title">Race Monitor<small>全場・全レースの本命と推奨、結果と的中をひと目で</small></h2></div>
        ${kpisHtml(state.day.totals)}
      </div>
      <div class="filters" role="group" aria-label="絞り込み">${FILTERS.map(([k, l]) => `<button class="filter" data-filter="${k}" aria-pressed="${k === filter}">${l}</button>`).join("")}</div>
      <div id="monitorBody">${monitorHtml(state.day, now, filter)}</div>
    </section>`;
  if (refresh && state.route.name === "home") {
    const y = scrollY;
    main.innerHTML = html.replaceAll(' rv"', '"').replaceAll(" rv ", " ");
    scrollTo({ top: y });
  } else {
    main.innerHTML = html;
  }
  state.firstPaint.monitor = true;
  scrollCellsToNow(main);
  animateRings(main, refresh);
  $$(".filter", main).forEach((b) => b.addEventListener("click", () => {
    state.filter = b.dataset.filter;
    $$(".filter", main).forEach((x) => x.setAttribute("aria-pressed", x === b));
    $("#monitorBody").innerHTML = monitorHtml(state.day, nowMs(), state.filter);
    scrollCellsToNow(main);
  }));
  tick();
}

// スマホでは各場の横スクロールを「いま」のレースに合わせる
function scrollCellsToNow(root) {
  for (const row of $$(".mon-cells", root)) {
    if (row.scrollWidth <= row.clientWidth) continue;
    const target = $(".cell.next", row) || $$(".cell.finished, .cell.closed", row).pop();
    if (target) row.scrollLeft = Math.max(0, target.getBoundingClientRect().left - row.getBoundingClientRect().left + row.scrollLeft - 96);
  }
}

/* ------------------------------------------------------------ race */
const WIND_DIR = (n) => (n ? ((n - 1) * 22.5) : null);
const FACTOR_KEYS_MODEL = [["skill", "選手力"], ["local", "当地"], ["motor", "モーター"], ["boat", "ボート"], ["start", "平均ST"], ["exhibition", "展示T"], ["exh_st", "展示ST"], ["flying", "F"], ["grade", "級別"], ["wind", "風"]];
const FACTOR_KEYS_ML = [["course", "コース"], ["start", "スタート力"], ["tenkai", "展開(ST順差)"], ["skill", "選手力"], ["local", "当地"], ["form", "調子"], ["racetime", "ﾚｰｽﾀｲﾑ"], ["motor", "モーター"], ["exhibition", "展示T"], ["exh_st", "展示ST"], ["original", "ｵﾘｼﾞﾅﾙ展示"], ["flying", "F"]];
const isML = (P) => String(P.engine || "").startsWith("lightgbm");
const factorKeys = (P) => (isML(P) ? FACTOR_KEYS_ML : FACTOR_KEYS_MODEL);
const ENGINE_LABEL = { "lightgbm-pre": "LightGBM · 展示前", "lightgbm-post": "LightGBM · 展示反映", model: "統計モデル" };

function factorBars(f, keys) {
  return `<div class="factors" aria-hidden="true">${keys.map(([k, label]) => {
    const v = f[k] || 0;
    const hgt = Math.min(13, Math.abs(v) * 30);
    return `<span class="factor" title="${label} ${v >= 0 ? "+" : ""}${v.toFixed(2)}">${v ? `<i class="${v > 0 ? "pos" : "neg"}" style="height:${hgt}px"></i>` : ""}</span>`;
  }).join("")}</div>`;
}

function rankClass(values, i, lowerBetter = false) {
  const vs = values.map((v, j) => [v, j]).filter(([v]) => v != null && v !== 0);
  if (vs.length < 3 || values[i] == null) return "";
  vs.sort((a, b) => (lowerBetter ? a[0] - b[0] : b[0] - a[0]));
  if (vs[0][1] === i) return "r1";
  if (vs[vs.length - 1][1] === i) return "r6";
  return "";
}

function sheetHtml(race) {
  const E = race.entries;
  const col = (k) => E.map((e) => e[k]);
  const cols = [
    ["全国勝率", "nat_win", false, 2], ["全国2連", "nat_2", false, 1], ["当地勝率", "loc_win", false, 2],
    ["モーター2連", "motor_2", false, 1], ...(E.some((e) => e.motor_kp != null) ? [["貢献P", "motor_kp", false, 2]] : []), ["ボート2連", "boat_2", false, 1], ["平均ST", "avg_st", true, 2],
    ...(E.some((e) => e.rt_series_rank != null) ? [["節ﾀｲﾑ順", "rt_series_rank", true, 0], ["節ﾍﾞｽﾄ", "rt_best", true, 1]] : []),
    ["展示T", "exhibition_time", true, 2], ["展示ST", "ex_st", true, 2],
    // オリジナル展示（場の公式サイト）。区間が場ごとに違うので、色付けはレース内の順位だけ
    ...[["一周", "lap_time", true, 2], ["まわり足", "turn_time", true, 2], ["直線", "straight_time", true, 2]].filter(([, k]) => E.some((e) => e[k] != null)),
    ["チルト", "tilt", null, 1], ["体重", "weight", null, 1],
  ];
  const fmt = (v, d) => (v == null || v === 0 && d === 2 && false ? "--" : typeof v === "number" ? v.toFixed(d) : "--");
  const rows = E.map((e, i) => `<tr>
    <td>${boat(e.boat, "sm")}</td>
    <td class="name">${esc(e.name)} <small class="muted num">${esc(e.toban)} ${esc(e.grade)}</small></td>
    ${cols.map(([, k, lb, d]) => {
      let v = e[k];
      let cls = lb === null ? "" : rankClass(col(k).map((x) => (k === "ex_st" && x != null ? Math.abs(x) : x)), i, lb);
      if (k === "ex_st" && v != null && v < 0) return `<td class="f">F${Math.abs(v).toFixed(2).slice(1)}</td>`;
      if (k === "rt_best" && v != null) return `<td class="${cls}">${Math.floor(v / 60)}'${String(Math.floor(v % 60)).padStart(2, "0")}"${Math.round((v * 10) % 10)}</td>`;
      if (k === "loc_win" && !v) return `<td class="muted">--</td>`;
      return `<td class="${cls}">${fmt(v, d)}</td>`;
    }).join("")}
    <td class="${e.f_count ? "f" : "muted"}">${e.f_count ? "F" + e.f_count : "-"}</td>
  </tr>`).join("");
  return `<div class="panel sheet"><table>
    <thead><tr><th>艇</th><th>選手</th>${cols.map(([l]) => `<th>${l}</th>`).join("")}<th>F</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function boardHtml(race) {
  const P = race.prediction;
  const E = Object.fromEntries(race.entries.map((e) => [e.boat, e]));
  const maxWin = Math.max(...P.boats.map((b) => b.win));
  const rows = P.boats.map((b) => {
    const e = E[b.boat] || {};
    return `<div class="board-row ${b.win === maxWin ? "top" : ""}">
      <div>${boat(b.boat, "lg")}</div>
      <div class="racer"><b>${esc(e.name)}<span class="g ${esc(e.grade)}">${esc(e.grade)}</span></b><small>${b.start_order != null ? `<span class="so">予想ST順 ${Number.isInteger(b.start_order) ? b.start_order : b.start_order.toFixed(1)}番手</span> · ` : ""}${esc(e.branch)} · ${e.age ?? "-"}歳 · ${b.course}コース${e.ex_course && e.ex_course !== e.boat ? " (進入変化)" : ""}</small></div>
      <div class="winbar" data-b="${b.boat}" style="${cVar(b.boat)}"><div class="track"><span class="fill" style="width:${(b.win / maxWin) * 100}%"></span></div><span class="v">${pct(b.win)}<small>%</small></span></div>
      <div class="num">${pct(b.top2)}%</div>
      <div class="num" data-l="3連対">${pct(b.top3)}%</div>
      <div>${factorBars(b.factors, factorKeys(P))}</div>
    </div>`;
  }).join("");
  return `<div class="panel board">
    <div class="board-head"><div>艇</div><div>選手</div><div>1着確率</div><div style="text-align:right">2連対</div><div style="text-align:right">3連対</div><div>要因（＋/−）</div></div>
    ${rows}
    <div class="factor-legend"><span class="chip ${isML(P) ? "src-claude" : ""}">${esc(ENGINE_LABEL[P.engine] || "統計モデル")}</span>${factorKeys(P).map(([, l], i) => `<span>${i + 1}.${l}</span>`).join("")}</div>
  </div>`;
}

function ticketsHtml(race) {
  const ai = race.ai || {};
  const odds = race.odds || {};
  const P = Object.fromEntries((race.prediction.trifecta || []).map((t) => [t.combo, t.p]));
  const modelPicks = Object.fromEntries((race.prediction.picks || []).map((p) => [p.combo, p]));
  const won = race.result && race.result.trifecta;
  const main = (ai.picks || []).map((p) => ({ combo: p.combo, weight: p.weight, kind: "本線" }));
  const value = (race.prediction.picks || []).filter((p) => p.kind === "妙味" && !main.some((m) => m.combo === p.combo)).map((p) => ({ combo: p.combo, kind: "妙味" }));
  const all = [...main, ...value];
  return `<div class="tickets">${all.map((t, i) => {
    const p = P[t.combo] ?? modelPicks[t.combo]?.p;
    const o = odds[t.combo];
    const ev = p && o ? p * o : null;
    return `<div class="ticket ${t.kind === "妙味" ? "value" : ""} ${won === t.combo ? "won" : ""} rv" style="--i:${i}">
      <div class="kind"><span>3連単 ${String(i + 1).padStart(2, "0")} · <b>${won === t.combo ? "HIT" : t.kind}</b></span>${t.weight ? `<span class="weight">${t.weight}<small>%</small></span>` : ""}</div>
      <div class="cmb">${t.combo.split("-").map((b) => boat(b)).join(`<span class="arrow"></span>`)}</div>
      <div class="stats"><div><span>PROB</span>${p != null ? pct(p, 1) + "%" : "--"}</div><div><span>ODDS</span>${o ?? "--"}</div><div><span>EV</span>${ev ? ev.toFixed(2) : "--"}</div></div>
    </div>`;
  }).join("")}</div>`;
}

function weatherHtml(w) {
  if (!w) return `<div class="weather"><div class="wx"><span>直前情報</span>展示後に更新</div></div>`;
  const deg = WIND_DIR(w.wind_dir);
  const windIcon = deg == null ? "" : `<svg viewBox="0 0 16 16" style="transform:rotate(${deg}deg)"><path d="M8 2v12M4 10l4 4 4-4"/></svg>`;
  return `<div class="weather">
    <div class="wx"><span>天候</span>${esc(w.weather || "-")}</div>
    <div class="wx">${windIcon}<span>風速</span>${w.wind_speed ?? "-"}m</div>
    <div class="wx"><span>波高</span>${w.wave_cm ?? "-"}cm</div>
    <div class="wx"><span>気温</span>${w.air_temp ?? "-"}℃</div>
    <div class="wx"><span>水温</span>${w.water_temp ?? "-"}℃</div>
  </div>`;
}

function resultHtml(race) {
  const r = race.result;
  if (!r) return "";
  if (r.cancelled) return `<div class="result-band"><div><div class="lbl">Result</div><b>レース中止</b></div></div>`;
  const s = race.settle || {};
  return `<div class="result-band ${s.trifecta_hit ? "hit" : ""} rv">
    <div><div class="lbl">Result · ${esc(r.kimarite || "")}</div><div class="result-order">${combo(r.trifecta, "")}</div></div>
    <div><div class="lbl">3連単払戻${r.popularity ? ` · ${r.popularity}番人気` : ""}</div><div class="result-pay">${yen(r.payout)}</div></div>
    <div>${s.trifecta_hit ? `<div class="hit-stamp">HIT</div>` : `<div class="miss-stamp">${s.honmei_win ? "本命1着" : "MISS"}</div>`}</div>
  </div>`;
}

const MARK_SYM = [["honmei", "◎", "本命"], ["taikou", "○", "対抗"], ["ana", "▲", "穴"]];

async function renderRace(r, refresh = false) {
  const race = await getJSON(`data/${r.date}/${r.jcd}-${String(r.rno).padStart(2, "0")}.json`);
  state.race = race;
  if (r.date !== state.date) { state.date = r.date; $("#dateSelect").value = r.date; }
  const ai = race.ai || {};
  const P = race.prediction;
  const E = Object.fromEntries(race.entries.map((e) => [e.boat, e]));
  const dl = deadlineMs(race.date, race.deadline);
  const now = nowMs();
  const done = !!race.result;
  const srcChip = ai.source === "claude" ? `<span class="chip src-claude">● CLAUDE${ai.model ? " · " + esc(ai.model) : ""}</span>` : ai.source === "demo" ? `<span class="chip">DEMO · MODEL TEXT</span>` : `<span class="chip">STATISTICAL MODEL</span>`;
  const stage = (race.stage === "exhibition" ? `<span class="chip">展示反映済</span>` : `<span class="chip">出走表段階</span>`)
    + (isML(race.prediction) ? `<span class="chip src-claude">LightGBM</span>` : "");
  race.__showResult = done;
  const prevNext = `
    <div style="display:flex;gap:8px;margin-top:26px;flex-wrap:wrap">
      ${r.rno > 1 ? `<a class="btn ghost" href="#/race/${r.date}/${r.jcd}/${r.rno - 1}">← ${r.rno - 1}R</a>` : ""}
      ${r.rno < 12 ? `<a class="btn ghost" href="#/race/${r.date}/${r.jcd}/${r.rno + 1}">${r.rno + 1}R →</a>` : ""}
    </div>`;
  const html = `
  <div class="wrap">
    <a class="back" href="#/"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M14 8H3M7 4L3 8l4 4"/></svg>Race monitor</a>
    <header class="race-hero">
      <div class="rv">
        <div class="race-title"><span class="jp">${esc(race.venue.name)}</span><span class="rno">${race.rno}<small>R</small></span><span class="roman">${esc(race.venue.roman)}</span></div>
        <div class="race-sub">
          <span class="t">${esc(race.title)}</span>
          ${race.grade && race.grade !== "一般" ? `<span class="chip grade-${esc(race.grade)}">${esc(race.grade)}</span>` : ""}
          <span class="chip">${esc(race.race_name)} ${race.distance}m</span>
          ${race.day_label ? `<span class="chip">${esc(race.day_label)}</span>` : ""}
          ${stage}
        </div>
        ${weatherHtml(race.weather)}
      </div>
      <div class="race-clock rv" style="--i:1">
        ${done ? `<div class="cd done">FINISHED</div>` : `<div class="cd" data-deadline="${dl}" data-fmt="big">${fmtCountdown(dl - now)}</div>`}
        <div class="lbl">締切 ${esc(race.deadline)} JST · ${fmtDate(race.date)}</div>
      </div>
    </header>
    ${resultHtml(race)}
    <div class="verdict-grid">
      <article class="panel verdict rv" style="--i:2">
        <div class="who"><span class="eyebrow">Claude's verdict</span>${srcChip}</div>
        <h2>${esc(ai.headline)}</h2>
        <p class="body">${esc(ai.verdict)}</p>
        <div class="marks">${MARK_SYM.map(([k, sym, label]) => ai[k] ? `<div class="mark"><span class="sym">${sym}</span>${boat(ai[k])}<span class="nm">${esc(E[ai[k]]?.name || "")}<small>${label}</small></span></div>` : "").join("")}</div>
        ${ai.key_points && ai.key_points.length ? `<ol class="points">${ai.key_points.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>` : ""}
        ${ai.risk ? `<p class="risk">RISK — ${esc(ai.risk)}</p>` : ""}
      </article>
      <div class="side-stack">
        <div class="panel confidence-card rv" style="--i:3">
          <div class="ring">${ringSvg(ai.confidence ?? P.confidence)}<div class="ring-center"><div><b>${ai.confidence ?? P.confidence}</b><span>CONFIDENCE</span></div></div></div>
          <div><div class="eyebrow">Race type</div><div class="tier" style="margin-top:8px">${tierOf(ai.confidence ?? P.confidence)}</div><div class="tier-note">モデル確信度 ${P.confidence} · ${esc(P.tier)}</div></div>
        </div>
        <div class="panel scenario rv" style="--i:4">
          <div class="eyebrow">Winning move · 決まり手予測</div>
          ${Object.entries(P.scenario).slice(0, 5).map(([k, v]) => `<div class="scen-row"><span>${esc(k)}</span><span class="bar"><i style="width:${v * 100}%"></i></span><span class="num">${pct(v)}%</span></div>`).join("")}
        </div>
      </div>
    </div>

    <section class="panel sim rv" style="--i:5">
      <div class="sim-head"><span class="eyebrow">${done ? "First turn · result replay" : "First turn simulation"}</span><button class="replay" id="replay" type="button"><svg viewBox="0 0 12 12" fill="currentColor"><path d="M3 1.5v9l7.5-4.5z"/></svg>Replay</button></div>
      <canvas class="sim-canvas" id="sim" role="img" aria-label="1周1マークの展開シミュレーション"></canvas>
      <div class="sim-foot"><span>進入 ${P.boats.slice().sort((a, b) => a.course - b.course).map((b) => b.boat).join("")} ${race.stage === "exhibition" ? "（展示進入）" : "（枠なり想定）"}</span><span>${isML(P) ? "予想スタート順" : "ST"}・予想着順から描画したイメージです</span></div>
    </section>

    <section class="section">
      <div class="section-head"><div><span class="eyebrow">Probability board</span><h2 class="section-title">Who wins<small>予想エンジンが出した1着・2連対・3連対確率と、その根拠</small></h2></div></div>
      ${boardHtml(race)}
    </section>

    <section class="section">
      <div class="section-head"><div><span class="eyebrow">Tickets</span><h2 class="section-title">Picks<small>Claudeの推奨買い目（配分％）と、オッズから見た妙味</small></h2></div></div>
      ${ticketsHtml(race)}
    </section>

    <section class="section">
      <div class="section-head"><div><span class="eyebrow">Data sheet</span><h2 class="section-title">The numbers<small>出走表・直前情報（● はレース内1位）</small></h2></div></div>
      ${sheetHtml(race)}
      ${prevNext}
    </section>
  </div>`;
  const main = $("#main");
  if (refresh) {
    const y = scrollY;
    main.innerHTML = html.replaceAll(" rv\"", "\"").replaceAll(" rv ", " ");
    scrollTo({ top: y });
  } else main.innerHTML = html;
  animateRings(main, refresh);
  const canvas = $("#sim");
  document.fonts.ready.then(() => {
    if (!canvas.isConnected) return;
    state.sim = mountSim(canvas, race);
    $("#replay").addEventListener("click", () => state.sim.play());
  });
  document.title = `${race.venue.name}${race.rno}R — MINAMO`;
  tick();
}

/* ------------------------------------------------------------ record */
async function renderRecord() {
  const [rec] = await Promise.all([getJSON("data/record.json")]);
  const dates = (state.latest.dates || []).slice(-14);
  const days = (await Promise.all(dates.map((d) => getJSON(`data/${d}/day.json`).catch(() => null)))).filter(Boolean);
  const T = rec.totals;
  const hitRate = T.settled ? (T.hits / T.settled) * 100 : 0;
  const roi = T.stake ? (T.return / T.stake) * 100 : 0;
  const honmei = T.settled ? (T.honmei_hits / T.settled) * 100 : 0;
  const tiers = { 鉄板: [0, 0, 0], 本線: [0, 0, 0], 混戦: [0, 0, 0], 波乱: [0, 0, 0] };
  let best = null;
  for (const d of days) for (const v of d.venues) for (const r of v.races) {
    if (!r.result || r.cancelled) continue;
    const t = tiers[tierOf(r.confidence ?? 0)];
    t[0]++; t[1] += r.honmei_win ? 1 : 0; t[2] += r.hit ? 1 : 0;
    if (r.hit && (!best || r.payout > best.payout)) best = { ...r, v, date: d.date };
  }
  const dd = rec.days.slice(-14);
  const W = 800, H = 260, pad = 36;
  const maxRoi = Math.max(150, ...dd.map((d) => (d.stake ? (d.return / d.stake) * 100 : 0)));
  const bw = (W - pad * 2) / Math.max(1, dd.length);
  const y = (v) => H - pad - (v / maxRoi) * (H - pad * 2);
  const bars = dd.map((d, i) => {
    const r = d.stake ? (d.return / d.stake) * 100 : 0;
    const bwid = Math.min(bw * 0.6, 44);
    const cx = pad + i * bw + bw * 0.5;
    return `<rect class="bar-ret ${r < 100 ? "lo" : ""}" x="${cx - bwid / 2}" y="${y(r)}" width="${bwid}" height="${H - pad - y(r)}" rx="3"><title>${fmtDate(d.date)} 回収率 ${r.toFixed(1)}%</title></rect>
      <text x="${cx}" y="${H - pad + 16}" text-anchor="middle">${d.date.slice(4, 6)}/${d.date.slice(6)}</text>
      <text x="${cx}" y="${y(r) - 6}" text-anchor="middle">${r.toFixed(0)}%</text>`;
  }).join("");
  const line = dd.map((d, i) => `${pad + i * bw + bw * 0.5},${y(d.settled ? (d.hits / d.settled) * 100 : 0)}`).join(" ");
  $("#main").innerHTML = `
  <div class="wrap">
    <section class="section">
      <span class="eyebrow">Track record · ${rec.days.length} days</span>
      <h1 class="section-title" style="font-size:clamp(48px,7vw,110px)">Record<small>すべての予想は締切前に公開し、結果と自動照合しています。${rec.demo ? "（現在はデモデータ）" : ""}</small></h1>
      <div class="rec-hero">
        <div class="panel rec-kpi gold rv"><span class="eyebrow">3連単 的中率</span><div class="v">${hitRate.toFixed(1)}<small>%</small></div><p>${T.hits} / ${T.settled} レース</p></div>
        <div class="panel rec-kpi rv" style="--i:1"><span class="eyebrow">回収率</span><div class="v">${roi.toFixed(1)}<small>%</small></div><p>推奨買い目を各100円で購入した場合</p></div>
        <div class="panel rec-kpi rv" style="--i:2"><span class="eyebrow">本命 1着率</span><div class="v">${honmei.toFixed(1)}<small>%</small></div><p>◎ が1着になった割合</p></div>
        <div class="panel rec-kpi rv" style="--i:3"><span class="eyebrow">最高払戻</span><div class="v" style="font-size:clamp(36px,4vw,56px)">${best ? yen(best.payout) : "--"}</div><p>${best ? `${fmtDate(best.date)} ${esc(best.v.name)}${best.rno}R ${esc(best.result)}` : "まだありません"}</p></div>
      </div>
      <div class="panel chart rv" style="--i:4">
        <div class="section-head" style="margin:0 0 8px"><span class="eyebrow">Daily return rate / hit rate</span><span class="chip">■ 回収率 ― 的中率</span></div>
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="日別回収率">
          <line class="ref" x1="${pad}" x2="${W - pad}" y1="${y(100)}" y2="${y(100)}"/>
          <text x="${W - pad}" y="${y(100) - 6}" text-anchor="end">100%</text>
          ${bars}
          <polyline class="line" points="${line}"/>
          <line class="axis" x1="${pad}" x2="${W - pad}" y1="${H - pad}" y2="${H - pad}"/>
        </svg>
      </div>
      ${T.ml_races ? `<div class="section-head" style="margin-top:40px"><div><span class="eyebrow">Engine duel</span><h2 class="section-title">LightGBM vs 統計モデル<small>同じレースで、それぞれの本命（1着確率1位）が1着になった割合</small></h2></div></div>
      <div class="calib">
        <div class="panel rv"><h4>LightGBM</h4><div class="big" style="color:var(--accent)">${((T.ml_fav_hits / T.ml_races) * 100).toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">本命1着 · ${T.ml_races}R</div></div>
        <div class="panel rv" style="--i:1"><h4>統計モデル</h4><div class="big">${((T.shadow_fav_hits / T.ml_races) * 100).toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">本命1着 · ${T.ml_races}R</div></div>
      </div>` : ""}
      <div class="section-head" style="margin-top:40px"><div><span class="eyebrow">Calibration</span><h2 class="section-title">By confidence<small>確信度の帯ごとの成績。数字が高いレースほど当たっているかを検証</small></h2></div></div>
      <div class="calib">${Object.entries(tiers).map(([k, [n, h1, h3]], i) => `<div class="panel rv" style="--i:${i}"><h4>${k}</h4><div class="big">${n ? ((h3 / n) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">3連単的中 · 本命1着 ${n ? ((h1 / n) * 100).toFixed(1) : "--"}% · ${n}R</div></div>`).join("")}</div>
    </section>
  </div>`;
}

/* ------------------------------------------------------------ about */
function renderAbout() {
  $("#main").innerHTML = `
  <div class="wrap">
    <section class="section">
      <span class="eyebrow">About MINAMO</span>
      <p class="manifesto rv">水面は、<em>数字</em>でできている。</p>
      <div class="about">
        <div class="panel rv" style="--i:1"><div class="step">01</div><h3>取得する</h3><p>毎朝、BOAT RACE公式サイトから開催場と全レースの出走表を取得。締切30分前からは展示タイム・展示進入・スタート展示・気象・3連単オッズを数分おきに取り直します。アクセスは1秒1回以下に抑えています。</p></div>
        <div class="panel rv" style="--i:2"><div class="step">02</div><h3>計算する</h3><p>場ごとのコース別1着率を土台に、勝率・当地勝率・モーター/ボート2連率・平均ST・F持ち・展示タイム・展示ST・風を加点減点。1着確率から3連単120通りの確率を算出し、オッズと掛けて期待値を出します。</p></div>
        <div class="panel rv" style="--i:3"><div class="step">03</div><h3>Claudeが読む</h3><p>数字とデータをClaudeが読み解き、展開・本命・対抗・穴・買い目と配分を決定。レース後は結果を自動取得して照合し、的中率と回収率をそのまま公開します。</p></div>
      </div>
      <div class="panel disclaimer rv" style="--i:4">
        本サイトの予想は統計モデルとAIによる参考情報であり、的中や払戻を保証するものではありません。舟券の購入は20歳以上の方が、ご自身の判断と責任で行ってください。データの著作権はそれぞれの権利者に帰属します。公式情報は必ずBOAT RACE公式サイトでご確認ください。
      </div>
    </section>
  </div>`;
}

/* ------------------------------------------------------------ live tick */
function tick() {
  const now = nowMs();
  const p = jstParts(now);
  $("#clock").textContent = `${pad(p.hh)}:${pad(p.mm)}:${pad(p.ss)}`;
  for (const el of $$("[data-deadline]")) {
    const left = Number(el.dataset.deadline) - now;
    const txt = left <= 0 ? "CLOSED" : fmtCountdown(left);
    if (el.dataset.fmt === "big" && left > 0) {
      const [a, b, c] = txt.split(":");
      el.innerHTML = c ? `${a}<span class="sep">:</span>${b}<span class="sep">:</span>${c}` : `${a}<span class="sep">:</span>${b}`;
    } else el.textContent = txt;
    el.classList.toggle("imminent", left > 0 && left <= 5 * 60e3);
    if (left <= 0 && el.hasAttribute("data-hero") && state.route?.name === "home" && !tick.rolling) {
      // 締切を過ぎたら次のレースへ切り替える
      tick.rolling = true;
      setTimeout(() => renderHome(true).finally(() => (tick.rolling = false)), 1200);
    }
  }
  for (const el of $$("[data-window]")) {
    const left = Number(el.dataset.window) - now;
    el.style.transform = `scaleX(${Math.max(0, Math.min(1, 1 - left / (30 * 60e3)))})`;
  }
}

/* ------------------------------------------------------------ boot */
function initTheme() {
  const saved = (() => { try { return localStorage.getItem("minamo-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeToggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("minamo-theme", next); } catch { /* 保存できない環境でも表示は切替 */ }
  });
}

async function boot() {
  initTheme();
  startWater($("#water"));
  tick();
  setInterval(tick, 1000);
  try {
    await loadLatest();
  } catch (e) {
    $("#main").innerHTML = `<div class="wrap"><div class="loading">データがまだありません<br><small>python -m minamo sync または demo を実行してください</small></div></div>`;
    return;
  }
  $("#dateSelect").addEventListener("change", (e) => {
    state.date = e.target.value;
    state.day = null;
    if (location.hash && location.hash !== "#/") location.hash = "#/";
    else route();
  });
  addEventListener("hashchange", route);
  await route();

  // 定期更新：一覧は60秒、レース詳細は締切前のみ30秒
  setInterval(async () => {
    if (document.hidden) return;
    try {
      if (state.route.name === "home" && state.date === todayJst()) await renderHome(true);
      else if (state.route.name === "race" && state.race && !state.race.result && nowMs() > deadlineMs(state.race.date, state.race.deadline) - 35 * 60e3) {
        const before = state.race.updated_at;
        await renderRace(state.route, true);
        if (state.race.updated_at !== before) toast("直前情報を反映しました");
      }
    } catch (e) { console.warn(e); }
  }, 30000);
}

boot();
