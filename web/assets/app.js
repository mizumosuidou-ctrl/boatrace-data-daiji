import { startWater } from "./water.js";
import { mountSim } from "./sim.js";

/* ------------------------------------------------------------ utils */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => (p == null ? "--" : (p * 100).toFixed(d));
const yen = (n) => (n == null ? "--" : "¥" + Number(n).toLocaleString("ja-JP"));
const signedYen = (n) => (n == null ? "--" : (n < 0 ? "−" : "+") + "¥" + Math.abs(Number(n)).toLocaleString("ja-JP"));
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
  sel.innerHTML = [...dates].reverse().map((d) => `<option value="${d}" ${d === state.date ? "selected" : ""}>${fmtDate(d)} (${weekday(d)})${d === today ? " 今日" : ""}</option>`).join("");
  $("#demoBadge").hidden = !state.latest.demo;
  if (state.latest.demo && state.latest.generated_at) clockOffset = Date.parse(state.latest.generated_at) - Date.now();
}

async function loadDay(date = state.date) {
  state.day = await getJSON(`data/${date}/day.json`);
  $("#footerMeta").textContent = `更新 ${state.day.generated_at.replace("T", " ").slice(0, 16)}`;
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
  if (parts[0] === "picks") return { name: "picks" };
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
    $("#main").innerHTML = `<div class="loading"><div class="wave"><i></i><i></i><i></i><i></i><i></i></div>読み込み中…</div>`;
    window.scrollTo({ top: 0 });
  }
  try {
    if (r.name === "race") await renderRace(r);
    else if (r.name === "record") await renderRecord();
    else if (r.name === "picks") await renderPicks();
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
  return `<div class="ribbon"><div class="ribbon-bar" role="img" aria-label="各艇の1着確率">${segs}</div><div class="ribbon-legend"><span>1着確率</span><span>予想エンジン</span></div></div>`;
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
  const label = h.mode === "next" ? `<span class="live-dot"></span> まもなく締切` : `本日の最高払戻`;
  return `
  <section class="hero wrap">
    <div class="hero-grid">
      <article class="panel hero-main rv" style="--i:0">
        <div class="hero-top">
          <span class="eyebrow" style="gap:8px">${label}</span>
          ${v.grade && v.grade !== "一般" ? `<span class="chip grade-${esc(v.grade)}">${esc(v.grade)}</span>` : ""}
          ${v.is_nighter ? `<span class="chip nighter">ナイター</span>` : ""}
          <span class="chip">${esc(v.day_label)}</span>
        </div>
        <a href="${link}" class="hero-venue" aria-label="${esc(v.name)} ${r.rno}R の予想を見る">
          <span class="jp">${esc(v.name)}</span>
          <span class="rno">${r.rno}<small>R</small></span>
          
        </a>
        <p class="hero-headline">${esc(r.headline)}</p>
        <div class="hero-meta"><span>${esc(v.title)}</span><span>${esc(r.race_name)}</span><span class="num">締切 ${esc(r.deadline)}</span></div>
        ${ribbon(r.win || [])}
      </article>
      <div class="hero-side">
        <div class="panel countdown-card rv" style="--i:1">
          <div class="countdown-label"><span class="eyebrow">${h.mode === "next" ? "締切まで" : "払戻"}</span><span class="chip num">${esc(r.deadline)} 締切</span></div>
          ${h.mode === "next"
            ? `<div class="countdown" data-deadline="${dl}" data-fmt="big" data-hero>${fmtCountdown(dl - now)}</div>
               <div class="countdown-sub"><span>本命 <b>${r.honmei ?? "-"}号艇</b></span><span>推奨 <b>${esc(r.top_pick || "-")}</b></span></div>
               <div class="progress"><i data-window="${dl}" style="transform:scaleX(0)"></i></div>`
            : `<div class="countdown" style="color:var(--hit)">${yen(r.payout)}</div>
               <div class="countdown-sub"><span>結果 <b>${esc(r.result || "-")}</b></span><span>推奨 <b>${esc(r.top_pick || "-")}</b></span></div>`}
          <div style="margin-top:22px"><a class="btn" href="${link}">予想を見る ${arrow}</a></div>
        </div>
        <div class="panel confidence-card rv" style="--i:2">
          <div class="ring">${ringSvg(conf)}<div class="ring-center"><div><b>${conf}</b><span>確信度</span></div></div></div>
          <div>
            <div class="eyebrow">レースの見立て</div>
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
    <div class="section-head"><div><h2 class="section-title">まもなく締切<small>締切が近い順に、見出しと推奨買い目</small></h2></div></div>
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
      <div class="pay">${yen(r.payout)}</div>${r.hit ? `<span class="stamp">的中</span>` : ""}`;
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
  ["all", "すべて"],
  ["graded", "重賞"],
  ["nighter", "ナイター"],
  ["hits", "的中"],
];

function monitorHtml(day, now, filter) {
  let venues = day.venues;
  if (filter === "graded") venues = venues.filter((v) => v.grade && v.grade !== "一般");
  if (filter === "nighter") venues = venues.filter((v) => v.is_nighter);
  if (filter === "hits") venues = venues.filter((v) => v.races.some((r) => r.hit));
  const head = `<div class="mon-head" role="row"><div>場</div>${Array.from({ length: 12 }, (_, i) => `<div>${i + 1}R</div>`).join("")}</div>`;
  const rows = venues.map((v, i) => {
    const byR = Object.fromEntries(v.races.map((r) => [r.rno, r]));
    const next = v.races.find((r) => !r.result && deadlineMs(state.date, r.deadline) > now);
    const cells = Array.from({ length: 12 }, (_, k) => {
      const r = byR[k + 1];
      return r ? cellHtml(v, r, now, next && next.rno === r.rno) : `<div class="cell empty"></div>`;
    }).join("");
    return `<div class="mon-row ${state.firstPaint.monitor ? "" : "rv"}" style="--i:${Math.min(i, 12) + 4}">
      <div class="mon-venue">
        <div class="name"><b>${esc(v.name)}</b></div>
        <div class="tags">${v.grade && v.grade !== "一般" ? `<span class="chip grade-${esc(v.grade)}">${esc(v.grade)}</span>` : ""}${v.is_nighter ? `<span class="chip nighter">ナイター</span>` : ""}<span class="chip">${esc(v.day_label)}</span></div>
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
    <div class="kpi"><b class="num">${t.settled}<span class="muted" style="font-size:.5em">/${t.races}</span></b><span>確定</span></div>
    <div class="kpi hit"><b>${f(hitRate)}<small style="font-size:.5em">%</small></b><span>3連単的中</span></div>
    <div class="kpi"><b>${f(honmei)}<small style="font-size:.5em">%</small></b><span>本命 1着</span></div>
    <div class="kpi"><b>${f(roi)}<small style="font-size:.5em">%</small></b><span>回収率</span></div>
    <div class="kpi"><b style="font-size:.7em">${t.stake ? signedYen((t.return - t.stake) * 10) : "--"}</b><span>収支（1点1,000円）</span></div>
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
        <div><span class="eyebrow">${fmtDate(state.date)} · ${state.day.venues.length}場で開催</span><h2 class="section-title">全場のレース<small>本命・推奨買い目・結果をひと目で</small></h2></div>
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
const FACTOR_KEYS_ML = [["course", "コース"], ["start", "スタート力"], ["tenkai", "展開(ST順差)"], ["skill", "選手力"], ["local", "当地"], ["form", "調子"], ["racetime", "ﾚｰｽﾀｲﾑ"], ["motor", "モーター"], ["exhibition", "展示T"], ["exh_st", "展示ST"], ["original", "ｵﾘｼﾞﾅﾙ展示"], ["flying", "F"], ["wall", "壁"], ["wind", "風"]];
const isML = (P) => String(P.engine || "").startsWith("lightgbm");
const factorKeys = (P) => (isML(P) ? FACTOR_KEYS_ML : FACTOR_KEYS_MODEL);
const ENGINE_LABEL = { "lightgbm-pre": "機械学習 · 展示前", "lightgbm-post": "機械学習 · 展示反映", model: "統計モデル" };

function factorBars(f, keys) {
  return `<div class="factors" aria-hidden="true">${keys.map(([k, label]) => {
    const v = f[k] || 0;
    const hgt = Math.min(13, Math.abs(v) * 30);
    return `<span class="factor" title="${label} ${v >= 0 ? "+" : ""}${v.toFixed(2)}">${v ? `<i class="${v > 0 ? "pos" : "neg"}" style="height:${hgt}px"></i>` : ""}</span>`;
  }).join("")}</div>`;
}

// 予想が使った進入コース（展示後は展示進入、展示前は枠なり）。表はこの順に並べる
const courseOf = (race) => Object.fromEntries(race.prediction.boats.map((b) => [b.boat, b.course || b.boat]));
const exEntry = (race) => race.stage === "exhibition";
const courseTag = (c, boatNo, ex) => `<span class="ctag ${ex ? "" : "guess"} ${c !== boatNo ? "moved" : ""}" title="${ex ? "展示進入" : "枠なり想定"} ${c}コース">${c}<small>C</small></span>`;

// 風の補正（場の風向き×風速別のコース別1着率）。変化の大きいコースを3つまで
function windLine(P) {
  const w = P.wind;
  if (!w || !w.category) return "";
  const ch = Object.entries(w.factors || {}).filter(([, f]) => Math.abs(f - 1) >= 0.05)
    .sort((a, b) => Math.abs(b[1] - 1) - Math.abs(a[1] - 1)).slice(0, 3)
    .map(([c, f]) => `${"①②③④⑤⑥"[c - 1]}×${f.toFixed(2)}`).join(" ");
  return `風 ${esc(w.category)}${w.speed ? ` ${w.speed}m` : ""}${w.stabilizer ? "（安定板）" : ""}${ch ? `：${ch}` : ""}<br>`;
}

// スタート隊形トゥエルブ（予想手順 STEP②）と、その場・種類・隊形での過去の逃げ率と2着
function formationLine(race) {
  const f = race.formation;
  if (!f) return "";
  const s = f.stats;
  const where = s ? `${s.scope === "ALL" ? "全場" : esc(race.venue.name)}・${esc(s.category || f.category)}` : "";
  const sec = s && s.second ? Object.entries(s.second).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([c, p]) => `${"①②③④⑤⑥"[c - 1]}${pct(p)}%`).join(" ") : "";
  return `隊形 <b>${esc(f.label)}</b>${s ? `（${where} 逃げ${pct(s.rate)}%・${s.rank}/12位）` : ""}<br>${sec ? `逃げたら2着 ${sec}<br>` : ""}`;
}

// 2連単オッズから見た「市場の1着の見込み」（1/オッズの割合）。予想手順 STEP⑥：人気順ではなく市場心理として見る
function marketHead(race, b) {
  const o = race.odds2 || {};
  const inv = Object.entries(o).filter(([, v]) => v > 0).map(([c, v]) => [c, 1 / v]);
  if (inv.length < 20) return "";
  const tot = inv.reduce((a, [, x]) => a + x, 0);
  const head = inv.filter(([c]) => c.split("-")[0] === String(b)).reduce((a, [, x]) => a + x, 0);
  return `市場（2連単）の①頭 ${Math.round((head / tot) * 100)}%<br>`;
}

function rankClass(values, i, lowerBetter = false) {
  const vs = values.map((v, j) => [v, j]).filter(([v]) => v != null && v !== 0);
  if (vs.length < 3 || values[i] == null) return "";
  vs.sort((a, b) => (lowerBetter ? a[0] - b[0] : b[0] - a[0]));
  if (vs[0][1] === i) return "r1";
  if (vs[vs.length - 1][1] === i) return "r6";
  return "";
}

// レースタイム：6艇内の順位（節間ベストの速い順）と、節内の順位の帯 → 過去の成績から「同じコースの平均より3連対率が何ポイント上か」
const RT_BAND_EDGES = [0.1, 0.3, 0.6, 9];
function rtFields(E, ins) {
  const bests = E.map((e) => e.rt_best).filter((v) => v != null && v > 0);
  const R = ins && ins.racetime && ins.racetime.rank ? ins.racetime : null;
  return Object.fromEntries(E.map((e) => {
    if (e.rt_best == null || !(e.rt_best > 0)) return [e.boat, {}];
    const rr = 1 + bests.filter((v) => v < e.rt_best).length;
    const pctv = e.rt_series_rank && e.rt_series_n ? e.rt_series_rank / e.rt_series_n : null;
    const band = pctv == null ? null : RT_BAND_EDGES.findIndex((hi) => pctv <= hi);
    const ev = R ? (band != null && R.cell[`${rr}-${band}`]) || R.rank[String(rr)] : null;
    return [e.boat, { rr, band, ev }];
  }));
}

function sheetHtml(race, ins = null) {
  const C = courseOf(race);
  const ex = exEntry(race);
  const E = race.entries.slice().sort((a, b) => (C[a.boat] ?? a.boat) - (C[b.boat] ?? b.boat));
  const RT = rtFields(E, ins);
  E.forEach((e) => { e.rt_race_rank = RT[e.boat].rr ?? null; e.rt_eval = RT[e.boat].ev ? RT[e.boat].ev.top3_pt : null; });
  const col = (k) => E.map((e) => e[k]);
  const cols = [
    ["全国勝率", "nat_win", false, 2], ["全国2連", "nat_2", false, 1], ["当地勝率", "loc_win", false, 2],
    ["モーター2連", "motor_2", false, 1], ...(E.some((e) => e.motor_kp != null) ? [["貢献P", "motor_kp", false, 2]] : []), ["ボート2連", "boat_2", false, 1], ["平均ST", "avg_st", true, 2],
    ...(E.some((e) => e.rt_best != null) ? [["ﾀｲﾑ6艇内", "rt_race_rank", true, 0], ["選手別順位", "rt_series_rank", true, 0],
      ...(E.some((e) => e.rt_all_rank != null) ? [["全走順位", "rt_all_rank", true, 0], ["前走順位", "rt_last_rank", true, 0]] : []), ["節ﾍﾞｽﾄ", "rt_best", true, 1],
      ...(E.some((e) => e.rt_eval != null) ? [["ﾀｲﾑ評価", "rt_eval", false, 1]] : [])] : []),
    ["展示T", "exhibition_time", true, 2], ["展示ST", "ex_st", true, 2],
    // オリジナル展示（場の公式サイト）。区間が場ごとに違うので、色付けはレース内の順位だけ
    ...[["一周", "lap_time", true, 2], ["まわり足", "turn_time", true, 2], ["直線", "straight_time", true, 2]].filter(([, k]) => E.some((e) => e[k] != null)),
    ["チルト", "tilt", null, 1], ["体重", "weight", null, 1],
  ];
  const fmt = (v, d) => (v == null || v === 0 && d === 2 && false ? "--" : typeof v === "number" ? v.toFixed(d) : "--");
  const rows = E.map((e, i) => `<tr class="${C[e.boat] && C[e.boat] !== e.boat ? "moved" : ""}">
    <td>${C[e.boat] ? courseTag(C[e.boat], e.boat, ex) : "--"}</td>
    <td>${boat(e.boat, "sm")}</td>
    <td class="name">${esc(e.name)} <small class="muted num">${esc(e.toban)} ${esc(e.grade)}</small></td>
    ${cols.map(([, k, lb, d]) => {
      let v = e[k];
      let cls = lb === null ? "" : rankClass(col(k).map((x) => (k === "ex_st" && x != null ? Math.abs(x) : x)), i, lb);
      if (k === "ex_st" && v != null && v < 0) return `<td class="f">F${Math.abs(v).toFixed(2).slice(1)}</td>`;
      if (k === "rt_race_rank" && v != null) return `<td class="${cls}">${v}位</td>`;
      if (k === "rt_series_rank" && v != null) return `<td class="${cls}">${v}<small class="muted">/${e.rt_series_n ?? "?"}</small></td>`;
      if (k === "rt_all_rank" && v != null) return `<td class="${cls}">${v}<small class="muted">/${e.rt_all_n}</small></td>`;
      if (k === "rt_last_rank" && v != null) return `<td class="${cls}" title="前走 ${Math.floor(e.rt_last / 60)}'${String(Math.floor(e.rt_last % 60)).padStart(2, "0")}&quot;${Math.round((e.rt_last * 10) % 10)}">${v}<small class="muted">/${e.rt_last_n}</small></td>`;
      if (k === "rt_eval" && v != null) {
        const x = RT[e.boat].ev;
        return `<td class="${cls} ${v > 0 ? "pos" : v < 0 ? "neg" : ""}" title="この順位の選手の過去の3連対率 ${(x.top3 * 100).toFixed(1)}%（同じコースの平均との差 ${v > 0 ? "+" : ""}${v}ポイント・${x.n.toLocaleString("ja-JP")}走）">${v > 0 ? "+" : ""}${v.toFixed(1)}</td>`;
      }
      if (k === "rt_best" && v != null) return `<td class="${cls}">${Math.floor(v / 60)}'${String(Math.floor(v % 60)).padStart(2, "0")}"${Math.round((v * 10) % 10)}</td>`;
      if (k === "loc_win" && !v) return `<td class="muted">--</td>`;
      return `<td class="${cls}">${fmt(v, d)}</td>`;
    }).join("")}
    <td class="${e.f_count ? "f" : "muted"}">${e.f_count ? "F" + e.f_count : "-"}</td>
  </tr>`).join("");
  return `<div class="panel sheet"><table>
    <thead><tr><th>進入</th><th>艇</th><th>選手</th>${cols.map(([l]) => `<th>${l}</th>`).join("")}<th>F</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

// 実力：進入コースでの選手の成績（前日まで・全場）。期間（半年・1年・全期間）を切り替え、F持ちの選手はF持ちのときの成績
const SCOPES = [["6m", "半年"], ["1y", "1年"], ["all", "全期間"]];
const getScope = () => { try { return localStorage.getItem("minamo-scope") || "1y"; } catch { return "1y"; } };
function abilityHtml(race, scope = getScope()) {
  const C = courseOf(race);
  const S = Object.fromEntries(race.prediction.boats.map((b) => [b.boat, b.stats || {}]));
  if (!Object.values(S).some((s) => s.n_c || (s.profile && Object.keys(s.profile).length))) return "";
  const hasProfile = Object.values(S).some((s) => s.profile && Object.keys(s.profile).length);
  const ex = exEntry(race);
  const E = race.entries.slice().sort((a, b) => (C[a.boat] ?? a.boat) - (C[b.boat] ?? b.boat));
  const pc = (v) => (v == null ? "--" : (v * 100).toFixed(1));
  // 表に出す数字：F持ちの選手はF持ちのとき（全期間ためた分）、それ以外は選んだ期間。古いレースは前の形（全期間）
  const pick = (e) => {
    const s = S[e.boat] || {};
    if (!hasProfile) return { v: { n: s.n_c, win: s.win_c, top2: s.top2_c, top3: s.top3_c, sr: s.sr_c, topst: s.top_st }, tag: "" };
    const P = s.profile || {};
    if (e.f_count && P.f && P.f.n) return { v: P.f, tag: "F" };
    return { v: P[scope] || {}, tag: "" };
  };
  const V = E.map(pick);
  const cols = [["1着率", "win", false, pc], ["2連対率", "top2", false, pc], ["3連対率", "top3", false, pc],
    ["平均ST順位", "sr", true, (v) => (v == null ? "--" : v.toFixed(2))], ["トップST率", "topst", false, pc],
    ...(hasProfile ? [["トップ時1着", "topst_win", false, pc], ["トップ時2連", "topst_top2", false, pc]] : [])];
  const rows = E.map((e, i) => {
    const { v, tag } = V[i];
    const cells = cols.map(([, k, lb, f]) => `<td class="${rankClass(V.map((x) => x.v[k]), i, lb)}">${f(v[k])}</td>`).join("");
    const s = S[e.boat] || {};
    const wall = (C[e.boat] || e.boat) === 1 ? `<td class="muted">--</td>` : `<td>${pc(s.wall)}<small class="muted">${s.wall_n ? `（${s.wall_n}走）` : ""}</small></td>`;
    return `<tr class="${C[e.boat] && C[e.boat] !== e.boat ? "moved" : ""}"><td>${C[e.boat] ? courseTag(C[e.boat], e.boat, ex) : "--"}</td><td>${boat(e.boat, "sm")}</td>
      <td class="name">${esc(e.name)}${tag ? ` <span class="ftag" title="F持ちのときの成績">F持ち時</span>` : ""}</td><td class="num">${v.n ?? 0}</td>${cells}${wall}</tr>`;
  }).join("");
  const seg = hasProfile ? `<div class="seg" role="group" aria-label="期間">${SCOPES.map(([k, l]) => `<button type="button" data-scope="${k}" class="${k === scope ? "on" : ""}">${l}</button>`).join("")}</div>` : "";
  return `${seg}<div class="panel sheet"><table>
    <thead><tr><th>進入</th><th>艇</th><th>選手</th><th>出走</th>${cols.map(([l]) => `<th>${l}</th>`).join("")}<th>壁率</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}
// 選手別アビリティ：展示後の進入コースで判定（展示前は枠なり想定で「仮」）。買い目反映ありのものだけ予想に効く
function skillsHtml(race) {
  const C = courseOf(race);
  const S = Object.fromEntries(race.prediction.boats.map((b) => [b.boat, (b.stats || {}).abilities || []]));
  if (!Object.values(S).some((l) => l.length)) return "";
  const ex = exEntry(race);
  const E = race.entries.slice().sort((a, b) => (C[a.boat] ?? a.boat) - (C[b.boat] ?? b.boat));
  const rows = E.filter((e) => (S[e.boat] || []).length).map((e) => `<div class="skill-row">
    <div class="who">${C[e.boat] ? courseTag(C[e.boat], e.boat, ex) : ""}${boat(e.boat, "sm")}<b>${esc(e.name)}</b></div>
    <div class="skills">${S[e.boat].map((a) => `<div class="skill ${a.bet ? "bet" : ""}">
      <div class="skill-h"><span class="rank r-${esc(a.rank)}">${esc(a.rank || "-")}</span><b>${esc(a.name)}</b>
        <span class="chip">${a.kind === "report" ? "報告登録" : a.kind === "discover" ? "自動発見" : "自動検出"}</span>${a.bet ? `<span class="chip src-claude">買い目反映あり</span>` : ""}${a.prelim ? `<span class="chip">仮（展示前）</span>` : ""}</div>
      ${a.detail ? `<div class="small">${esc(a.detail)}</div>` : ""}
      ${a.strengthen ? `<div class="small muted">強化条件：${esc(a.strengthen)}</div>` : ""}
      ${a.example ? `<div class="small muted">実例：${esc(a.example)}</div>` : ""}
    </div>`).join("")}</div>
  </div>`).join("");
  return `<div class="section-head" style="margin-top:28px"><div><h2 class="section-title">選手別アビリティ<small>${ex ? "展示の進入コース" : "枠なり想定のコース（展示後に進入コースで判定し直します）"}で判定。原則は検証用の表示で、「買い目反映あり」だけが予想・買い目に効きます</small></h2></div></div>
    <div class="panel skills-panel">${rows}</div>`;
}

function bindAbility(race) {
  const box = $("#ability");
  if (!box) return;
  box.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-scope]");
    if (!b) return;
    try { localStorage.setItem("minamo-scope", b.dataset.scope); } catch { /* 保存できなくても切替はする */ }
    box.innerHTML = abilityHtml(race, b.dataset.scope);
  });
}

function boardHtml(race) {
  const P = race.prediction;
  const E = Object.fromEntries(race.entries.map((e) => [e.boat, e]));
  const maxWin = Math.max(...P.boats.map((b) => b.win));
  const ex = exEntry(race);
  const rows = P.boats.slice().sort((a, b) => (a.course || a.boat) - (b.course || b.boat)).map((b) => {
    const e = E[b.boat] || {};
    return `<div class="board-row ${b.win === maxWin ? "top" : ""}">
      <div class="lane">${courseTag(b.course || b.boat, b.boat, ex)}${boat(b.boat, "lg")}</div>
      <div class="racer"><b>${esc(e.name)}<span class="g ${esc(e.grade)}">${esc(e.grade)}</span></b><small>${b.start_order != null ? `<span class="so">予想ST順 ${Number.isInteger(b.start_order) ? b.start_order : b.start_order.toFixed(1)}番手</span> · ` : ""}${esc(e.branch)} · ${e.age ?? "-"}歳 · ${b.course}コース${e.ex_course && e.ex_course !== e.boat ? "（進入変化）" : ""}</small></div>
      <div class="winbar" data-b="${b.boat}" style="${cVar(b.boat)}"><div class="track"><span class="fill" style="width:${(b.win / maxWin) * 100}%"></span></div><span class="v">${pct(b.win)}<small>%</small></span></div>
      <div class="num">${pct(b.top2)}%</div>
      <div class="num" data-l="3連対">${pct(b.top3)}%</div>
      <div>${factorBars(b.factors, factorKeys(P))}</div>
    </div>`;
  }).join("");
  return `<div class="panel board">
    <div class="board-head"><div>進入</div><div>選手</div><div>1着確率</div><div style="text-align:right">2連対</div><div style="text-align:right">3連対</div><div>要因（＋/−）</div></div>
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
  const main = (ai.picks || []).map((p) => ({ combo: p.combo, weight: p.weight, kind: "本線", ability: p.ability }));
  const value = (race.prediction.picks || []).filter((p) => p.kind === "妙味" && !main.some((m) => m.combo === p.combo)).map((p) => ({ combo: p.combo, kind: "妙味" }));
  const all = [...main, ...value];
  return `<div class="tickets">${all.map((t, i) => {
    const p = P[t.combo] ?? modelPicks[t.combo]?.p;
    const o = odds[t.combo];
    const ev = p && o ? p * o : null;
    return `<div class="ticket ${t.kind === "妙味" ? "value" : ""} ${won === t.combo ? "won" : ""} rv" style="--i:${i}">
      <div class="kind"><span>3連単 ${String(i + 1).padStart(2, "0")} · <b>${won === t.combo ? "的中" : t.kind}</b>${t.ability ? ` · ${esc(t.ability)}` : ""}</span>${t.weight ? `<span class="weight">${t.weight}<small>%</small></span>` : ""}</div>
      <div class="cmb">${t.combo.split("-").map((b) => boat(b)).join(`<span class="arrow"></span>`)}</div>
      <div class="stats"><div><span>確率</span>${p != null ? pct(p, 1) + "%" : "--"}</div><div><span>オッズ</span>${o ?? "--"}</div><div><span>期待値</span>${ev ? ev.toFixed(2) : "--"}</div></div>
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

// 3連単・2連単以外の払戻（3連複・2連複・拡連複・単勝・複勝）
function otherPayHtml(P) {
  if (!P || !Object.keys(P).length) return "";
  const kinds = [["trio", "3連複"], ["quinella", "2連複"], ["wide", "拡連複"], ["win", "単勝"], ["place", "複勝"]];
  const cmb = (c) => c.split("=").map((b) => boat(b, "sm")).join(`<span class="eq">=</span>`);
  const cells = kinds.filter(([k]) => P[k]).map(([k, label]) => `<div class="op"><span class="lbl">${label}</span>${Object.entries(P[k]).map(([c, v]) =>
    `<span class="op-i"><span class="op-c">${cmb(c)}</span><b>${yen(v)}</b>${P[`${k}_pop`] && P[`${k}_pop`][c] ? `<small>${P[`${k}_pop`][c]}番人気</small>` : ""}</span>`).join("")}</div>`).join("");
  return `<div class="panel other-pay">${cells}</div>`;
}

function resultHtml(race) {
  const r = race.result;
  if (!r) return "";
  if (r.cancelled) return `<div class="result-band"><div><div class="lbl">結果</div><b>レース中止</b></div></div>`;
  const s = race.settle || {};
  const E = Object.fromEntries(race.entries.map((e) => [e.boat, e]));
  const pay = (combo, yenv, pop) => `<div class="res-pay"><div class="result-order">${combo}</div>
      <div class="res-money"><b>${yenv != null ? yen(yenv) : "--"}</b>${pop ? `<span>${pop}番人気</span>` : ""}</div></div>`;
  const rows = (r.rows || []).slice().sort((a, b) => (a.place ?? 9) - (b.place ?? 9)).map((x) => `<tr>
      <td class="pl">${x.place ? `${x.place}着` : esc(x.status || "－")}</td><td>${boat(x.boat, "sm")}</td>
      <td class="name">${esc(x.name || E[x.boat]?.name || "")}</td><td>${x.course ? `${x.course}コース` : "--"}</td>
      <td>${x.st != null ? (x.st < 0 ? `F${Math.abs(x.st).toFixed(2).slice(1)}` : x.st.toFixed(2).replace(/^0/, "")) : "--"}</td><td>${esc(x.time || "--")}</td></tr>`).join("");
  return `<div class="result-band ${s.trifecta_hit ? "hit" : ""} rv">
    <div class="res-cols">
      <div><div class="lbl">3連単</div>${pay(combo(r.trifecta, ""), r.payout, r.popularity)}</div>
      ${r.exacta ? `<div><div class="lbl">2連単</div>${pay(combo(r.exacta, ""), r.exacta_payout, r.exacta_popularity)}</div>` : ""}
      ${r.kimarite ? `<div><div class="lbl">決まり手</div><div class="res-kima">${esc(r.kimarite)}</div></div>` : ""}
    </div>
    <div>${s.trifecta_hit ? `<div class="hit-stamp">的中</div>` : `<div class="miss-stamp">${s.honmei_win ? "本命1着" : "はずれ"}</div>`}</div>
  </div>
  ${otherPayHtml(r.payouts)}
  ${rows ? `<div class="panel res-table"><div class="ledger-scroll"><table class="streak-t res-t"><thead><tr><th>着順</th><th>艇</th><th>選手</th><th>進入</th><th>ST</th><th>レースタイム</th></tr></thead><tbody>${rows}</tbody></table></div></div>` : ""}`;
}

// あなたの予想：1着・2着・3着に入れる艇を選ぶと、組み合わせ（フォーメーション）を作り、MINAMOの確率・オッズと並べる。
// このブラウザだけに保存する（ほかの人には見えない）
const myKey = (race) => `minamo-my-${race.date}-${race.venue.jcd}-${race.rno}`;
const myLoad = (race) => { try { return JSON.parse(localStorage.getItem(myKey(race)) || "null") || { f: [[], [], []] }; } catch { return { f: [[], [], []] }; } };
const mySave = (race, v) => { try { localStorage.setItem(myKey(race), JSON.stringify(v)); } catch { /* 保存できなくても表示は続ける */ } };
function myCombos(f) {
  const out = [];
  for (const a of f[0]) for (const b of f[1]) for (const c of f[2]) if (a !== b && b !== c && a !== c) out.push(`${a}-${b}-${c}`);
  return out;
}
function myPickHtml(race) {
  const v = myLoad(race);
  const P = race.tri_all || Object.fromEntries((race.prediction.trifecta || []).map((t) => [t.combo, t.p]));
  const O = race.odds_all || race.odds || {};
  const rank = Object.fromEntries(Object.keys(P).sort((a, b) => P[b] - P[a]).map((c, i) => [c, i + 1]));
  const mine = myCombos(v.f);
  const minamo = new Set((race.ai?.picks || []).map((p) => p.combo));
  const trial = new Set(race.ev_pick || []);
  const won = race.result && !race.result.cancelled ? race.result.trifecta : null;
  const sumP = mine.reduce((a, c) => a + (P[c] || 0), 0);
  const evs = mine.filter((c) => P[c] != null && O[c]).map((c) => P[c] * O[c]);
  const hitMine = won && mine.includes(won);
  const rows = mine.slice().sort((a, b) => (P[b] || 0) - (P[a] || 0)).map((c) => `<tr class="${c === won ? "on" : ""}">
    <td>${combo(c)}</td><td>${P[c] != null ? pct(P[c], 1) + "%" : "--"}</td><td>${rank[c] ? rank[c] + "位" : "--"}</td><td>${O[c] ?? "--"}</td>
    <td>${P[c] != null && O[c] ? (P[c] * O[c]).toFixed(2) : "--"}</td><td>${minamo.has(c) ? "◎" : ""}${trial.has(c) ? "★" : ""}</td></tr>`).join("");
  const picker = ["1着", "2着", "3着"].map((lab, i) => `<div class="my-row"><span class="my-lab">${lab}</span>${[1, 2, 3, 4, 5, 6].map((b) => `<button type="button" class="my-b ${v.f[i].includes(b) ? "on" : ""}" data-pos="${i}" data-boat="${b}" aria-pressed="${v.f[i].includes(b)}">${boat(b, "sm")}</button>`).join("")}</div>`).join("");
  const sameN = mine.filter((c) => minamo.has(c)).length;
  return `<div class="panel my-panel">
    <div class="my-pick">${picker}<button type="button" class="btn ghost my-clear">クリア</button></div>
    ${mine.length ? `<div class="calib my-sum">
      <div class="panel"><h4>点数</h4><div class="big">${mine.length}<small style="font-size:.45em">点</small></div><div class="small">1点1,000円で ${yen(mine.length * 1000)}</div></div>
      <div class="panel"><h4>MINAMOから見た的中率</h4><div class="big">${pct(sumP, 1)}<small style="font-size:.45em">%</small></div><div class="small">選んだ組の確率の合計</div></div>
      <div class="panel"><h4>期待値の平均</h4><div class="big">${evs.length ? (evs.reduce((a, x) => a + x, 0) / evs.length).toFixed(2) : "--"}</div><div class="small">確率×オッズ（1.0より上なら割安）</div></div>
      <div class="panel"><h4>${won ? (hitMine ? "的中" : "はずれ") : "MINAMOと同じ組"}</h4><div class="big">${won ? (hitMine ? yen((race.result.payout || 0) * BET_UNIT) : "--") : `${sameN}<small style="font-size:.45em">/${mine.length}点</small>`}</div><div class="small">${won ? `結果 ${esc(won)}${race.settle ? ` · MINAMOは${race.settle.trifecta_hit ? "的中" : "はずれ"}` : ""}` : "◎＝MINAMOの推奨買い目にもある組"}</div></div>
    </div>
    <div class="ledger-scroll"><table class="streak-t my-t"><thead><tr><th>3連単</th><th>MINAMOの確率</th><th>120通り中</th><th>オッズ</th><th>期待値</th><th>MINAMO</th></tr></thead><tbody>${rows}</tbody></table></div>` : `<p class="small muted" style="margin:12px 2px 0">1着・2着・3着に入れる艇を押してください。押した艇で組み合わせを作ります（例：1着①、2着②③、3着②③④ → 4点）。</p>`}
  </div>`;
}
function bindMyPick(race) {
  const box = $("#mypick");
  if (!box) return;
  box.addEventListener("click", (ev) => {
    const b = ev.target.closest(".my-b");
    const v = myLoad(race);
    if (b) {
      const i = Number(b.dataset.pos), n = Number(b.dataset.boat);
      v.f[i] = v.f[i].includes(n) ? v.f[i].filter((x) => x !== n) : [...v.f[i], n].sort();
    } else if (ev.target.closest(".my-clear")) v.f = [[], [], []];
    else return;
    mySave(race, v);
    box.innerHTML = myPickHtml(race);
  });
}

const MARK_SYM = [["honmei", "◎", "本命"], ["taikou", "○", "対抗"], ["ana", "▲", "穴"]];

async function renderRace(r, refresh = false) {
  const [race, ins] = await Promise.all([getJSON(`data/${r.date}/${r.jcd}-${String(r.rno).padStart(2, "0")}.json`),
    state.insights !== undefined ? state.insights : getJSON("data/insights.json").catch(() => null)]);
  state.insights = ins;
  state.race = race;
  if (r.date !== state.date) { state.date = r.date; $("#dateSelect").value = r.date; }
  const ai = race.ai || {};
  const P = race.prediction;
  const E = Object.fromEntries(race.entries.map((e) => [e.boat, e]));
  const dl = deadlineMs(race.date, race.deadline);
  const now = nowMs();
  const done = !!race.result;
  const srcChip = ai.source === "claude" ? `<span class="chip src-claude">● Claudeの見解</span>` : ai.source === "demo" ? `<span class="chip">デモ</span>` : `<span class="chip">自動の見解</span>`;
  const stage = (race.stage === "exhibition" ? `<span class="chip">展示反映済</span>` : `<span class="chip">出走表段階</span>`)
    + (isML(race.prediction) ? `<span class="chip src-claude">機械学習</span>` : "");
  race.__showResult = done;
  const prevNext = `
    <div style="display:flex;gap:8px;margin-top:26px;flex-wrap:wrap">
      ${r.rno > 1 ? `<a class="btn ghost" href="#/race/${r.date}/${r.jcd}/${r.rno - 1}">← ${r.rno - 1}R</a>` : ""}
      ${r.rno < 12 ? `<a class="btn ghost" href="#/race/${r.date}/${r.jcd}/${r.rno + 1}">${r.rno + 1}R →</a>` : ""}
    </div>`;
  const html = `
  <div class="wrap">
    <a class="back" href="#/"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M14 8H3M7 4L3 8l4 4"/></svg>レース一覧へ</a>
    <header class="race-hero">
      <div class="rv">
        <div class="race-title"><span class="jp">${esc(race.venue.name)}</span><span class="rno">${race.rno}<small>R</small></span></div>
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
        ${done ? `<div class="cd done">終了</div>` : `<div class="cd" data-deadline="${dl}" data-fmt="big">${fmtCountdown(dl - now)}</div>`}
        <div class="lbl">締切 ${esc(race.deadline)} · ${fmtDate(race.date)}</div>
      </div>
    </header>
    ${resultHtml(race)}
    <div class="verdict-grid">
      <article class="panel verdict rv" style="--i:2">
        <div class="who"><span class="eyebrow">見解</span>${srcChip}</div>
        <h2>${esc(ai.headline)}</h2>
        <p class="body">${esc(ai.verdict)}</p>
        <div class="marks">${MARK_SYM.map(([k, sym, label]) => ai[k] ? `<div class="mark"><span class="sym">${sym}</span>${boat(ai[k])}<span class="nm">${esc(E[ai[k]]?.name || "")}<small>${label}</small></span></div>` : "").join("")}</div>
        ${ai.key_points && ai.key_points.length ? `<ol class="points">${ai.key_points.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>` : ""}
        ${ai.risk ? `<p class="risk">注意 — ${esc(ai.risk)}</p>` : ""}
      </article>
      <div class="side-stack">
        <div class="panel confidence-card rv" style="--i:3">
          ${P.escape && P.escape.index != null ? `<div class="ring">${ringSvg(P.escape.index)}<div class="ring-center"><div><b>${P.escape.index}</b><span>逃げ指数</span></div></div></div>
          <div><div class="eyebrow">イン逃げ指数</div><div class="tier" style="margin-top:8px">${esc(P.escape.label)}</div><div class="tier-note">1コース ${boat(P.escape.boat, "sm")} ${esc(E[P.escape.boat]?.name || "")} · 1着率 ${pct(P.escape.p)}%<br>${marketHead(race, P.escape.boat)}${formationLine(race)}${windLine(P)}モデル確信度 ${P.confidence} · ${esc(P.tier)}</div></div>`
          : `<div class="ring">${ringSvg(ai.confidence ?? P.confidence)}<div class="ring-center"><div><b>${ai.confidence ?? P.confidence}</b><span>確信度</span></div></div></div>
          <div><div class="eyebrow">レースの見立て</div><div class="tier" style="margin-top:8px">${tierOf(ai.confidence ?? P.confidence)}</div><div class="tier-note">モデル確信度 ${P.confidence} · ${esc(P.tier)}</div></div>`}
        </div>
        <div class="panel scenario rv" style="--i:4">
          <div class="eyebrow">決まり手予測</div>
          ${Object.entries(P.scenario).slice(0, 5).map(([k, v]) => `<div class="scen-row"><span>${esc(k)}</span><span class="bar"><i style="width:${v * 100}%"></i></span><span class="num">${pct(v)}%</span></div>`).join("")}
        </div>
      </div>
    </div>

    <section class="panel sim rv" style="--i:5">
      <div class="sim-head"><span class="eyebrow">${done ? "1周1マーク（結果の再現）" : "1周1マークのイメージ"}</span><button class="replay" id="replay" type="button"><svg viewBox="0 0 12 12" fill="currentColor"><path d="M3 1.5v9l7.5-4.5z"/></svg>もう一度</button></div>
      <canvas class="sim-canvas" id="sim" role="img" aria-label="1周1マークの展開シミュレーション"></canvas>
      <div class="sim-foot"><span>進入 ${P.boats.slice().sort((a, b) => a.course - b.course).map((b) => b.boat).join("")} ${race.stage === "exhibition" ? "（展示進入）" : "（枠なり想定）"}</span><span>${isML(P) ? "予想スタート順" : "ST"}・予想着順から描画したイメージです</span></div>
    </section>

    <section class="section">
      <div class="section-head"><div><h2 class="section-title">1着確率と根拠<small>予想エンジンが出した1着・2連対・3連対確率と、その根拠。${exEntry(race) ? "展示の進入コース順" : "枠なり想定のコース順（展示後に展示進入で並べ替え）"}</small></h2></div></div>
      ${boardHtml(race)}
    </section>

    <section class="section">
      <div class="section-head"><div><h2 class="section-title">推奨買い目<small>買い目と配分（％）、オッズから見た妙味</small></h2></div></div>
      ${ticketsHtml(race)}
    </section>

    <section class="section">
      <div class="section-head"><div><h2 class="section-title">あなたの予想と比べる<small>1着・2着・3着の艇を選ぶと、MINAMOの確率・オッズ・期待値と並べます。◎はMINAMOの推奨買い目、★は試験中の買い目にもある組。このブラウザだけに保存されます</small></h2></div></div>
      <div id="mypick">${myPickHtml(race)}</div>
    </section>

    <section class="section">
      <div class="section-head"><div><h2 class="section-title">出走表データ<small>出走表・直前情報（● はレース内1位）。${exEntry(race) ? "展示の進入コース順" : "枠なり想定のコース順"}</small></h2></div></div>
      ${sheetHtml(race, ins)}
      ${race.entries.some((e) => e.rt_best != null) ? `<p class="small muted" style="margin:10px 2px 0;line-height:1.7">ﾀｲﾑ6艇内＝節間ベストのレースタイムの、このレースの6艇の中での順位。選手別順位＝節間ベストの、その節に出ている選手の中での順位（順位/人数）。全走順位＝節間ベストの、その節の全部の走り（1走ずつ数える）の中での順位。前走順位＝いちばん新しい走りのタイムの、各選手のいちばん新しい走りの中での順位（数字に触れると前走のタイム）。ﾀｲﾑ評価＝この「6艇内の順位」と「節内の順位（上位10%・10〜30%・30〜60%・それより下）」だった選手の過去の3連対率が、同じコースの平均より何ポイント高いか${ins && ins.racetime && ins.racetime.n ? `（${ins.racetime.n.toLocaleString("ja-JP")}走から）` : ""}。</p>` : ""}
      ${abilityHtml(race) ? `<div class="section-head" style="margin-top:28px"><div><h2 class="section-title">実力（進入コースでの成績）<small>この進入コースに入ったときの成績（前日まで・全場、%）。F持ちの選手は、F持ちだったときの成績</small></h2></div></div><div id="ability">${abilityHtml(race)}</div>
      <p class="small muted" style="margin:10px 2px 0;line-height:1.7">出走＝そのコースでの出走数。平均ST順位＝そのコースでの本番のスタート順位の平均（小さいほど早い。スタート隊形トゥエルブの元の数字）。トップST率＝そのコースで本番のスタートが1番だった割合。トップ時1着・2連＝そのトップスタートのときの1着率・2連対率。「F持ち時」＝今F持ちの選手は、F持ちだったときの成績（期間で区切らず、ためていく）。壁率＝その選手がこのコースのとき①が1着だった割合（高いほど①が逃げやすい）。</p>` : ""}
      ${skillsHtml(race)}
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
  bindAbility(race);
  bindMyPick(race);
  const canvas = $("#sim");
  document.fonts.ready.then(() => {
    if (!canvas.isConnected) return;
    state.sim = mountSim(canvas, race);
    $("#replay").addEventListener("click", () => state.sim.play());
  });
  document.title = `${race.venue.name}${race.rno}R — MINAMO`;
  tick();
}

/* ------------------------------------------------------------ 今買う候補（試験中：オッズで絞った買い目） */
const hhmm = (iso) => (iso ? iso.slice(11, 16) : "--:--");
const PICK_KIND = {
  ev: { label: "3連単", rule: "MINAMOの確率を市場（オッズ）と合わせて補正し、期待値（確率×オッズ）が1.2以上の3連単を最大9点", check: "過去の検証（学習に使っていない約3,000レース）で回収率124%（幅101〜146%）" },
  co: { label: "3連単 合成", rule: "3連単と同じ組を、合成オッズ配分（オッズの低い組を多めに。どれが当たっても払戻が同じ）で買う。当たれば必ず投資の合成オッズ倍（真ん中18倍）が戻り、トリガミになりません", check: "過去の検証（学習に使っていない約3,000レース）で回収率122%（幅99〜149%）" },
  ex: { label: "2連単", rule: "補正した確率を2連単にまとめ、2連単のオッズで期待値1.2以上の組を最大3点", check: "過去の検証（学習に使っていない約3,000レース）で回収率120%（幅109〜131%）" },
};
// 組ごとの金額で当たったときの払戻の幅（100円単位にそろえるので、組によって少しずれる）
const payRange = (items, stakes) => {
  const v = items.map((x, i) => Math.round(stakes[i] * x.odds));
  const lo = Math.min(...v), hi = Math.max(...v);
  return lo === hi ? yen(lo) : `${yen(lo)}〜${yen(hi)}`;
};
// 合成オッズ：1 ÷ Σ(1/オッズ)
const compositeOf = (items) => { const inv = items.reduce((a, x) => a + (x.odds ? 1 / x.odds : NaN), 0); return inv > 0 ? 1 / inv : null; };
// 1点の金額（ev-check「10.」：資金10万円なら、3連単は平掛け1点100円かケリー1/4で1点1,000円まで、2連単は300〜500円か3,000円まで）
const STAKE_DEFAULT = { ev: { bank: 100000, how: "kelly", flat: 100, cap: 1000 }, ex: { bank: 100000, how: "kelly", flat: 300, cap: 3000 }, co: { budget: 500 } };
function getStake(k) {
  let v = {};
  try { v = JSON.parse(localStorage.getItem(`minamo-stake-${k}`) || "{}") || {}; } catch { /* 読めなければ既定 */ }
  return { ...STAKE_DEFAULT[k], ...v };
}
function setStake(k, v) { try { localStorage.setItem(`minamo-stake-${k}`, JSON.stringify(v)); } catch { /* 保存できなくても表示はする */ } }
// 1点の金額（円）。ケリー1/4：資金×(確率×オッズ−1)/(オッズ−1)×1/4、資金の5%と上限まで、100円単位（最低100円）。期待値1以下は0円
function stakeFor(x, s) {
  if (s.how === "flat") return s.flat;
  if (x.p == null || !x.odds || x.odds <= 1) return null;
  const f = (x.p * x.odds - 1) / (x.odds - 1);
  return Math.round(Math.max(0, Math.min(s.bank * f * 0.25, s.bank * 0.05, s.cap)) / 100) * 100;
}
const pickRes = (r, k) => (k === "ex" ? r.result_ex : r.result);
const pickPay = (r, k) => (k === "ex" ? r.payout_ex : r.payout);
function pickCard(date, r, now, k = "ev", s = getStake(k)) {
  const dl = deadlineMs(date, r.deadline);
  const res = pickRes(r, k);
  const done = res && !r.cancelled;
  const items = r[`${k}_items`] || (r[`${k}_pick`] || []).map((c) => ({ combo: c }));
  const won = done && r[`${k}_hit`];
  const comp = k === "co" ? compositeOf(items) : null;
  // 合成オッズ配分：1レースの金額 × 合成オッズ ÷ オッズ（100円単位）
  const stakes = items.map((x) => (k === "co" ? (comp && x.odds ? Math.max(100, Math.round((s.budget * comp) / x.odds / 100) * 100) : null) : stakeFor(x, s)));
  const total = stakes.reduce((a, v) => a + (v || 0), 0);
  const hitStake = done ? stakes[items.findIndex((x) => x.combo === res)] : null;
  return `<a class="panel pick-card ${done ? (won ? "won" : "lost") : ""}" href="#/race/${date}/${r.v.jcd}/${r.rno}">
    <div class="pick-h"><b>${esc(r.v.name)} ${r.rno}R</b><span class="muted">締切 ${esc(r.deadline)}</span>${!done && !r.cancelled ? `<span class="chip fix ${r.pick_fixed ? "src-claude" : ""}">${r.pick_fixed ? "決定" : "仮"}</span>` : ""}
      ${done ? `<span class="chip ${won ? "src-claude" : ""}">${won ? `的中 ${yen(k === "co" ? (r.co_return || 0) * BET_UNIT : (pickPay(r, k) || 0) * BET_UNIT)}` : "はずれ"}</span>`
        : r.cancelled ? `<span class="chip">中止</span>` : dl > now ? `<span class="cd" data-deadline="${dl}">${fmtCountdown(dl - now)}</span>` : `<span class="chip">締切</span>`}</div>
    <table class="pick-t"><thead><tr><th>${PICK_KIND[k].label}</th><th>確率</th><th>オッズ</th><th>${k === "co" ? "配分" : "期待値"}</th><th>1点</th></tr></thead><tbody>
      ${items.map((x, i) => `<tr class="${done && x.combo === res ? "on" : ""}"><td>${combo(x.combo)}</td><td>${x.p != null ? pct(x.p, 1) + "%" : "--"}</td><td>${x.odds ?? "--"}</td><td>${k === "co" ? (comp && x.odds ? Math.round((100 * comp) / x.odds) + "%" : "--") : x.ev != null ? x.ev.toFixed(2) : "--"}</td><td>${stakes[i] == null ? "--" : yen(stakes[i])}</td></tr>`).join("")}
    </tbody></table>
    <div class="small muted">${k === "co" && comp ? `<span class="nowrap">合成 ${comp.toFixed(1)}倍</span> · ` : ""}${items.length}点 · 合計 ${stakes.some((v) => v == null) ? "--" : yen(total)}${k === "co" && comp && !done && stakes.every((v) => v) ? ` · <span class="nowrap">当たれば ${payRange(items, stakes)}</span>` : ""}${won && hitStake ? ` · <span class="nowrap">払戻 ${yen(Math.round(hitStake * (pickPay(r, k) || 0) / 100))}</span>` : ""} · オッズ ${hhmm(r[`${k}_at`])} 時点${done ? ` · <span class="nowrap">結果 ${esc(res)}</span>` : ""}</div>
  </a>`;
}
const getPickKind = () => { try { return localStorage.getItem("minamo-pick-kind") || "ev"; } catch { return "ev"; } };
async function renderPicks(refresh = false, k = getPickKind()) {
  if (!state.day || state.day.date !== state.date || refresh) await loadDay();
  const date = state.date, now = nowMs();
  const races = allRaces().filter((r) => Array.isArray(r[`${k}_pick`]));
  const bought = races.filter((r) => r[`${k}_pick`].length);
  const skipped = races.length - bought.length;
  const open = bought.filter((r) => !pickRes(r, k) && !r.cancelled).sort((a, b) => a.deadline.localeCompare(b.deadline));
  const done = bought.filter((r) => pickRes(r, k) || r.cancelled).sort((a, b) => b.deadline.localeCompare(a.deadline));
  const settled = done.filter((r) => pickRes(r, k) && !r.cancelled && r[`${k}_stake`] != null);
  const stake = settled.reduce((a, r) => a + r[`${k}_stake`], 0), ret = settled.reduce((a, r) => a + (r[`${k}_return`] || 0), 0);
  const hits = settled.filter((r) => r[`${k}_hit`]).length;
  const s = getStake(k);
  const y = scrollY;
  $("#main").innerHTML = `<div class="wrap"><section class="section">
    <div class="seg seg-big" role="group" aria-label="選び方" id="pickKind">${Object.entries(PICK_KIND).map(([kk, x]) => `<button type="button" data-kind="${kk}" class="${kk === k ? "on" : ""}">${x.label}</button>`).join("")}</div>
    <span class="eyebrow">${fmtDate(date)}（${weekday(date)}）</span>
    <h1 class="section-title" style="font-size:clamp(36px,5vw,72px)">今買う候補<small>試験中の選び方（${PICK_KIND[k].label}）：${PICK_KIND[k].rule}。無ければ見送り。${PICK_KIND[k].check}でしたが、まだ試験中です</small></h1>
    ${k === "co" ? `<form class="panel stake-form" id="stakeForm" onsubmit="return false">
      <label>1レースの金額<input type="number" inputmode="numeric" name="budget" min="100" step="100" value="${s.budget}">円</label>
      <p class="small muted">組ごとの金額は「1レースの金額×合成オッズ÷オッズ」（100円単位、最低100円なので合計は少しずれます）。過去の検証（資金10万円）の目安は1レース300〜500円。この端末だけに保存します。</p>
    </form>` : `<form class="panel stake-form" id="stakeForm" onsubmit="return false">
      <div class="seg" role="group" aria-label="1点の金額の決め方">${[["kelly", "ケリー1/4"], ["flat", "平掛け"]].map(([h, n]) => `<button type="button" data-how="${h}" class="${s.how === h ? "on" : ""}">${n}</button>`).join("")}</div>
      ${s.how === "kelly"
        ? `<label>資金<input type="number" inputmode="numeric" name="bank" min="1000" step="1000" value="${s.bank}">円</label>
           <label>1点の上限<input type="number" inputmode="numeric" name="cap" min="100" step="100" value="${s.cap}">円</label>`
        : `<label>1点<input type="number" inputmode="numeric" name="flat" min="100" step="100" value="${s.flat}">円</label>`}
      <p class="small muted">${s.how === "kelly"
        ? `確率とオッズから1点の金額を決めます（資金×ケリーの1/4。資金の5%と上限まで、100円単位）。資金は買った結果に合わせてご自分で入れ直してください。`
        : `どの組も同じ金額で買います。`}
        過去の検証（資金10万円）の目安：${k === "ev" ? "平掛け1点100円、またはケリー1/4で1点1,000円まで" : "平掛け1点300〜500円、またはケリー1/4で1点3,000円まで"}。この端末だけに保存します。</p>
    </form>`}
    <div class="calib">
      <div class="panel"><h4>今日の候補</h4><div class="big">${bought.length}<small style="font-size:.45em">R</small></div><div class="small">見送り ${skipped}R · 締切前 ${open.length}R</div></div>
      <div class="panel"><h4>的中</h4><div class="big">${hits}<small style="font-size:.45em">/${settled.length}R</small></div><div class="small">結果の出たレース</div></div>
      <div class="panel"><h4>今日の回収率</h4><div class="big">${stake ? ((ret / stake) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">${k === "co" ? "1レース同じ金額で買った場合" : "1点同じ金額で買った場合"}</div></div>
      <div class="panel"><h4>今日の収支</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px)">${stake ? signedYen((ret - stake) * BET_UNIT) : "--"}</div><div class="small">${k === "co" ? "1レース1,000円" : "1点1,000円"}</div></div>
    </div>
    <div class="section-head" style="margin-top:34px"><div><h2 class="section-title">締切前・結果待ち<small>締切の近い順。締切の約5分前のオッズで組を決めて「決定」にし、Discord に知らせます。それまでは「仮」で、オッズが変わると組が入れ替わります。成績は決定した組で数えます</small></h2></div></div>
    ${open.length ? `<div class="pick-grid">${open.map((r) => pickCard(date, r, now, k, s)).join("")}</div>` : `<div class="panel" style="padding:20px">今は締切前の候補がありません。直前情報（展示）が出たレースから順に候補を決めます。</div>`}
    ${done.length ? `<div class="section-head" style="margin-top:34px"><div><h2 class="section-title">結果<small>新しい順</small></h2></div></div><div class="pick-grid">${done.map((r) => pickCard(date, r, now, k, s)).join("")}</div>` : ""}
    <p class="small muted" style="margin-top:22px;line-height:1.7">これまでの通算は<a href="#/record">成績</a>の「試験中：オッズで絞った買い目」にあります。舟券の購入はご自身の判断でお願いします。</p>
  </section></div>`;
  if (refresh) scrollTo({ top: y });
  $("#pickKind").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-kind]");
    if (!b) return;
    try { localStorage.setItem("minamo-pick-kind", b.dataset.kind); } catch { /* 保存できなくても切替はする */ }
    renderPicks(false, b.dataset.kind);
  });
  const form = $("#stakeForm");
  form.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-how]");
    if (!b) return;
    setStake(k, { ...getStake(k), how: b.dataset.how });
    renderPicks(false, k);
  });
  form.addEventListener("change", (ev) => {
    const el = ev.target;
    const v = Math.round(Number(el.value) / 100) * 100;
    if (!el.name || !(v >= 100)) return;
    setStake(k, { ...getStake(k), [el.name]: v });
    renderPicks(false, k);
  });
  document.title = "今買う候補 — MINAMO";
  tick();
}

/* ------------------------------------------------------------ record */
async function renderRecord() {
  const [rec] = await Promise.all([getJSON("data/record.json")]);
  const dates = (state.latest.dates || []).slice(-30);
  const days = (await Promise.all(dates.map((d) => getJSON(`data/${d}/day.json`).catch(() => null)))).filter(Boolean);
  const T = rec.totals;
  const hitRate = T.settled ? (T.hits / T.settled) * 100 : 0;
  const roi = T.stake ? (T.return / T.stake) * 100 : 0;
  const honmei = T.settled ? (T.honmei_hits / T.settled) * 100 : 0;
  const tiers = { 鉄板: [0, 0, 0], 本線: [0, 0, 0], 混戦: [0, 0, 0], 波乱: [0, 0, 0] };
  let best = null;
  for (const d of days.slice(-14)) for (const v of d.venues) for (const r of v.races) {
    if (!r.result || r.cancelled) continue;
    const t = tiers[tierOf(r.confidence ?? 0)];
    t[0]++; t[1] += r.honmei_win ? 1 : 0; t[2] += r.hit ? 1 : 0;
    if (r.hit && (!best || r.payout > best.payout)) best = { ...r, v, date: d.date };
  }
  // まだ1レースも確定していない日（当日の朝など）は 0% に見えてしまうので描かない
  const dd = rec.days.filter((d) => d.settled).slice(-14);
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
  const pts = dd.map((d, i) => [pad + i * bw + bw * 0.5, y((d.hits / d.settled) * 100), d]);
  const line = pts.map(([x, py]) => `${x},${py}`).join(" ");
  // 点も打つ（1日だけだと線にならないため）。横に伸びる SVG でも丸く見えるよう、長さ0の線の丸端で描く
  const dots = pts.map(([x, py, d]) => `<path class="dot" d="M${x} ${py}h0"><title>${fmtDate(d.date)} 的中率 ${((d.hits / d.settled) * 100).toFixed(1)}%</title></path>`).join("");
  $("#main").innerHTML = `
  <div class="wrap">
    <section class="section">
      <span class="eyebrow">${rec.days.filter((d) => d.settled).length}日分</span>
      <h1 class="section-title" style="font-size:clamp(40px,6vw,90px)">成績<small>すべての予想は締切前に公開し、結果と自動照合しています。${rec.demo ? "（現在はデモデータ）" : ""}</small></h1>
      <div class="rec-hero">
        <div class="panel rec-kpi gold rv"><span class="eyebrow">3連単 的中率</span><div class="v">${hitRate.toFixed(1)}<small>%</small></div><p>${T.hits} / ${T.settled} レース</p></div>
        <div class="panel rec-kpi rv" style="--i:1"><span class="eyebrow">回収率</span><div class="v">${roi.toFixed(1)}<small>%</small></div><p>推奨買い目を各100円で購入した場合</p></div>
        <div class="panel rec-kpi rv" style="--i:2"><span class="eyebrow">本命 1着率</span><div class="v">${honmei.toFixed(1)}<small>%</small></div><p>◎ が1着になった割合</p></div>
        <div class="panel rec-kpi rv" style="--i:3"><span class="eyebrow">最高払戻</span><div class="v" style="font-size:clamp(36px,4vw,56px)">${best ? yen(best.payout) : "--"}</div><p>${best ? `${fmtDate(best.date)} ${esc(best.v.name)}${best.rno}R ${esc(best.result)}` : "まだありません"}</p></div>
      </div>
      <div class="panel chart rv" style="--i:4">
        <div class="section-head" style="margin:0 0 8px"><span class="eyebrow">日別の回収率・的中率</span><span class="chip">■ 回収率 ― 的中率</span></div>
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="日別回収率">
          <line class="ref" x1="${pad}" x2="${W - pad}" y1="${y(100)}" y2="${y(100)}"/>
          <text x="${W - pad}" y="${y(100) - 6}" text-anchor="end">100%</text>
          ${bars}
          <polyline class="line" points="${line}"/>
          ${dots}
          <line class="axis" x1="${pad}" x2="${W - pad}" y1="${H - pad}" y2="${H - pad}"/>
        </svg>
      </div>
      <div class="section-head" style="margin-top:40px"><div><h2 class="section-title">1点1,000円で買った場合<small>推奨買い目を各1,000円で買ったときの金額（払戻は100円あたりの配当×10）。回収率は100円のときと同じです</small></h2></div></div>
      <div class="calib">
        <div class="panel rv"><h4>投資</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px)">${yen(T.stake * 10)}</div><div class="small">${T.settled}レース</div></div>
        <div class="panel rv" style="--i:1"><h4>払戻</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px)">${yen(T.return * 10)}</div><div class="small">的中 ${T.hits}レース</div></div>
        <div class="panel rv" style="--i:2"><h4>収支</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px);color:${T.return >= T.stake ? "var(--hit)" : "var(--muted)"}">${signedYen((T.return - T.stake) * 10)}</div><div class="small">払戻 − 投資</div></div>
        <div class="panel rv" style="--i:3"><h4>回収率</h4><div class="big">${roi.toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">払戻 ÷ 投資</div></div>
      </div>
      ${myRecordHtml(days)}
      ${streakHtml(rec.streaks)}
      ${T.ev_races ? `<div class="section-head" style="margin-top:40px"><div><h2 class="section-title">試験中：オッズで絞った買い目<small>締切前のオッズで「MINAMOの確率（市場と合わせて補正）×オッズ」が1.2以上の組だけを最大9点（無ければ見送り。10/4までは最大6点）。実際の推奨買い目は変えず、成績だけを数えています</small></h2></div></div>
      <div class="calib">
        <div class="panel rv"><h4>的中率</h4><div class="big">${T.ev_bought ? ((T.ev_hits / T.ev_bought) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">的中 ${T.ev_hits}/${T.ev_bought}R（買ったレースのうち）</div></div>
        <div class="panel rv" style="--i:1"><h4>回収率</h4><div class="big" style="color:var(--accent)">${T.ev_stake ? ((T.ev_return / T.ev_stake) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">払戻 ÷ 投資</div></div>
        <div class="panel rv" style="--i:2"><h4>買ったレース</h4><div class="big">${T.ev_bought}<small style="font-size:.45em">R</small></div><div class="small">見送り ${T.ev_races - T.ev_bought}R · 平均 ${T.ev_bought ? (T.ev_stake / 100 / T.ev_bought).toFixed(1) : "--"}点</div></div>
        <div class="panel rv" style="--i:3"><h4>1点1,000円の収支</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px);color:${T.ev_return >= T.ev_stake ? "var(--hit)" : "var(--muted)"}">${signedYen((T.ev_return - T.ev_stake) * 10)}</div><div class="small">投資 ${yen(T.ev_stake * 10)} · 払戻 ${yen(T.ev_return * 10)}</div></div>
      </div>
      ${evTrendHtml(rec.days)}` : ""}
      ${T.co_races ? `<div class="section-head" style="margin-top:40px"><div><h2 class="section-title">試験中：3連単の合成オッズ配分<small>${PICK_KIND.co.rule}。1レースの投資は同じ（1,000円で数えます）</small></h2></div></div>
      <div class="calib">
        <div class="panel rv"><h4>的中率</h4><div class="big">${T.co_bought ? ((T.co_hits / T.co_bought) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">的中 ${T.co_hits}/${T.co_bought}R（買ったレースのうち）</div></div>
        <div class="panel rv" style="--i:1"><h4>回収率</h4><div class="big" style="color:var(--accent)">${T.co_stake ? ((T.co_return / T.co_stake) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">払戻 ÷ 投資</div></div>
        <div class="panel rv" style="--i:2"><h4>買ったレース</h4><div class="big">${T.co_bought}<small style="font-size:.45em">R</small></div><div class="small">見送り ${T.co_races - T.co_bought}R</div></div>
        <div class="panel rv" style="--i:3"><h4>1レース1,000円の収支</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px);color:${T.co_return >= T.co_stake ? "var(--hit)" : "var(--muted)"}">${signedYen((T.co_return - T.co_stake) * 10)}</div><div class="small">投資 ${yen(T.co_stake * 10)} · 払戻 ${yen(T.co_return * 10)}</div></div>
      </div>
      ${evTrendHtml(rec.days, "co", "合成オッズ配分")}` : ""}
      ${T.ex_races ? `<div class="section-head" style="margin-top:40px"><div><h2 class="section-title">試験中：2連単の買い目<small>${PICK_KIND.ex.rule}（無ければ見送り）。実際の推奨買い目は変えず、成績だけを数えています</small></h2></div></div>
      <div class="calib">
        <div class="panel rv"><h4>的中率</h4><div class="big">${T.ex_bought ? ((T.ex_hits / T.ex_bought) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">的中 ${T.ex_hits}/${T.ex_bought}R（買ったレースのうち）</div></div>
        <div class="panel rv" style="--i:1"><h4>回収率</h4><div class="big" style="color:var(--accent)">${T.ex_stake ? ((T.ex_return / T.ex_stake) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">払戻 ÷ 投資</div></div>
        <div class="panel rv" style="--i:2"><h4>買ったレース</h4><div class="big">${T.ex_bought}<small style="font-size:.45em">R</small></div><div class="small">見送り ${T.ex_races - T.ex_bought}R · 平均 ${T.ex_bought ? (T.ex_stake / 100 / T.ex_bought).toFixed(1) : "--"}点</div></div>
        <div class="panel rv" style="--i:3"><h4>1点1,000円の収支</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px);color:${T.ex_return >= T.ex_stake ? "var(--hit)" : "var(--muted)"}">${signedYen((T.ex_return - T.ex_stake) * 10)}</div><div class="small">投資 ${yen(T.ex_stake * 10)} · 払戻 ${yen(T.ex_return * 10)}</div></div>
      </div>
      ${evTrendHtml(rec.days, "ex", "試験中の2連単")}` : ""}
      ${T.ml_races ? `<div class="section-head" style="margin-top:40px"><div><h2 class="section-title">予想エンジンの比較<small>同じレースで、それぞれの本命（1着確率1位）が1着になった割合</small></h2></div></div>
      <div class="calib">
        <div class="panel rv"><h4>機械学習</h4><div class="big" style="color:var(--accent)">${((T.ml_fav_hits / T.ml_races) * 100).toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">本命1着 · ${T.ml_races}R</div></div>
        <div class="panel rv" style="--i:1"><h4>統計モデル</h4><div class="big">${((T.shadow_fav_hits / T.ml_races) * 100).toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">本命1着 · ${T.ml_races}R</div></div>
      </div>` : ""}
      ${T.alt_races ? `<div class="section-head" style="margin-top:40px"><div><h2 class="section-title">買い目の組み方<small>予想手順（逃げるか→展開→相手）で組んだ6点と、確率の高い順の6点を、同じレースで比べた成績</small></h2></div></div>
      <div class="calib">
        <div class="panel rv"><h4>予想手順</h4><div class="big" style="color:var(--accent)">${((T.method_return / T.method_stake) * 100 || 0).toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">回収率 · 的中 ${T.method_hits}/${T.alt_races}R</div></div>
        <div class="panel rv" style="--i:1"><h4>確率上位6点</h4><div class="big">${((T.alt_return / T.alt_stake) * 100 || 0).toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">回収率 · 的中 ${T.alt_hits}/${T.alt_races}R</div></div>
      </div>` : ""}
      <div class="section-head" style="margin-top:40px"><div><h2 class="section-title">日別・場別の収支<small>推奨買い目を1点1,000円で買った場合。日付を押すと、その日の場ごと・レースごとの成績が出ます</small></h2></div></div>
      <div class="panel ledger rv">${ledgerDays(days)}</div>
      <div id="ledger-day"></div>
      <div class="section-head" style="margin-top:40px"><div><h2 class="section-title">確信度別の成績<small>確信度の帯ごとの成績。数字が高いレースほど当たっているかを検証</small></h2></div></div>
      <div class="calib">${Object.entries(tiers).map(([k, [n, h1, h3]], i) => `<div class="panel rv" style="--i:${i}"><h4>${k}</h4><div class="big">${n ? ((h3 / n) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">3連単的中 · 本命1着 ${n ? ((h1 / n) * 100).toFixed(1) : "--"}% · ${n}R</div></div>`).join("")}</div>
    </section>
  </div>`;
  bindLedger(days);
}

/* ------------------------------------------------------------ あなたの予想の成績（このブラウザに保存した分） */
function myRecordHtml(days) {
  let keys = [];
  try { keys = Object.keys(localStorage).filter((k) => k.startsWith("minamo-my-")); } catch { return ""; }
  const byKey = {};
  for (const d of days) for (const v of d.venues) for (const r of v.races) byKey[`minamo-my-${d.date}-${v.jcd}-${r.rno}`] = r;
  let n = 0, pts = 0, hits = 0, ret = 0, mStake = 0, mRet = 0, mHits = 0;
  for (const k of keys) {
    const r = byKey[k];
    if (!r || !r.result || r.cancelled) continue;
    let f;
    try { f = JSON.parse(localStorage.getItem(k)).f; } catch { continue; }
    const c = myCombos(f);
    if (!c.length) continue;
    n++; pts += c.length;
    if (c.includes(r.result)) { hits++; ret += r.payout || 0; }
    if (r.stake != null) { mStake += r.stake; mRet += r.return || 0; mHits += r.hit ? 1 : 0; }
  }
  if (!n) return "";
  return `<div class="section-head" style="margin-top:40px"><div><h2 class="section-title">あなたの予想<small>レース画面の「あなたの予想と比べる」で選んだ組（このブラウザに保存した分）を、同じレースのMINAMOの推奨買い目と比べた成績。1点1,000円</small></h2></div></div>
    <div class="calib">
      <div class="panel rv"><h4>あなた</h4><div class="big">${((ret * 100) / (pts * 100) * 100).toFixed(1)}<small style="font-size:.45em">%</small></div><div class="small">回収率 · 的中 ${hits}/${n}R · 平均${(pts / n).toFixed(1)}点</div></div>
      <div class="panel rv" style="--i:1"><h4>あなたの収支</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px)">${signedYen((ret - pts * 100) * BET_UNIT)}</div><div class="small">投資 ${yen(pts * 1000)}</div></div>
      <div class="panel rv" style="--i:2"><h4>MINAMO（同じレース）</h4><div class="big">${mStake ? ((mRet / mStake) * 100).toFixed(1) : "--"}<small style="font-size:.45em">%</small></div><div class="small">回収率 · 的中 ${mHits}/${n}R</div></div>
      <div class="panel rv" style="--i:3"><h4>MINAMOの収支</h4><div class="big" style="white-space:nowrap;font-size:clamp(22px,3vw,36px)">${mStake ? signedYen((mRet - mStake) * BET_UNIT) : "--"}</div><div class="small">推奨買い目</div></div>
    </div>`;
}

/* ------------------------------------------------------------ 試験中の買い目の回収率の推移（累計） */
function evTrendHtml(days, k = "ev", label = "試験中の買い目") {
  const D = days.filter((d) => d[`${k}_stake`] > 0);
  if (!D.length) return "";
  let es = 0, er = 0, ps = 0, pr = 0;
  const pts = D.map((d) => {
    es += d[`${k}_stake`]; er += d[`${k}_return`] || 0; ps += d.stake || 0; pr += d.return || 0;
    return { date: d.date, ev: (er / es) * 100, pk: ps ? (pr / ps) * 100 : null, evDay: ((d[`${k}_return`] || 0) / d[`${k}_stake`]) * 100, n: d[`${k}_bought`] };
  });
  // 画面の幅に合わせて描く（スマホで文字が小さくならないように）
  const W = Math.round(Math.min(800, Math.max(300, ($("#main")?.clientWidth || 800) - 90))), H = 260, padL = 40, padR = 86, padT = 16, padB = 34;
  const vals = pts.flatMap((p) => [p.ev, p.pk]).filter((v) => v != null);
  const lo = Math.min(50, Math.floor(Math.min(...vals) / 10) * 10), hi = Math.max(150, Math.ceil(Math.max(...vals) / 10) * 10);
  const x = (i) => padL + (pts.length === 1 ? (W - padL - padR) / 2 : (i * (W - padL - padR)) / (pts.length - 1));
  const y = (v) => padT + ((hi - v) / (hi - lo)) * (H - padT - padB);
  const path = (k) => pts.map((p, i) => (p[k] == null ? "" : `${i && pts[i - 1][k] != null ? "L" : "M"}${x(i).toFixed(1)} ${y(p[k]).toFixed(1)}`)).join(" ");
  const grid = [lo, 100, hi].filter((v, i, a) => a.indexOf(v) === i).map((v) => `<line class="${v === 100 ? "ref100" : "gridl"}" x1="${padL}" x2="${W - padR}" y1="${y(v)}" y2="${y(v)}"/><text x="${padL - 6}" y="${y(v) + 3}" text-anchor="end">${v}%</text>`).join("");
  const step = Math.max(1, Math.ceil(pts.length / Math.max(3, Math.floor(W / 100))));
  const xl = pts.map((p, i) => (i % step === 0 || i === pts.length - 1 ? `<text x="${x(i)}" y="${H - padB + 16}" text-anchor="middle">${p.date.slice(4, 6)}/${p.date.slice(6)}</text>` : "")).join("");
  const last = pts[pts.length - 1];
  const bw = pts.length > 1 ? (W - padL - padR) / (pts.length - 1) : 40;
  const hits = pts.map((p, i) => `<rect class="hit" x="${x(i) - bw / 2}" y="${padT}" width="${bw}" height="${H - padT - padB}"><title>${fmtDate(p.date)}  ${label} 累計 ${p.ev.toFixed(1)}%（この日 ${p.evDay.toFixed(1)}%・${p.n}R）${p.pk != null ? ` ／ 推奨買い目 累計 ${p.pk.toFixed(1)}%` : ""}</title></rect>`).join("");
  const rows = pts.map((p) => `<tr><td>${fmtDate(p.date)}</td><td>${p.n}R</td><td>${p.evDay.toFixed(1)}%</td><td>${p.ev.toFixed(1)}%</td><td>${p.pk != null ? p.pk.toFixed(1) + "%" : "--"}</td></tr>`).join("");
  return `<div class="panel chart trend rv">
    <div class="section-head" style="margin:0 0 8px"><span class="eyebrow">回収率の推移（はじめからの累計）</span>
      <span class="legend"><span class="key ev"></span>${label} <span class="key pk"></span>推奨買い目（3連単・確率上位）</span></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}と推奨買い目の、累計回収率の推移">
      ${grid}${xl}
      <path class="ln pk" d="${path("pk")}"/><path class="ln ev" d="${path("ev")}"/>
      ${pts.map((p, i) => `<circle class="pt ev" cx="${x(i)}" cy="${y(p.ev)}" r="4"/>`).join("")}
      ${last.pk != null ? `<text class="lab" x="${x(pts.length - 1) + 8}" y="${y(last.pk) + 4}">推奨 ${last.pk.toFixed(0)}%</text>` : ""}
      <text class="lab strong" x="${x(pts.length - 1) + 8}" y="${y(last.ev) + 4}">${k === "ex" ? "2連単" : "試験中"} ${last.ev.toFixed(0)}%</text>
      ${hits}
    </svg>
    <details class="small"><summary>数字の表で見る</summary><div class="ledger-scroll"><table class="streak-t"><thead><tr><th>日付</th><th>買ったレース</th><th>その日の回収率</th><th>試験中 累計</th><th>推奨買い目 累計</th></tr></thead><tbody>${rows}</tbody></table></div></details>
  </div>`;
}

/* ------------------------------------------------------------ 連敗の記録（締切順） */
const raceLabel = (l) => (l ? `${l.slice(4, 6)}/${l.slice(6, 8)} ${esc(l.slice(9))}` : "--");
function streakBlock(name, S) {
  if (!S || !S.races) return "";
  const runs = S.buckets.reduce((a, b) => a + b.count, 0);
  const buckets = S.buckets.map((b) => `<tr><td>${b.label === "0" ? "0（続けて的中）" : b.label + "連敗"}</td><td>${b.count}回</td><td>${rate(b.count, runs)}</td></tr>`).join("");
  const recent = (S.recent || []).map((x) => `<tr><td>${x.len}連敗</td><td>${raceLabel(x.end)}</td><td>${yen(x.payout)}</td><td>${yen(x.martingale)}</td></tr>`).join("");
  return `<h3 class="streak-name">${name}<small class="muted"> ${S.races}レース・的中 ${S.hits}</small></h3>
    <div class="calib">
      <div class="panel rv"><h4>今の連敗</h4><div class="big" style="color:${S.current ? "var(--accent)" : "var(--hit)"}">${S.current}<small style="font-size:.45em">連敗</small></div><div class="small">${S.current ? `${raceLabel(S.current_from)} から` : "直前のレースは的中"}</div></div>
      <div class="panel rv" style="--i:1"><h4>最大連敗</h4><div class="big">${S.max}<small style="font-size:.45em">連敗</small></div><div class="small">${raceLabel(S.max_from)} 〜 ${S.max_to ? raceLabel(S.max_to) + " で的中" : "継続中"}</div></div>
      <div class="panel rv" style="--i:2"><h4>倍賭けの必要資金</h4><div class="big" style="white-space:nowrap;font-size:clamp(20px,2.6vw,32px)">${yen(S.max_martingale)}</div><div class="small">はずれたら次は2倍、当たったら元に戻す（1点1,000円から）。一番長い連敗の間に投じた合計</div></div>
    </div>
    <div class="streak-grid">
      <div class="panel ledger rv"><div class="ledger-scroll"><table class="streak-t">
        <thead><tr><th>当たるまで</th><th>回数</th><th>割合</th></tr></thead><tbody>${buckets}</tbody></table></div></div>
      ${recent ? `<div class="panel ledger rv"><div class="ledger-scroll"><table class="streak-t">
        <thead><tr><th>連敗</th><th>止めたレース</th><th>配当</th><th>倍賭けの合計</th></tr></thead><tbody>${recent}</tbody></table></div></div>` : ""}
    </div>`;
}
function streakHtml(st) {
  if (!st || !(st.picks && st.picks.races)) return "";
  return `<div class="section-head" style="margin-top:40px"><div><h2 class="section-title">連敗の記録（締切順）<small>全場のレースを締切の早い順に並べ、当たるまでに何レース続けてはずれたか。中止・見送りのレースは数えません</small></h2></div></div>
    ${streakBlock("推奨買い目", st.picks)}
    ${streakBlock("試験中：オッズで絞った買い目（買ったレースだけ）", st.ev)}`;
}

/* ------------------------------------------------------------ ledger（日別・場別の収支） */
const BET_UNIT = 10;  // 保存は1点100円。サイトの表示は1点1,000円にそろえる
function sumRaces(races) {
  const t = { races: 0, hits: 0, stake: 0, ret: 0, known: true };
  for (const r of races) {
    if (!r.result || r.cancelled) continue;
    t.races++;
    t.hits += r.hit ? 1 : 0;
    if (r.stake == null) { t.known = false; continue; }
    t.stake += r.stake * BET_UNIT; t.ret += (r.return || 0) * BET_UNIT;
  }
  return t;
}
const plus = (n) => `<span class="${n > 0 ? "pos" : n < 0 ? "neg" : ""}">${n > 0 ? "+" : n < 0 ? "−" : "±"}¥${Math.abs(n).toLocaleString("ja-JP")}</span>`;
const rate = (a, b) => (b ? ((a / b) * 100).toFixed(1) + "%" : "--");
function ledgerRow(label, t, attrs = "") {
  const money = t.stake ? `<td>${yen(t.stake)}</td><td>${yen(t.ret)}</td><td>${plus(t.ret - t.stake)}</td><td>${rate(t.ret, t.stake)}</td>` : `<td class="muted" colspan="4">--</td>`;
  return `<tr ${attrs}><th scope="row">${label}</th><td>${t.races}</td><td>${t.hits}</td><td>${rate(t.hits, t.races)}</td>${money}</tr>`;
}
const LEDGER_HEAD = `<tr><th></th><th>レース</th><th>的中</th><th>的中率</th><th>投資</th><th>払戻</th><th>収支</th><th>回収率</th></tr>`;
function ledgerDays(days) {
  const rows = days.slice().reverse().map((d) => {
    const t = sumRaces(d.venues.flatMap((v) => v.races));
    return ledgerRow(`<button class="linkish" data-ledger="${d.date}">${fmtDate(d.date)}</button>`, t, `data-row="${d.date}"`);
  }).join("");
  return `<div class="ledger-scroll"><table class="ledger-t"><thead>${LEDGER_HEAD}</thead><tbody>${rows || `<tr><td colspan="8" class="muted">まだ結果がありません</td></tr>`}</tbody></table></div>`;
}
function ledgerDay(day) {
  const venues = day.venues.map((v, i) => {
    const t = sumRaces(v.races);
    const races = v.races.map((r) => {
      const done = r.result && !r.cancelled;
      const profit = done && r.stake != null ? ((r.return || 0) - r.stake) * BET_UNIT : null;
      return `<tr class="${r.hit ? "hitrow" : ""}">
        <td><a href="#/race/${day.date}/${v.jcd}/${r.rno}">${r.rno}R</a></td>
        <td class="num">${r.cancelled ? "中止" : esc(r.result || "--")}</td>
        <td>${done ? yen(r.payout) : "--"}${r.popularity ? `<small class="muted"> ${r.popularity}人気</small>` : ""}</td>
        <td class="num picks">${(r.picks || []).map((c) => `<span class="${c === r.result ? "on" : ""}">${esc(c)}</span>`).join(" ") || "--"}</td>
        <td>${r.pick_no ? `<b class="pos">${r.pick_no}点目</b>` : done ? "×" : "--"}</td>
        <td>${r.model_rank ? `${r.model_rank}番目` : done ? "41番目以下" : "--"}</td>
        <td>${r.stake != null ? yen(r.stake * BET_UNIT) : "--"}</td>
        <td>${done ? yen((r.return || 0) * BET_UNIT) : "--"}</td>
        <td>${profit == null ? "--" : plus(profit)}</td>
      </tr>`;
    }).join("");
    return `<details class="ledger-venue" ${i === 0 ? "open" : ""}>
      <summary><table class="ledger-t"><tbody>${ledgerRow(`${esc(v.name)}<small class="muted"> ${esc(v.day_label || "")}</small>`, t)}</tbody></table></summary>
      <div class="ledger-scroll"><table class="ledger-t races"><thead><tr><th>R</th><th>結果</th><th>配当</th><th>買い目</th><th>何点目で的中</th><th>予想の順位</th><th>投資</th><th>払戻</th><th>収支</th></tr></thead><tbody>${races}</tbody></table></div>
    </details>`;
  }).join("");
  const all = sumRaces(day.venues.flatMap((v) => v.races));
  return `<div class="section-head" style="margin-top:28px"><div><h3 class="ledger-title">${fmtDate(day.date)} の場別成績</h3><p class="muted small">「何点目で的中」は推奨買い目の何点目が当たったか、「予想の順位」は3連単120通りを確率の高い順に並べたとき何番目だったか</p></div></div>
    <div class="panel ledger"><div class="ledger-scroll"><table class="ledger-t"><thead>${LEDGER_HEAD}</thead><tbody>${ledgerRow("合計", all)}</tbody></table></div>${venues}</div>`;
}
function bindLedger(days) {
  const byDate = Object.fromEntries(days.map((d) => [d.date, d]));
  const show = (date) => {
    const day = byDate[date];
    if (!day) return;
    $("#ledger-day").innerHTML = ledgerDay(day);
    document.querySelectorAll("[data-row]").forEach((tr) => tr.classList.toggle("sel", tr.dataset.row === date));
  };
  document.querySelectorAll("[data-ledger]").forEach((b) => b.addEventListener("click", () => show(b.dataset.ledger)));
  const settled = days.filter((d) => sumRaces(d.venues.flatMap((v) => v.races)).races > 0);
  if (settled.length) show(settled[settled.length - 1].date);
}

/* ------------------------------------------------------------ about */
function renderAbout() {
  $("#main").innerHTML = `
  <div class="wrap">
    <section class="section">
      <span class="eyebrow">MINAMOについて</span>
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
    const txt = left <= 0 ? "締切" : fmtCountdown(left);
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
      else if (state.route.name === "picks" && state.date === todayJst() && !document.activeElement?.closest("#stakeForm")) await renderPicks(true);
      else if (state.route.name === "race" && state.race && !state.race.result && nowMs() > deadlineMs(state.race.date, state.race.deadline) - 35 * 60e3) {
        const before = state.race.updated_at;
        await renderRace(state.route, true);
        if (state.race.updated_at !== before) toast("直前情報を反映しました");
      }
    } catch (e) { console.warn(e); }
  }, 30000);
}

boot();
