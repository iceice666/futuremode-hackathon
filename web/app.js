/* 記憶帳本 — the Mneme timeline, against the live API on this Orin.
 *
 * React + htm over vendored UMD builds rather than a bundler: JetPack 6.2 ships
 * Node 12, which no current build tool supports, and vendoring keeps the page
 * working when the demo unplugs the network.
 *
 * API surface used (spec.md §2):
 *   GET  /api/health               status line, model names, offline flag
 *   GET  /api/events?limit&cursor  the ledger, newest first
 *   GET  /api/stream               SSE, one "observed" frame per new event
 *   GET  /api/frames/{id}/thumb    the still behind an event
 *   POST /api/ask                  a question, answered with citations
 */
'use strict';

const { useState, useEffect, useRef, useCallback, useMemo, Fragment } = React;
const html = htm.bind(React.createElement);

const PAGE = 200;
const LIVE_MS = 2000;         // still-frame fallback poll, when the stream will not open
const STREAM_RETRY_MS = 20000; // ...and how often to try the stream again
// Time-lapse rates, in frames per second. 4 is the default: the room changes
// slowly enough that anything below this reads as a slideshow, and the thumbs
// are small enough that 16 still decodes in time on this box.
const RATES = [2, 4, 8, 16];
const MONO = "var(--mono)";
const ACCENT = "var(--accent)";
const DIM = "var(--dim)";

// -- helpers -------------------------------------------------------------

/** Events carry UTC (spec.md §0); the timeline is a wall clock, so render local. */
const local = (iso) => new Date(iso);
const hm = (iso) => {
  const d = local(iso);
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
};
const hms = (iso) => hm(iso) + ':' + String(local(iso).getSeconds()).padStart(2, '0');
const minuteOfDay = (iso) => { const d = local(iso); return d.getHours() * 60 + d.getMinutes(); };
const nowMinute = () => { const d = new Date(); return d.getHours() * 60 + d.getMinutes(); };

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).error?.message || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}

// -- data hooks ----------------------------------------------------------

/** Health drives the status bar. Polled: it reports uptime, fps and the
 *  offline probe, all of which move without any event being emitted. */
function useHealth() {
  const [health, setHealth] = useState(null);
  const [reachable, setReachable] = useState(true);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const h = await getJSON('/api/health');
        if (alive) { setHealth(h); setReachable(true); }
      } catch (_) {
        if (alive) setReachable(false);
      }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return { health, reachable };
}

/** The ledger: one page of history, then SSE for anything new.
 *  Sorted newest-first to match /api/events, and deduped by id because a
 *  reconnecting stream can replay an event the initial page already had. */
function useEvents() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    getJSON(`/api/events?limit=${PAGE}`)
      .then((d) => { if (alive) { setEvents(d.events || []); setLoading(false); } })
      .catch((e) => { if (alive) { setError(e.message); setLoading(false); } });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    const es = new EventSource('/api/stream');
    es.addEventListener('observed', (m) => {
      let ev;
      try { ev = JSON.parse(m.data); } catch (_) { return; }
      if (!ev || !ev.id) return;
      setEvents((prev) => (
        prev.some((e) => e.id === ev.id)
          ? prev
          : [ev, ...prev].sort((a, b) => (a.ts < b.ts ? 1 : -1))
      ));
    });
    return () => es.close();
  }, []);

  return { events, loading, error };
}

// -- pieces --------------------------------------------------------------

function StatusBar({ health, reachable }) {
  const offline = health?.offline;
  // Three states, not two: the backend being unreachable is a different fact
  // from the backend telling us it has no internet, and only the second one is
  // the thing the demo is proving.
  const label = !reachable ? '後端無回應'
    : offline == null ? '偵測中'
    : offline ? 'OFFLINE · 無雲端連線'
    : 'ONLINE · 可連外網';
  const colour = !reachable ? '#8a5a5a' : offline ? ACCENT : '#6f7f6a';

  return html`
    <div style=${{ display: 'flex', alignItems: 'center', gap: 20, fontFamily: MONO, fontSize: 11, color: '#a89f8f', flexWrap: 'wrap' }}>
      <div style=${{ display: 'flex', alignItems: 'center', gap: 8, border: `1px solid ${offline ? '#33291a' : '#252520'}`, padding: '5px 10px' }}>
        <span style=${{ width: 6, height: 6, background: colour, animation: 'breathe 2.4s ease-in-out infinite' }}></span>
        <span style=${{ letterSpacing: '.12em', whiteSpace: 'nowrap', color: colour }}>${label}</span>
      </div>
      <div style=${{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 14 }}>
        ${[0.35, 0.6, 0.45, 0.8, 0.55, 1, 0.5, 0.7].map((n, i) => html`
          <span key=${i} style=${{ width: 3, height: 4 + n * 10, background: `rgba(var(--accent-rgb), ${0.3 + n * 0.6})`, animation: `breathe ${2 + i * 0.2}s ease-in-out infinite` }}></span>
        `)}
      </div>
      <span style=${{ color: ACCENT, whiteSpace: 'nowrap' }}>
        ${health ? `${health.capture_fps?.toFixed(2) ?? '0.00'} fps` : '—'}
      </span>
      <span style=${{ whiteSpace: 'nowrap' }} title=${health ? `VLM ${health.vlm_model} · LLM ${health.llm_model} · EMBED ${health.embed_model}` : ''}>
        ${health ? `${health.sidecar} · ${health.mode} · ${health.event_count} 事件` : '—'}
      </span>
    </div>
  `;
}

/** The viewer has two modes. Live plays the camera; picking an event pins it
 *  to that snapshot.
 *
 *  Live is a motion-JPEG stream (spec.md 2.8), not a poll: the camera runs at
 *  21fps and `<img src>` plays multipart/x-mixed-replace with no JavaScript,
 *  no codec and no buffer, so what is on screen is the room as it is now. The
 *  VLM still only describes a couple of frames a second -- the picture is live,
 *  the memory is sampled, and those are different rates on purpose.
 *
 *  If the stream will not open -- seed-only mode, a backend restart, a camera
 *  hiccup -- it falls back to polling the newest kept frame and keeps retrying,
 *  because nobody is reloading the page in the middle of a demo. */
function Viewer({ event, live, cited, playing }) {
  const [failures, setFailures] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const streaming = live && failures === 0;
  const src = live
    ? (streaming ? '/api/frames/live.mjpg' : `/api/frames/latest/thumb?t=${attempt}`)
    : event?.thumb_url;

  useEffect(() => { setFailures(0); setAttempt(0); }, [live]);
  // Two clocks: retry the still every LIVE_MS so the panel keeps moving, and
  // try the stream again occasionally in case the camera came back.
  useEffect(() => {
    if (!live || streaming) return undefined;
    const id = setInterval(() => {
      setAttempt((n) => {
        if (n > 0 && n % Math.round(STREAM_RETRY_MS / LIVE_MS) === 0) setFailures(0);
        return n + 1;
      });
    }, LIVE_MS);
    return () => clearInterval(id);
  }, [live, streaming]);

  const broken = !live && failures > 0;
  if (!src) {
    return html`<div style=${{ position: 'relative', aspectRatio: '16 / 9', background: '#0a0907', border: '1px solid var(--edge)', display: 'grid', placeItems: 'center', color: DIM, fontFamily: MONO, fontSize: 12 }}>等待畫面</div>`;
  }
  return html`
    <div style=${{ position: 'relative', aspectRatio: '16 / 9', background: '#0a0907', border: `1px solid ${live || playing ? 'rgba(var(--accent-rgb),.4)' : cited ? 'rgba(var(--accent-rgb),.55)' : 'var(--edge)'}`, overflow: 'hidden' }}>
      ${broken
        ? html`<div style=${{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: DIM, fontFamily: MONO, fontSize: 12 }}>縮圖不存在</div>`
        // Keyed by src in snapshot mode so each pick re-runs the fade; NOT keyed
        // in live mode or during playback, where remounting on every frame would
        // flicker the panel -- swapping src on one element is what makes a run of
        // stills read as video rather than as a slideshow.
        : html`<img key=${live ? (streaming ? 'stream' : 'still') : playing ? 'play' : src} src=${src} alt=""
             onError=${() => setFailures((n) => n + 1)}
             style=${{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', animation: live || playing ? 'none' : 'rise .4s ease-out both' }} />`}
      <div style=${{ position: 'absolute', inset: 0, background: 'repeating-linear-gradient(rgba(0,0,0,.32) 0 1px, rgba(0,0,0,0) 1px 3px)', pointerEvents: 'none' }}></div>
      <div style=${{ position: 'absolute', inset: 0, boxShadow: 'inset 0 0 130px 34px rgba(0,0,0,.6)', pointerEvents: 'none' }}></div>
      <div style=${{ position: 'absolute', left: 0, right: 0, height: '16%', background: 'linear-gradient(rgba(var(--accent-rgb),.07), rgba(var(--accent-rgb),0))', animation: 'sweep 8s linear infinite', pointerEvents: 'none' }}></div>
      ${!live && !playing && html`<div key=${'cut' + src} style=${{ position: 'absolute', inset: 0, background: 'rgba(var(--accent-rgb),.16)', mixBlendMode: 'screen', pointerEvents: 'none', animation: 'cut .42s ease-out forwards' }}></div>`}
      <div style=${{ position: 'absolute', top: 12, left: 14, display: 'flex', alignItems: 'center', gap: 8, fontFamily: MONO, fontSize: 11, letterSpacing: '.16em', color: 'rgba(232,216,190,.85)' }}>
        ${live
          ? html`<${Fragment}>
              <span style=${{ width: 6, height: 6, borderRadius: '50%', background: '#d0563f', animation: 'breathe 1.6s ease-in-out infinite' }}></span>
              <span>LIVE · ${streaming ? '即時轉播 21fps' : '即時畫面(逐張)'}</span>
            <//>`
          : playing
            ? html`<${Fragment}>
                <span style=${{ width: 6, height: 6, background: ACCENT, animation: 'breathe 1s ease-in-out infinite' }}></span>
                <span>縮時播放 · ${hms(event.ts)}</span>
              <//>`
            : html`<span>回放 · ${hms(event.ts)} · ${event.source === 'seed' ? 'SEED' : 'VLM'}</span>`}
      </div>
    </div>
  `;
}

/** One bar for the whole day: the transport and the timeline are the same
 *  object, because they were always answering the same question -- where in the
 *  day am I looking, and what is there to look at.
 *
 *  Indexed by wall clock, so the shape of the bar is the shape of the day: the
 *  15-minute buckets are event density, the hairlines are individual snapshots,
 *  and the empty stretches are hours when the room sat still. Playback still
 *  advances event by event, so the playhead jumps across those gaps rather than
 *  crawling through them -- which is what a time-lapse looks like. */
function Transport({ events, ordered, current, citedIds, live, playing, rate,
                     onSeek, onToggle, onRate, onLive }) {
  const trackRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const n = ordered.length;
  const idx = current ? ordered.findIndex((e) => e.id === current.id) : -1;
  const at = idx < 0 ? n - 1 : idx;   // live pins the head of the strip
  const frame = ordered[at];

  const buckets = useMemo(() => {
    const out = new Array(96).fill(0).map(() => ({ n: 0, hit: false }));
    for (const e of events) {
      const b = Math.min(95, Math.floor(minuteOfDay(e.ts) / 15));
      out[b].n += 1;
      if (citedIds.has(e.id)) out[b].hit = true;
    }
    const peak = Math.max(1, ...out.map((b) => b.n));
    return out.map((b) => ({ ...b, w: b.n / peak }));
  }, [events, citedIds]);

  // Dragging is a request for a moment, and only the moments we have pictures
  // of can be shown, so the handle lands on the nearest event rather than
  // between two of them.
  const nearest = useCallback((minute) => {
    let best = -1, d = Infinity;
    ordered.forEach((e, i) => {
      const dd = Math.abs(minuteOfDay(e.ts) - minute);
      if (dd < d) { d = dd; best = i; }
    });
    return best;
  }, [ordered]);

  const move = useCallback((e) => {
    const r = trackRef.current?.getBoundingClientRect();
    if (!r || !r.width || !n) return;
    const p = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const i = nearest(p * 1440);
    if (i >= 0) onSeek(i);
  }, [n, nearest, onSeek]);

  const activeBucket = current ? Math.floor(minuteOfDay(current.ts) / 15) : -1;
  const headPct = current ? minuteOfDay(current.ts) / 1440 * 100 : null;
  const span = n ? `${hm(ordered[0].ts)} → ${hm(ordered[n - 1].ts)}` : '';

  return html`
    <div style=${{ display: 'grid', gridTemplateColumns: '34px minmax(0,1fr) auto', columnGap: 16, rowGap: 7, alignItems: 'center' }}>
      <button onClick=${onToggle} title="空白鍵播放/暫停" style=${{
        width: 34, height: 34, display: 'grid', placeItems: 'center',
        background: 'transparent', cursor: 'pointer',
        border: `1px solid rgba(var(--accent-rgb),${playing ? .7 : .4})`,
        color: ACCENT, fontFamily: MONO, fontSize: 12, lineHeight: 1,
        animation: playing ? 'glow 2.6s ease-in-out infinite' : 'none'
      }}>${playing ? '❙❙' : '▶'}</button>

      <div ref=${trackRef}
        onPointerDown=${(e) => { try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_) {} setDragging(true); move(e); }}
        onPointerMove=${(e) => dragging && move(e)}
        onPointerUp=${() => setDragging(false)}
        onPointerCancel=${() => setDragging(false)}
        style=${{ position: 'relative', height: 44, display: 'flex', alignItems: 'center', gap: 2, cursor: 'ew-resize', touchAction: 'none' }}>
        ${buckets.map((b, i) => html`
          <span key=${i} style=${{
            flex: '1 1 0',
            height: b.w * 34 + 5,
            borderRadius: 1,
            transformOrigin: 'bottom',
            background: b.hit ? 'rgba(var(--accent-rgb),.95)'
              : i === activeBucket ? 'rgba(var(--accent-rgb),.75)'
              : `rgba(var(--accent-rgb), ${0.1 + b.w * 0.22})`,
            animation: b.hit ? 'hit 1.1s ease-out infinite' : 'none',
            transition: 'background .18s linear'
          }}></span>
        `)}
        ${/* One hairline per snapshot, so the bar reads as "when the room was
              described" rather than only as density. Citations from the last
              question sit on the same axis, so an answer can be scrubbed to. */
          events.map((e) => html`
          <span key=${'t' + e.id} style=${{
            position: 'absolute',
            left: `${minuteOfDay(e.ts) / 1440 * 100}%`,
            ...(citedIds.has(e.id)
              ? { top: 2, height: 14, width: 2, marginLeft: -1, background: 'rgba(var(--accent-rgb),.95)', animation: 'hit 1.1s ease-out infinite' }
              : { top: 2, height: 7, width: 1, background: 'rgba(var(--accent-rgb),.4)' }),
            pointerEvents: 'none'
          }}></span>
        `)}
        ${!live && headPct !== null && html`<span style=${{ position: 'absolute', top: 0, bottom: 0, left: `${headPct}%`, width: 2, marginLeft: -1, background: ACCENT, pointerEvents: 'none', boxShadow: '0 0 8px rgba(var(--accent-rgb),.7)', animation: playing ? 'none' : 'glow 2.6s ease-in-out infinite', transition: dragging ? 'none' : 'left .12s cubic-bezier(.2,.8,.2,1)' }}></span>`}
        ${live && html`<span style=${{ position: 'absolute', top: 0, bottom: 0, left: `${nowMinute() / 1440 * 100}%`, width: 1, background: 'rgba(208,86,63,.8)', pointerEvents: 'none' }}></span>`}
      </div>

      <div style=${{ display: 'flex', gap: 4 }}>
        ${RATES.map((r) => html`
          <button key=${r} onClick=${() => onRate(r)} title=${`每秒 ${r} 格`} style=${{
            padding: '4px 7px', background: 'transparent', cursor: 'pointer',
            border: `1px solid ${r === rate ? 'rgba(var(--accent-rgb),.6)' : 'var(--line)'}`,
            color: r === rate ? ACCENT : DIM, fontFamily: MONO, fontSize: 10.5, letterSpacing: '.06em'
          }}>${r}×</button>
        `)}
      </div>

      ${/* Second row, middle column only: the hour ruler belongs to the track,
            so the grid keeps the labels over the times they mark. */ ''}
      <div style=${{ gridColumn: 2, display: 'flex', justifyContent: 'space-between', fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.14em' }}>
        ${['00', '04', '08', '12', '16', '20', '24'].map((h) => html`<span key=${h}>${h}</span>`)}
      </div>

      <div style=${{ gridColumn: '1 / -1', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.12em', borderTop: '1px solid var(--line-soft)', paddingTop: 9 }}>
        <span>${live ? '拖曳時間軸以回放記憶' : span}</span>
        <span style=${{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style=${{ color: playing ? ACCENT : DIM }}>
            ${live ? `${events.length} 個事件` : `${frame ? hms(frame.ts) : '—'} · ${Math.min(at + 1, n)}/${n} 格`}
          </span>
          <button onClick=${onLive} disabled=${live} style=${{
            fontFamily: MONO, fontSize: 11, letterSpacing: '.12em',
            padding: '5px 12px', cursor: live ? 'default' : 'pointer',
            background: 'transparent',
            border: `1px solid ${live ? '#252520' : 'rgba(var(--accent-rgb),.55)'}`,
            color: live ? DIM : ACCENT,
            transition: 'border-color .16s linear, color .16s linear'
          }}>${live ? '● 即時中' : '回到現在'}</button>
        </span>
      </div>
    </div>
  `;
}

/** Time ranges the ask box can scope a question to.
 *
 *  Relative ranges are computed at submit time, not at click time: "最近 10 分鐘"
 *  has to mean ten minutes before the question, not ten minutes before whenever
 *  the chip happened to be pressed. */
const RANGES = [
  { key: 'all', label: '全部' },
  { key: '10m', label: '10 分', minutes: 10 },
  { key: '1h', label: '1 時', minutes: 60 },
  { key: '6h', label: '6 時', minutes: 360 },
  { key: 'custom', label: '自訂' },
];

/** `HH:MM` typed by a human, as a UTC instant on today's local date. The API
 *  takes UTC (spec.md §0); the person typing is looking at a wall clock. */
const atToday = (hhmm) => {
  if (!/^\d{2}:\d{2}$/.test(hhmm || '')) return null;
  const [h, m] = hhmm.split(':').map(Number);
  const d = new Date();
  d.setHours(h, m, 0, 0);
  return d;
};

function askScope(rangeKey, from, to) {
  const range = RANGES.find((r) => r.key === rangeKey);
  if (!range || range.key === 'all') return { body: {}, label: '' };
  if (range.minutes) {
    const since = new Date(Date.now() - range.minutes * 60000);
    return { body: { since: since.toISOString() }, label: `${range.label}內` };
  }
  let a = atToday(from), b = atToday(to);
  if (a && b && a > b) [a, b] = [b, a];   // typed backwards; take what they meant
  if (!a && !b) return { body: {}, label: '' };
  return {
    body: { ...(a ? { since: a.toISOString() } : {}), ...(b ? { until: b.toISOString() } : {}) },
    label: `${a ? hm(a.toISOString()) : ''}–${b ? hm(b.toISOString()) : ''}`,
  };
}

/** The ask panel. The reference filtered client-side; the real system answers
 *  semantically over embeddings, so the box posts to /api/ask and the answer's
 *  citations then light up the ledger and the timeline.
 *
 *  The range chips scope retrieval to a window. Without one, a question about a
 *  thing that has been on the desk all day is answered from whichever frame the
 *  embedding liked best, which may be eight hours old -- the backend narrows to
 *  the newest frames by itself only when the question says 現在/剛剛/目前. */
function AskPanel({ onCitations, onPick, events }) {
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [range, setRange] = useState('all');
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [scopeLabel, setScopeLabel] = useState('');

  const submit = async (e) => {
    e.preventDefault();
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true); setError(null);
    const scope = askScope(range, from, to);
    try {
      const d = await getJSON('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, ...scope.body })
      });
      setResult(d);
      setScopeLabel(scope.label);
      onCitations((d.citations || []).map((c) => c.event_id));
      const first = (d.citations || [])[0];
      if (first) {
        const match = events.find((ev) => ev.id === first.event_id);
        if (match) onPick(match);
      }
    } catch (err) {
      setError(err.message);
      setResult(null);
      onCitations([]);
    } finally {
      setBusy(false);
    }
  };

  const clear = () => { setQ(''); setResult(null); setError(null); onCitations([]); };

  const chip = (on) => ({
    padding: '3px 8px', background: 'transparent', cursor: 'pointer',
    border: `1px solid ${on ? 'rgba(var(--accent-rgb),.6)' : 'var(--line)'}`,
    color: on ? ACCENT : DIM, fontFamily: MONO, fontSize: 10.5, letterSpacing: '.06em',
  });

  return html`
    <form onSubmit=${submit} style=${{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style=${{ display: 'flex', alignItems: 'center', gap: 9, padding: '0 2px 8px' }}>
        <span style=${{ fontFamily: MONO, fontSize: 12, color: ACCENT }}>?</span>
        <input class="ask" value=${q} onInput=${(e) => setQ(e.target.value)}
          placeholder="問這個空間的記憶,例如「我的杯子在哪」"
          style=${{ flex: 1, minWidth: 0, background: 'transparent', border: 0, color: 'var(--ink)', fontFamily: MONO, fontSize: 12.5 }} />
        ${(result || error) && html`<span onClick=${clear} style=${{ fontFamily: MONO, fontSize: 11, color: DIM, cursor: 'pointer' }}>清除</span>`}
        ${busy && html`<span style=${{ width: 10, height: 10, border: '1px solid rgba(var(--accent-rgb),.3)', borderTopColor: ACCENT, borderRadius: '50%', animation: 'spin .7s linear infinite' }}></span>`}
      </div>

      <div style=${{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', borderBottom: '1px solid var(--edge)', padding: '0 2px 10px' }}>
        <span style=${{ fontFamily: MONO, fontSize: 10.5, color: DIM, letterSpacing: '.12em', marginRight: 2 }}>範圍</span>
        ${RANGES.map((r) => html`
          <button key=${r.key} type="button" onClick=${() => setRange(r.key)}
            style=${chip(range === r.key)}>${r.label}</button>
        `)}
        ${range === 'custom' && html`
          <span style=${{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 2 }}>
            <input type="time" value=${from} onInput=${(e) => setFrom(e.target.value)}
              style=${{ background: 'transparent', border: '1px solid var(--line)', color: 'var(--ink)', fontFamily: MONO, fontSize: 10.5, padding: '2px 4px', colorScheme: 'dark' }} />
            <span style=${{ fontFamily: MONO, fontSize: 10.5, color: DIM }}>–</span>
            <input type="time" value=${to} onInput=${(e) => setTo(e.target.value)}
              style=${{ background: 'transparent', border: '1px solid var(--line)', color: 'var(--ink)', fontFamily: MONO, fontSize: 10.5, padding: '2px 4px', colorScheme: 'dark' }} />
          </span>
        `}
      </div>

      ${error && html`<div style=${{ fontFamily: MONO, fontSize: 11.5, color: '#c98b6a', lineHeight: 1.6 }}>${error}</div>`}
      ${result && html`
        <div style=${{ display: 'flex', flexDirection: 'column', gap: 6, padding: '2px 2px 0', animation: 'rise .3s ease-out both' }}>
          <div style=${{ fontSize: 14, lineHeight: 1.65, color: 'var(--ink)' }}>${result.answer}</div>
          <div style=${{ fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.1em' }}>
            ${result.citations?.length ? `${result.citations.length} 筆佐證 · ` : '無佐證 · '}${result.latency_ms} ms${scopeLabel ? ` · ${scopeLabel}` : ''}
          </div>
        </div>
      `}
    </form>
  `;
}

function Ledger({ events, current, citedIds, onPick }) {
  if (!events.length) {
    return html`<div style=${{ fontFamily: MONO, fontSize: 12, color: DIM, padding: '14px 4px' }}>尚無事件</div>`;
  }
  return html`
    <div style=${{ display: 'flex', flexDirection: 'column', maxHeight: 392, overflow: 'auto' }}>
      ${events.map((e) => {
        const active = current?.id === e.id;
        const cited = citedIds.has(e.id);
        return html`
          <div key=${e.id} class="row" onClick=${() => onPick(e)} style=${{
            display: 'flex', gap: 12, alignItems: 'baseline', padding: '9px 10px', cursor: 'pointer',
            background: active ? 'rgba(var(--accent-rgb),.1)' : 'transparent',
            borderLeft: `1px solid ${active ? 'rgba(var(--accent-rgb),.8)' : 'transparent'}`,
            transition: 'background .16s linear'
          }}>
            <span style=${{ fontFamily: MONO, fontSize: 11, letterSpacing: '.06em', color: cited ? ACCENT : DIM, minWidth: 44 }}>${hm(e.ts)}</span>
            <span style=${{ fontSize: 12.5, lineHeight: 1.55, color: active ? 'var(--ink)' : 'rgba(var(--ink-rgb),.62)' }}>${e.summary}</span>
          </div>
        `;
      })}
    </div>
  `;
}

// -- app -----------------------------------------------------------------

function App() {
  const { health, reachable } = useHealth();
  const { events, loading, error } = useEvents();
  const [picked, setPicked] = useState(null);
  const [cited, setCited] = useState([]);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(4);
  const citedIds = useMemo(() => new Set(cited), [cited]);
  // The film strip runs forwards in time; /api/events hands us the reverse.
  const ordered = useMemo(() => events.slice().reverse(), [events]);

  // No pick means live. Picking pins a past moment and stays there -- being
  // yanked back to now mid-read because a frame arrived would be unusable, so
  // returning is always an explicit act.
  const live = picked === null;
  const current = picked;
  const onCitations = useCallback((ids) => setCited(ids), []);
  // Every pick that comes from a human (ledger row, timeline drag, a citation)
  // stops playback: it is a request to look at one moment, not to run from it.
  const pick = useCallback((e) => { setPlaying(false); setPicked(e); }, []);
  const toLive = useCallback(() => { setPlaying(false); setPicked(null); }, []);

  const seek = useCallback((i) => {
    if (!ordered.length) return;
    setPicked(ordered[Math.max(0, Math.min(ordered.length - 1, i))]);
  }, [ordered]);

  const toggle = useCallback(() => {
    if (playing) { setPlaying(false); return; }
    if (!ordered.length) return;
    // Play from wherever we are, except at the head (or live), where the only
    // sensible reading of "play" is to run the day again from the start.
    const i = picked ? ordered.findIndex((e) => e.id === picked.id) : -1;
    if (i < 0 || i >= ordered.length - 1) setPicked(ordered[0]);
    setPlaying(true);
  }, [playing, picked, ordered]);

  // One timeout per frame rather than one interval for the run: it re-arms off
  // the frame actually on screen, so scrubbing mid-playback just continues from
  // the new position, and reaching the end stops instead of wrapping.
  useEffect(() => {
    if (!playing) return undefined;
    const i = picked ? ordered.findIndex((e) => e.id === picked.id) : -1;
    if (i < 0) return undefined;
    if (i >= ordered.length - 1) { setPlaying(false); return undefined; }
    const id = setTimeout(() => setPicked(ordered[i + 1]), 1000 / rate);
    return () => clearTimeout(id);
  }, [playing, picked, ordered, rate]);

  // Decoding a thumb at 16 fps is only smooth if the bytes are already there.
  // Frames are immutable and served immutable, so the browser cache keeps them.
  useEffect(() => {
    if (!playing || !picked) return;
    const i = ordered.findIndex((e) => e.id === picked.id);
    for (let k = 1; k <= 5; k += 1) {
      const url = ordered[i + k]?.thumb_url;
      if (url) { const im = new Image(); im.src = url; }
    }
  }, [playing, picked, ordered]);

  // Transport keys, as in any video player. Ignored while typing a question.
  useEffect(() => {
    const onKey = (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      const i = picked ? ordered.findIndex((ev) => ev.id === picked.id) : ordered.length - 1;
      if (e.key === ' ') { e.preventDefault(); toggle(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); setPlaying(false); seek(i - 1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); setPlaying(false); seek(i + 1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggle, seek, picked, ordered]);

  // The caption under a live frame describes the newest event, which is the
  // most recent thing the VLM has actually said about the room.
  const caption = live ? events[0] : picked;

  return html`
    <div style=${{ minHeight: '100vh', padding: 40 }}>
      <div style=${{ maxWidth: 1180, margin: '0 auto', background: 'var(--panel)', border: '1px solid var(--line)', padding: '34px 36px 30px', display: 'flex', flexDirection: 'column', gap: 24 }}>

        <div style=${{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 24, flexWrap: 'wrap', borderBottom: '1px solid var(--line-soft)', paddingBottom: 18 }}>
          <div style=${{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
            <div style=${{ fontSize: 20, color: 'var(--ink)' }}>記憶帳本</div>
            <div style=${{ fontFamily: MONO, fontSize: 11, letterSpacing: '.2em', color: DIM }}>
              24H · CAM 01 · ${health?.device?.toUpperCase() || 'JETSON ORIN'}
            </div>
          </div>
          <${StatusBar} health=${health} reachable=${reachable} />
        </div>

        <div style=${{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 360px', gap: 34, alignItems: 'start' }}>
          <div style=${{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <${Viewer} event=${current} live=${live} playing=${playing}
              cited=${current ? citedIds.has(current.id) : false} />
            ${ordered.length > 0 && html`
              <${Transport} events=${events} ordered=${ordered} current=${current}
                citedIds=${citedIds} live=${live} playing=${playing} rate=${rate}
                onSeek=${seek} onToggle=${toggle} onRate=${setRate} onLive=${toLive} />`}
            ${/* Keyed per event so each pick re-runs the entrance -- except during
                 playback, where re-running it 16 times a second is a strobe. */
              caption && html`
              <div key=${playing ? 'play' : caption.id} style=${{ display: 'flex', gap: 22, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <div style=${{ fontFamily: MONO, fontSize: 38, fontWeight: 300, color: ACCENT, lineHeight: 1, animation: playing ? 'none' : 'tick .32s ease-out both' }}>${hm(caption.ts)}</div>
                <div style=${{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 2, flex: 1, minWidth: 260 }}>
                  <div style=${{ fontSize: 21, lineHeight: 1.5, color: 'var(--ink)', fontWeight: 300, animation: playing ? 'none' : 'rise .4s cubic-bezier(.2,.7,.2,1) both' }}>${caption.summary}</div>
                  <div style=${{ fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.1em' }}>
                    ${live ? '最近一次描述 · ' : ''}${hms(caption.ts)} · 信心 ${caption.confidence?.toFixed(2)} · ${caption.source}
                  </div>
                  ${caption.objects?.length ? html`
                    <div style=${{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      ${caption.objects.map((o) => html`
                        <span key=${o} style=${{ fontFamily: MONO, fontSize: 10.5, color: DIM, border: '1px solid var(--line)', padding: '2px 7px', letterSpacing: '.06em' }}>${o}</span>
                      `)}
                    </div>` : null}
                </div>
              </div>
            `}
          </div>

          <div style=${{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <${AskPanel} onCitations=${onCitations} onPick=${pick} events=${events} />
            ${loading
              ? html`<div style=${{ fontFamily: MONO, fontSize: 12, color: DIM, padding: '14px 4px' }}>載入中…</div>`
              : error
                ? html`<div style=${{ fontFamily: MONO, fontSize: 12, color: '#c98b6a', padding: '14px 4px' }}>${error}</div>`
                : html`<${Ledger} events=${events} current=${current} citedIds=${citedIds} onPick=${pick} />`}
          </div>
        </div>
      </div>
    </div>
  `;
}

ReactDOM.createRoot(document.getElementById('root')).render(html`<${App} />`);
