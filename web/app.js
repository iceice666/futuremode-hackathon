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

const { useState, useEffect, useRef, useCallback, useMemo } = React;
const html = htm.bind(React.createElement);

const PAGE = 200;
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

/** The frame behind the selected event. Real thumbnails, with the reference's
 *  scanline/vignette/sweep treatment kept as an overlay. */
function Viewer({ event, cited }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => setBroken(false), [event?.id]);
  if (!event) {
    return html`<div style=${{ position: 'relative', aspectRatio: '16 / 9', background: '#0a0907', border: '1px solid var(--edge)', display: 'grid', placeItems: 'center', color: DIM, fontFamily: MONO, fontSize: 12 }}>等待畫面</div>`;
  }
  return html`
    <div style=${{ position: 'relative', aspectRatio: '16 / 9', background: '#0a0907', border: `1px solid ${cited ? 'rgba(var(--accent-rgb),.55)' : 'var(--edge)'}`, overflow: 'hidden' }}>
      ${broken
        ? html`<div style=${{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: DIM, fontFamily: MONO, fontSize: 12 }}>縮圖不存在</div>`
        : html`<img key=${event.id} src=${event.thumb_url} alt="" onError=${() => setBroken(true)}
             style=${{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', animation: 'rise .4s ease-out both' }} />`}
      <div style=${{ position: 'absolute', inset: 0, background: 'repeating-linear-gradient(rgba(0,0,0,.32) 0 1px, rgba(0,0,0,0) 1px 3px)', pointerEvents: 'none' }}></div>
      <div style=${{ position: 'absolute', inset: 0, boxShadow: 'inset 0 0 130px 34px rgba(0,0,0,.6)', pointerEvents: 'none' }}></div>
      <div style=${{ position: 'absolute', left: 0, right: 0, height: '16%', background: 'linear-gradient(rgba(var(--accent-rgb),.07), rgba(var(--accent-rgb),0))', animation: 'sweep 8s linear infinite', pointerEvents: 'none' }}></div>
      <div key=${'cut' + event.id} style=${{ position: 'absolute', inset: 0, background: 'rgba(var(--accent-rgb),.16)', mixBlendMode: 'screen', pointerEvents: 'none', animation: 'cut .42s ease-out forwards' }}></div>
      <div style=${{ position: 'absolute', top: 12, left: 14, fontFamily: MONO, fontSize: 11, letterSpacing: '.16em', color: 'rgba(232,216,190,.8)' }}>
        ${event.source === 'seed' ? 'SEED' : 'LIVE'} · ${event.id.slice(-8)}
      </div>
    </div>
  `;
}

/** 96 buckets of 15 minutes. Height is event density, brightness marks the
 *  cited answers, and dragging scrubs to the nearest event. */
function Timeline({ events, current, onPick, citedIds }) {
  const trackRef = useRef(null);
  const [dragging, setDragging] = useState(false);

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

  const nearest = useCallback((minute) => {
    let best = null, d = Infinity;
    for (const e of events) {
      const dd = Math.abs(minuteOfDay(e.ts) - minute);
      if (dd < d) { d = dd; best = e; }
    }
    return best;
  }, [events]);

  const move = useCallback((e) => {
    const r = trackRef.current?.getBoundingClientRect();
    if (!r) return;
    const p = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const hitEvent = nearest(p * 1440);
    if (hitEvent) onPick(hitEvent);
  }, [nearest, onPick]);

  const activeBucket = current ? Math.floor(minuteOfDay(current.ts) / 15) : -1;

  return html`
    <div style=${{ display: 'flex', flexDirection: 'column', gap: 7, borderTop: '1px solid var(--line-soft)', paddingTop: 10 }}>
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
        ${current && html`<span style=${{ position: 'absolute', top: 0, bottom: 0, left: `${minuteOfDay(current.ts) / 1440 * 100}%`, width: 1, background: 'rgba(var(--accent-rgb),.85)', pointerEvents: 'none', animation: 'glow 2.6s ease-in-out infinite', transition: 'left .12s cubic-bezier(.2,.8,.2,1)' }}></span>`}
      </div>
      <div style=${{ display: 'flex', justifyContent: 'space-between', fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.14em' }}>
        ${['00', '04', '08', '12', '16', '20', '24'].map((h) => html`<span key=${h}>${h}</span>`)}
      </div>
      <div style=${{ display: 'flex', justifyContent: 'space-between', fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.12em' }}>
        <span>拖曳時間軸以回放記憶</span><span>${events.length} 個事件</span>
      </div>
    </div>
  `;
}

/** The ask panel. The reference filtered client-side; the real system answers
 *  semantically over embeddings, so the box posts to /api/ask and the answer's
 *  citations then light up the ledger and the timeline. */
function AskPanel({ onCitations, onPick, events }) {
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true); setError(null);
    try {
      const d = await getJSON('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      setResult(d);
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

  return html`
    <form onSubmit=${submit} style=${{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style=${{ display: 'flex', alignItems: 'center', gap: 9, borderBottom: '1px solid var(--edge)', padding: '0 2px 10px' }}>
        <span style=${{ fontFamily: MONO, fontSize: 12, color: ACCENT }}>?</span>
        <input class="ask" value=${q} onInput=${(e) => setQ(e.target.value)}
          placeholder="問這個空間的記憶,例如「我的杯子在哪」"
          style=${{ flex: 1, minWidth: 0, background: 'transparent', border: 0, color: 'var(--ink)', fontFamily: MONO, fontSize: 12.5 }} />
        ${(result || error) && html`<span onClick=${clear} style=${{ fontFamily: MONO, fontSize: 11, color: DIM, cursor: 'pointer' }}>清除</span>`}
        ${busy && html`<span style=${{ width: 10, height: 10, border: '1px solid rgba(var(--accent-rgb),.3)', borderTopColor: ACCENT, borderRadius: '50%', animation: 'spin .7s linear infinite' }}></span>`}
      </div>
      ${error && html`<div style=${{ fontFamily: MONO, fontSize: 11.5, color: '#c98b6a', lineHeight: 1.6 }}>${error}</div>`}
      ${result && html`
        <div style=${{ display: 'flex', flexDirection: 'column', gap: 6, padding: '2px 2px 0', animation: 'rise .3s ease-out both' }}>
          <div style=${{ fontSize: 14, lineHeight: 1.65, color: 'var(--ink)' }}>${result.answer}</div>
          <div style=${{ fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.1em' }}>
            ${result.citations?.length ? `${result.citations.length} 筆佐證 · ` : '無佐證 · '}${result.latency_ms} ms
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
  const citedIds = useMemo(() => new Set(cited), [cited]);

  // Follow the live edge until the operator picks something, then stay put --
  // jumping away mid-read because a frame arrived would make it unusable.
  const current = picked || events[0] || null;
  const onCitations = useCallback((ids) => setCited(ids), []);

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
            <${Viewer} event=${current} cited=${current ? citedIds.has(current.id) : false} />
            ${current && html`
              <div key=${current.id} style=${{ display: 'flex', gap: 22, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <div style=${{ fontFamily: MONO, fontSize: 38, fontWeight: 300, color: ACCENT, lineHeight: 1, animation: 'tick .32s ease-out both' }}>${hm(current.ts)}</div>
                <div style=${{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 2, flex: 1, minWidth: 260 }}>
                  <div style=${{ fontSize: 21, lineHeight: 1.5, color: 'var(--ink)', fontWeight: 300, animation: 'rise .4s cubic-bezier(.2,.7,.2,1) both' }}>${current.summary}</div>
                  <div style=${{ fontFamily: MONO, fontSize: 11, color: DIM, letterSpacing: '.1em' }}>
                    ${hms(current.ts)} · 信心 ${current.confidence?.toFixed(2)} · ${current.source}
                  </div>
                  ${current.objects?.length ? html`
                    <div style=${{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      ${current.objects.map((o) => html`
                        <span key=${o} style=${{ fontFamily: MONO, fontSize: 10.5, color: DIM, border: '1px solid var(--line)', padding: '2px 7px', letterSpacing: '.06em' }}>${o}</span>
                      `)}
                    </div>` : null}
                </div>
              </div>
            `}
          </div>

          <div style=${{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <${AskPanel} onCitations=${onCitations} onPick=${setPicked} events=${events} />
            ${loading
              ? html`<div style=${{ fontFamily: MONO, fontSize: 12, color: DIM, padding: '14px 4px' }}>載入中…</div>`
              : error
                ? html`<div style=${{ fontFamily: MONO, fontSize: 12, color: '#c98b6a', padding: '14px 4px' }}>${error}</div>`
                : html`<${Ledger} events=${events} current=${current} citedIds=${citedIds} onPick=${setPicked} />`}
          </div>
        </div>

        <${Timeline} events=${events} current=${current} onPick=${setPicked} citedIds=${citedIds} />
      </div>
    </div>
  `;
}

ReactDOM.createRoot(document.getElementById('root')).render(html`<${App} />`);
