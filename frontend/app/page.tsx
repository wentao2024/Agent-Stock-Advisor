"use client";
import { useState, useCallback, useRef } from "react";
import {
  analyzeStream, downloadReport, resolveCompany, indexCompany,
  type StreamEvent, type Recommendation,
} from "@/lib/api";

// ── Icons (inline SVG) ────────────────────────────────────────

const IconChart = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
  </svg>
);
const IconSearch = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
);
const IconLoader = ({ cls = "" }) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={cls}>
    <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
  </svg>
);
const IconDownload = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
);
const IconDatabase = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
  </svg>
);
const IconTrendUp = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
  </svg>
);
const IconTrendDown = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>
  </svg>
);
const IconMinus = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);
const IconShield = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const IconZap = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
);
const IconGitBranch = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>
    <path d="M18 9a9 9 0 0 1-9 9"/>
  </svg>
);
const IconCheck = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);
const IconAlert = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);
const IconArrow = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
  </svg>
);
const IconActivity = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);
const IconUsers = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
    <circle cx="9" cy="7" r="4"/>
    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
  </svg>
);

// ── Config ───────────────────────────────────────────────────

const DEMOS = ["Apple", "Microsoft", "NVIDIA", "Tesla", "Meta", "Amazon", "Google", "Netflix"];

type RecType = "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL";

const REC: Record<RecType, { color: string; bg: string; border: string; label: string; grad: string }> = {
  STRONG_BUY:  { color: "#10b981", bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.25)", label: "Strong Buy",  grad: "linear-gradient(135deg,#10b981,#059669)" },
  BUY:         { color: "#34d399", bg: "rgba(52,211,153,0.08)", border: "rgba(52,211,153,0.2)",  label: "Buy",         grad: "linear-gradient(135deg,#34d399,#10b981)" },
  HOLD:        { color: "#f59e0b", bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.25)", label: "Hold",        grad: "linear-gradient(135deg,#f59e0b,#d97706)" },
  SELL:        { color: "#f87171", bg: "rgba(248,113,113,0.08)",border: "rgba(248,113,113,0.2)", label: "Sell",        grad: "linear-gradient(135deg,#f87171,#ef4444)" },
  STRONG_SELL: { color: "#ef4444", bg: "rgba(239,68,68,0.08)",  border: "rgba(239,68,68,0.25)", label: "Strong Sell", grad: "linear-gradient(135deg,#ef4444,#dc2626)" },
};

const NODE_ICON: Record<string, React.ReactNode> = {
  planner:           <IconZap />,
  data_fetcher:      <IconChart />,
  technical_analyst: <IconActivity />,
  peer_benchmarker:  <IconUsers />,
  analyst:           <IconGitBranch />,
  synthesizer:       <IconCheck />,
  reflection:        <IconAlert />,
};

// ── Helpers ──────────────────────────────────────────────────

const pct = (v?: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : "—";
const usd = (v?: number | null) => v != null ? `$${v.toFixed(2)}` : "—";
const num = (v?: number | null, d = 1) => v != null ? v.toFixed(d) : "—";
const isPos = (v?: number | null) => v != null && v > 0;

type Notif = { type: "success" | "error"; message: string } | null;

// ── Main Component ───────────────────────────────────────────

export default function App() {
  const [input, setInput]         = useState("");
  const [ticker, setTicker]       = useState("");
  const [horizon, setHorizon]     = useState<"short" | "medium" | "long">("medium");
  const [loading, setLoading]     = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [indexing, setIndexing]   = useState(false);
  const [events, setEvents]       = useState<StreamEvent[]>([]);
  const [rec, setRec]             = useState<Recommendation | null>(null);
  const [tab, setTab]             = useState<"chain" | "metrics" | "risks">("chain");
  const [notif, setNotif]         = useState<Notif>(null);
  const [tokens, setTokens]       = useState({ input: 0, output: 0, total: 0 });
  const evRef                     = useRef<HTMLDivElement>(null);
  const abortRef                  = useRef<AbortController | null>(null);

  const showNotif = (type: "success" | "error", message: string) => {
    setNotif({ type, message });
    setTimeout(() => setNotif(null), 4000);
  };

  const run = useCallback(async (name?: string) => {
    const n = (name || input).trim();
    if (!n) return;

    // Cancel any in-progress request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true); setEvents([]); setRec(null); setTicker(""); setTokens({ input: 0, output: 0, total: 0 });
    try {
      for await (const ev of analyzeStream(n, true, horizon, controller.signal)) {
        if (controller.signal.aborted) break;
        if (ev.event_type === "final_recommendation" && ev.data) {
          setRec(ev.data as Recommendation);
          setTicker(ev.data.ticker || "");
        } else if (ev.event_type === "token_usage" && ev.data) {
          setTokens(p => ({
            input:  p.input  + (ev.data?.input_tokens  || 0),
            output: p.output + (ev.data?.output_tokens || 0),
            total:  p.total  + (ev.data?.total_tokens  || 0),
          }));
          setEvents(p => [...p, ev]);
        } else {
          setEvents(p => {
            const next = [...p, ev];
            setTimeout(() => evRef.current?.scrollTo({ top: 9999, behavior: "smooth" }), 50);
            return next;
          });
        }
      }
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setEvents(p => [...p, { event_type: "error", node: "client", message: e.message }]);
    } finally {
      setLoading(false);
    }
  }, [input, horizon]);

  const handleBlur = async (v: string) => {
    if (!v.trim()) return;
    const r = await resolveCompany(v);
    if (r?.ticker) setTicker(r.ticker);
  };

  const doDownload = async () => {
    if (!rec || downloading) return;
    setDownloading(true);
    try {
      await downloadReport(input.trim() || rec.company_name);
    } catch (e: any) {
      showNotif("error", `Download failed: ${e.message}`);
    } finally {
      setDownloading(false);
    }
  };

  const doIndex = async () => {
    const n = input.trim();
    if (!n || indexing) return;
    setIndexing(true);
    try {
      const r = await indexCompany(n);
      showNotif("success", `Index complete: ${r.ticker} — wrote ${r.chunks} chunks (${r.chroma_total} total)`);
    } catch (e: any) {
      showNotif("error", `Index failed: ${e.message}`);
    } finally {
      setIndexing(false);
    }
  };

  const cfg = rec ? REC[rec.recommendation] : null;

  // ── Metrics Data ──────────────────────────────────────────

  const mData = rec ? [
    { label: "Current Price",   val: usd(rec.metrics.current_price) },
    { label: "52W High",        val: usd(rec.metrics.price_52w_high) },
    { label: "52W Low",         val: usd(rec.metrics.price_52w_low) },
    { label: "Market Cap",      val: rec.metrics.market_cap_b ? `$${rec.metrics.market_cap_b.toFixed(1)}B` : "—" },
    { label: "P/E (TTM)",       val: num(rec.metrics.pe_ratio) },
    { label: "Forward P/E",     val: num(rec.metrics.forward_pe) },
    { label: "Revenue Growth",  val: pct(rec.metrics.revenue_growth_yoy), color: isPos(rec.metrics.revenue_growth_yoy) ? "#10b981" : "#ef4444", sub: "YoY" },
    { label: "Gross Margin",    val: pct(rec.metrics.gross_margin) },
    { label: "Net Margin",      val: pct(rec.metrics.net_margin) },
    { label: "ROE",             val: pct(rec.metrics.roe) },
    { label: "Debt/Equity",     val: num(rec.metrics.debt_to_equity) },
    { label: "Free Cash Flow",  val: rec.metrics.free_cash_flow_b ? `$${rec.metrics.free_cash_flow_b.toFixed(1)}B` : "—" },
  ] : [];

  return (
    <div className="app">
      {/* ── Notification Banner ── */}
      {notif && (
        <div className={`notif-banner notif-${notif.type}`}>
          {notif.type === "success" ? <IconCheck /> : <IconAlert />}
          <span>{notif.message}</span>
        </div>
      )}

      {/* ── Header ── */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon">📈</div>
          <span className="brand-name">StockAdvisor</span>
          <span className="brand-sub">AI · LangGraph</span>
        </div>

        <div className="search-wrap">
          <div className="search-box">
            <span className="search-icon"><IconSearch /></span>
            <input
              className="search-input"
              placeholder="Enter company name, e.g. Apple / Microsoft / NVIDIA"
              value={input}
              onChange={e => setInput(e.target.value)}
              onBlur={e => handleBlur(e.target.value)}
              onKeyDown={e => e.key === "Enter" && run()}
              style={{ paddingRight: ticker ? "80px" : "14px" }}
            />
            {ticker && <span className="ticker-pill">{ticker}</span>}
          </div>

          {/* Investment horizon selector */}
          <select
            className="horizon-select"
            value={horizon}
            onChange={e => setHorizon(e.target.value as "short" | "medium" | "long")}
            title="Investment Horizon"
          >
            <option value="short">Short</option>
            <option value="medium">Medium</option>
            <option value="long">Long</option>
          </select>

          <button className="btn-primary" onClick={() => run()} disabled={!input.trim()}>
            {loading ? <IconLoader cls="spin" /> : <IconSearch />}
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          <button className="btn-ghost" onClick={doIndex} disabled={indexing || !input.trim()} title="Pre-index data to ChromaDB">
            {indexing ? <IconLoader cls="spin" /> : <IconDatabase />}
            {indexing ? "Indexing..." : "Index"}
          </button>
        </div>

        <div className="demos">
          {DEMOS.map(d => (
            <button key={d} className="chip" onClick={() => { setInput(d); run(d); }}>{d}</button>
          ))}
        </div>
      </header>

      <div className="content">
        {/* ── Left: Agent Log ── */}
        <aside className="agent-panel">
          <div className="panel-header">
            <span className={`panel-dot ${loading ? "" : "idle"}`} />
            Agent Reasoning Log
          </div>
          <div className="log-scroll" ref={evRef}>
            {events.length === 0 && !loading && (
              <div className="log-empty">
                <div className="log-empty-icon">🤖</div>
                <div>Enter a company name to<br />see AI reasoning in real time</div>
              </div>
            )}
            {events.map((ev, i) => {
              const cls = ev.event_type === "reasoning_step" ? "reasoning"
                : ev.event_type === "error" ? "err"
                : ev.event_type === "done" ? "done-ev" : "";
              return (
                <div key={i} className={`log-item ${cls} fade-in`}>
                  <div className="log-icon">{NODE_ICON[ev.node] || <IconChart />}</div>
                  <div className="log-body">
                    <div className="log-node">{ev.node}</div>
                    <div className="log-msg">{ev.message}</div>
                    {ev.event_type === "reasoning_step" && ev.data?.finding && (
                      <div className="log-finding">{String(ev.data.finding).slice(0, 120)}</div>
                    )}
                  </div>
                </div>
              );
            })}
            {loading && (
              <div className="log-loading">
                <IconLoader cls="spin" />
                <span className="pulse">Agent thinking...</span>
              </div>
            )}
          </div>

          {/* Token counter */}
          {tokens.total > 0 && (
            <div className="token-bar">
              <span className="token-label">Tokens</span>
              <span className="token-val">{tokens.total.toLocaleString()}</span>
              <span className="token-detail">
                ↑{tokens.input.toLocaleString()} ↓{tokens.output.toLocaleString()}
              </span>
              <span className="token-cost">
                ~${((tokens.input * 2.5 + tokens.output * 10) / 1_000_000).toFixed(4)}
              </span>
            </div>
          )}
        </aside>

        {/* ── Right: Report ── */}
        <main className="report-panel">
          {!rec ? (
            <div className="empty-state">
              <div className="empty-hero">📊</div>
              <h1 className="empty-title">AI Stock Analysis Assistant</h1>
              <p className="empty-sub">
                Enter any company name. The system automatically invokes LangGraph multi-agent,
                completes 5-step multi-hop reasoning, and generates a structured investment recommendation report.
              </p>
              <div className="empty-tags">
                {["LangGraph", "Technical Analysis", "Peer Benchmarking", "Multi-hop Reasoning", "ChromaDB RAG"].map(t => (
                  <span key={t} className="empty-tag">{t}</span>
                ))}
              </div>
            </div>
          ) : cfg && (
            <div className="report-content fade-in">
              {/* Verdict Card */}
              <div className="verdict-card" style={{
                background: cfg.bg,
                borderColor: cfg.border,
                "--accent-grad": cfg.grad,
              } as any}>
                <div className="verdict-top">
                  <div className="verdict-company">
                    <div className="verdict-ticker" style={{ color: cfg.color }}>{rec.ticker}</div>
                    <div className="verdict-name">{rec.company_name}</div>
                  </div>
                  <div className="verdict-badge" style={{ color: cfg.color, background: "rgba(0,0,0,0.2)", borderColor: cfg.border }}>
                    {rec.recommendation === "STRONG_BUY" || rec.recommendation === "BUY"
                      ? <IconTrendUp />
                      : rec.recommendation === "STRONG_SELL" || rec.recommendation === "SELL"
                      ? <IconTrendDown />
                      : <IconMinus />}
                    {cfg.label}
                  </div>
                </div>

                <div className="confidence-row">
                  <span>Confidence</span>
                  <div className="conf-track">
                    <div className="conf-fill" style={{ width: `${rec.confidence * 100}%`, background: cfg.grad }} />
                  </div>
                  <span className="conf-pct" style={{ color: cfg.color }}>{Math.round(rec.confidence * 100)}%</span>
                </div>

                <div className="thesis" style={{ borderLeftColor: cfg.color }}>
                  {rec.one_line_thesis}
                </div>

                <div className="price-row">
                  <div className="price-item">
                    <span className="price-label">Current Price</span>
                    <span className="price-value">${rec.current_price.toFixed(2)}</span>
                  </div>
                  {rec.target_price_high && (
                    <div className="price-item">
                      <span className="price-label">Target Range</span>
                      <span className="price-value" style={{ color: cfg.color }}>
                        ${rec.target_price_low?.toFixed(0)} – ${rec.target_price_high?.toFixed(0)}
                      </span>
                    </div>
                  )}
                  {rec.upside_pct != null && (
                    <div className="price-item">
                      <span className="price-label">Upside</span>
                      <span className="price-value" style={{ color: rec.upside_pct >= 0 ? "#10b981" : "#ef4444" }}>
                        {rec.upside_pct >= 0 ? "+" : ""}{rec.upside_pct.toFixed(1)}%
                      </span>
                    </div>
                  )}
                  <div className="price-actions">
                    <button className="btn-ghost" onClick={doDownload} disabled={downloading}>
                      {downloading ? <IconLoader cls="spin" /> : <IconDownload />}
                      {downloading ? "Generating..." : "Download Report"}
                    </button>
                  </div>
                </div>
              </div>

              {/* Tabs */}
              <div className="tabs">
                <button className={`tab ${tab === "chain" ? "active" : ""}`} onClick={() => setTab("chain")}>Reasoning Chain</button>
                <button className={`tab ${tab === "metrics" ? "active" : ""}`} onClick={() => setTab("metrics")}>Financials</button>
                <button className={`tab ${tab === "risks" ? "active" : ""}`} onClick={() => setTab("risks")}>Bull / Bear</button>
              </div>

              {/* Reasoning Chain */}
              {tab === "chain" && (
                <div className="chain">
                  {rec.reasoning_chain.map((step, i) => (
                    <div key={i} className="chain-step">
                      <div className="chain-num">Step {step.step}</div>
                      <div className="chain-title">{step.title}</div>
                      <div className="chain-finding">{step.finding}</div>
                      {step.implication && (
                        <div className="chain-impl">
                          <IconArrow />
                          <span style={{ color: "var(--text2)", fontSize: "12px" }}>{step.implication}</span>
                        </div>
                      )}
                    </div>
                  ))}
                  {rec.sources?.length > 0 && (
                    <div style={{ fontSize: "11px", color: "var(--text3)", paddingTop: "8px" }}>
                      Data Sources: {rec.sources.map(s => s.name).join(" · ")}
                    </div>
                  )}
                </div>
              )}

              {/* Metrics */}
              {tab === "metrics" && (
                <div className="metrics">
                  {mData.map(({ label, val, sub, color }: any) => (
                    <div key={label} className="metric-card">
                      <div className="metric-label">{label}</div>
                      <div className="metric-val" style={color ? { color } : {}}>{val}</div>
                      {sub && <div className="metric-sub">{sub}</div>}
                    </div>
                  ))}
                </div>
              )}

              {/* Risks */}
              {tab === "risks" && (
                <>
                  <div className="bull-bear">
                    <div className="side bull">
                      <div className="side-title"><IconTrendUp />Bull Case</div>
                      {rec.bull_case.map((b, i) => (
                        <div key={i} className="case-item">
                          <div className="case-dot" />
                          {b}
                        </div>
                      ))}
                    </div>
                    <div className="side bear">
                      <div className="side-title"><IconTrendDown />Bear Case</div>
                      {rec.bear_case.map((b, i) => (
                        <div key={i} className="case-item">
                          <div className="case-dot" />
                          {b}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="risks-card">
                    <div className="risks-title"><IconShield />Key Risks</div>
                    {rec.key_risks.map((r, i) => (
                      <div key={i} className="risk-item">
                        <span className="risk-icon"><IconAlert /></span>
                        {r}
                      </div>
                    ))}
                  </div>
                  {rec.catalysts?.length > 0 && (
                    <div className="risks-card">
                      <div className="risks-title"><IconZap />Near-term Catalysts</div>
                      {rec.catalysts.map((c, i) => (
                        <div key={i} className="risk-item">
                          <span className="catalyst-icon"><IconCheck /></span>
                          {c}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
