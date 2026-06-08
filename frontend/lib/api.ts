const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface StreamEvent {
  event_type: string;
  node: string;
  message: string;
  data?: Record<string, any>;
}

export interface Metrics {
  current_price: number;
  price_52w_high?: number;
  price_52w_low?: number;
  price_change_1y_pct?: number;
  pe_ratio?: number;
  forward_pe?: number;
  pb_ratio?: number;
  market_cap_b?: number;
  revenue_growth_yoy?: number;
  gross_margin?: number;
  operating_margin?: number;
  net_margin?: number;
  debt_to_equity?: number;
  free_cash_flow_b?: number;
  revenue_ttm_b?: number;
  roe?: number;
}

export interface ReasoningStep {
  step: number;
  title: string;
  finding: string;
  implication?: string;
}

export interface Recommendation {
  ticker: string;
  company_name: string;
  recommendation: "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL";
  confidence: number;
  one_line_thesis: string;
  current_price: number;
  target_price_low?: number;
  target_price_high?: number;
  upside_pct?: number;
  bull_case: string[];
  bear_case: string[];
  key_risks: string[];
  catalysts: string[];
  metrics: Metrics;
  reasoning_chain: ReasoningStep[];
  sources: Array<{ source_type: string; name: string; relevance: string }>;
}

export async function* analyzeStream(
  companyName: string,
  includeRag = true,
  horizon = "medium",
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API}/api/analyze/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ company_name: companyName, include_rag: includeRag, horizon }),
    signal,
  });
  if (!res.ok) throw new Error("Analysis request failed");
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try { yield JSON.parse(line.slice(6)); } catch {}
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

export async function downloadReport(companyName: string): Promise<void> {
  const res = await fetch(`${API}/api/analyze/download`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ company_name: companyName, include_rag: true }),
  });
  if (!res.ok) throw new Error("Download failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const disp = res.headers.get("Content-Disposition") || "";
  const match = disp.match(/filename="([^"]+)"/);
  a.download = match ? match[1] : `analysis_${companyName}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function resolveCompany(name: string) {
  try {
    const res = await fetch(`${API}/api/resolve?name=${encodeURIComponent(name)}`);
    if (!res.ok) return null;
    return res.json() as Promise<{ input: string; ticker: string }>;
  } catch { return null; }
}

export async function indexCompany(name: string) {
  const res = await fetch(`${API}/api/index/${encodeURIComponent(name)}`, { method: "POST" });
  if (!res.ok) throw new Error("Index failed");
  return res.json();
}
