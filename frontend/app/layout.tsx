import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "StockAdvisor AI — Agentic Stock Analysis",
  description: "LangGraph multi-agent stock analysis with multi-hop reasoning and ChromaDB RAG",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
