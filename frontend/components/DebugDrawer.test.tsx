import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DebugDrawer } from "@/components/DebugDrawer";
import type { DebugTraceResponse } from "@/lib/types";

const candidates = Array.from({ length: 12 }, (_, index) => ({
  product_id: `prd_laptops_${String(index + 1).padStart(3, "0")}`,
  name: `Laptop ${index + 1}`,
  rank: index + 1,
  score: 1 - index / 100,
}));

const trace: DebugTraceResponse = {
  trace_version: 2,
  detail_level: "full",
  request_id: "22222222-2222-4222-8222-222222222222",
  session_id: "11111111-1111-4111-8111-111111111111",
  message_id: "33333333-3333-4333-8333-333333333333",
  status: "completed",
  model: { provider: "azure_openai", deployment: "gpt-test" },
  diagnosis: {
    code: "products_found",
    title: "Uyğun məhsullar tapıldı",
    detail: "Axtarış tamamlandı.",
    catalog_checked: true,
    data_status: "Tapılıb: 5",
    result_count: 5,
  },
  data_sources: {
    product_catalog_json: { configured: true, product_count: 300 },
    semantic_qdrant: { configured: true },
    documents: { configured: false },
  },
  timeline: [
    {
      stage: "product_search",
      status: "completed",
      retrieval: {
        mode: "qdrant_only_v2",
        query: "universitet üçün laptop",
        filters: { category_id: "laptops" },
        sort: "relevance",
        qdrant_checked: true,
        filtered_count: 50,
        exact_product_ids: [],
        exact_filter_conflict: false,
        exact_candidates: candidates,
        semantic_candidates: candidates,
        semantic_state: "active",
        sorted_candidates: candidates,
        hydrated_product_ids: candidates.slice(0, 5).map((item) => item.product_id),
        returned_product_ids: candidates.slice(0, 5).map((item) => item.product_id),
        total: 12,
      },
    },
  ],
  warnings: [],
  metrics: { model_rounds: 2, tool_count: 1 },
};

describe("DebugDrawer", () => {
  it("shows retrieval sources and expands candidate lists to twenty", () => {
    render(<DebugDrawer open trace={trace} loading={false} error={null} onClose={vi.fn()} />);

    expect(screen.getByText("Tapılıb: 5")).toBeTruthy();
    expect(screen.getByText("Qdrant semantic")).toBeTruthy();
    expect(screen.queryByText("Laptop 12")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "20 namizədə qədər göstər" }));
    expect(screen.getAllByText("Laptop 12")).toHaveLength(3);
  });

  it("closes from the drawer close action", () => {
    const onClose = vi.fn();
    render(<DebugDrawer open trace={trace} loading={false} error={null} onClose={onClose} />);

    fireEvent.click(screen.getAllByRole("button", { name: "Debug panelini bağla" })[1]);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("copies the complete debug trace as formatted JSON", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    render(<DebugDrawer open trace={trace} loading={false} error={null} onClose={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Debug-i kopyala" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(JSON.stringify(trace, null, 2)));
    expect(screen.getByRole("button", { name: "Debug kopyalandı" })).toBeTruthy();
  });
});
