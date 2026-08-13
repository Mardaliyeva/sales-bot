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
  trace_version: 7,
  detail_level: "full",
  request_id: "22222222-2222-4222-8222-222222222222",
  session_id: "11111111-1111-4111-8111-111111111111",
  message_id: "33333333-3333-4333-8333-333333333333",
  status: "completed",
  model: { provider: "azure_openai", deployment: "gpt-test" },
  prompt: {
    mode: "modular",
    active_phase: "response",
    versions: { planner: "planner_v3", response: "response_v3" },
    phase_hashes: { tool: "tool-hash", response: "response-hash" },
    phase_chars: { tool: 4989, response: 1755 },
    phase_token_estimates: { tool: 1248, response: 439 },
    legacy_chars: 10685,
    tool_char_reduction_percent: 53.3,
    response_char_reduction_percent: 83.6,
  },
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
        ranking_mode: "active",
        ranking_objectives: [
          {
            field: "display_size_in",
            direction: "maximize",
            priority: "primary",
            origin: "explicit",
          },
        ],
        candidate_generation_lanes: [
          { lane: "semantic", requested_limit: 50, returned_count: 12 },
          { lane: "objective:display_size_in:maximize", requested_limit: 20, returned_count: 12 },
        ],
        ranking_components: {
          prd_laptops_001: { final_score: 0.94, objective_score: 1 },
        },
        sorted_candidates: candidates,
        hydrated_product_ids: candidates.slice(0, 5).map((item) => item.product_id),
        returned_product_ids: candidates.slice(0, 5).map((item) => item.product_id),
        total: 12,
      },
    },
  ],
  warnings: [],
  metrics: { model_rounds: 2, tool_count: 1 },
  continuation_context_before: "",
  continuation_context_after:
    "İstifadəçinin aktiv məhsul məqsədi discover əməliyyatıdır. Kataloqda şərtlərə uyğun məhsullar tapıldı.",
  memory_transition: {
    revision_before: 0,
    revision_after: 1,
    action: "replace",
    changed_ids: ["mem_entity_1"],
    removed_ids: [],
    size_bytes: 640,
    context_source: "confirmed",
    summary_replaced: true,
    summary_size_chars: 112,
    cache_expires_at: "2026-08-12T10:00:00+04:00",
  },
  decision_explanation: {
    version: 2,
    basis: "product_search",
    narrative: "İstifadəçi uyğun laptop istəyir. Kataloq axtarışı tamamlandı.",
    summary: "Kataloq filterləri və retrieval sıralaması ilə uyğun məhsullar seçildi.",
    understood_request: { operation: "discover", hard_constraints: [{ field: "category_id" }] },
    context_used: [{ source: "current_message", detail: "Cari istifadəçi mesajı istifadə edildi." }],
    decision_path: [{ code: "product_decision", status: "matching_products", detail: "Uyğun məhsullar seçildi." }],
    evidence: { product_ids: ["prd_laptops_001"] },
    outcome: { match_status: "matching_products" },
    memory_effect: {
      revision_before: 0,
      revision_after: 1,
      action: "replace",
      changed_ids: ["mem_entity_1"],
      removed_ids: [],
      size_bytes: 640,
    },
    limitations: [],
  },
};

describe("DebugDrawer", () => {
  it("shows retrieval sources and expands candidate lists to twenty", () => {
    render(<DebugDrawer open trace={trace} loading={false} error={null} onClose={vi.fn()} />);

    expect(screen.getByText("Tapılıb: 5")).toBeTruthy();
    expect(screen.getByText("Qdrant semantic")).toBeTruthy();
    expect(screen.getByText("Cavabın əsaslandırması")).toBeTruthy();
    expect(screen.getAllByText(/gizli düşüncəsi deyil/)).toHaveLength(2);
    expect(screen.getByText("product_search")).toBeTruthy();
    expect(screen.getByText(/İstifadəçi uyğun laptop istəyir/)).toBeTruthy();
    expect(screen.getByText("Sonrakı mesaj üçün söhbət konteksti")).toBeTruthy();
    expect(screen.getByText(/aktiv məhsul məqsədi discover əməliyyatıdır/)).toBeTruthy();
    expect(screen.getByText("confirmed")).toBeTruthy();
    expect(screen.getByText("Prompt modulları")).toBeTruthy();
    expect(screen.getByText("modular")).toBeTruthy();
    expect(screen.getAllByText("active")).toHaveLength(2);
    expect(screen.getByText(/Keyfiyy/)).toBeTruthy();
    expect(screen.getByText("53.3%")).toBeTruthy();
    expect(screen.getByText("Texniki detallar")).toBeTruthy();
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

  it("shows document candidates and selected chunk metadata", () => {
    const documentTrace: DebugTraceResponse = {
      ...trace,
      data_sources: {
        ...trace.data_sources,
        documents: { configured: true, document_count: 5, chunk_count: 18 },
      },
      timeline: [
        {
          stage: "document_search",
          status: "completed",
          result: { status: "success", total: 1 },
          retrieval: {
            mode: "document_qdrant_v1",
            query: "çatdırılma ödənişi",
            qdrant_checked: true,
            semantic_state: "active",
            min_score: 0.61,
            source_checksum: "abc",
            candidate_count: 1,
            selected_chunk_ids: ["delivery_policy:0001"],
            candidates: [
              {
                chunk_id: "delivery_policy:0001",
                document_id: "delivery_policy",
                filename: "delivery_policy.md",
                title: "Çatdırılma qaydaları",
                heading: "Çatdırılma qaydaları > Bakı",
                score: 0.82,
                selected: true,
                text_preview: "Çatdırılma bir iş günü ərzində edilir.",
              },
            ],
          },
        },
      ],
    };
    render(
      <DebugDrawer open trace={documentTrace} loading={false} error={null} onClose={vi.fn()} />,
    );

    expect(screen.getByText("Document retrieval nəticələri")).toBeTruthy();
    expect(screen.getByText("delivery_policy.md")).toBeTruthy();
    expect(screen.getByText(/modelə seçilib/)).toBeTruthy();
    expect(screen.getByText("5 sənəd")).toBeTruthy();
    expect(screen.getByText("18 chunk")).toBeTruthy();
  });
});
