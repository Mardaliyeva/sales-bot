import type { ImgHTMLAttributes } from "react";

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ChatApp } from "@/components/ChatApp";
import { SalesBotApiError, salesBotApi } from "@/lib/api";
import type { ProductCardsPresentation } from "@/lib/types";

vi.mock("next/image", () => ({
  default: ({ priority, ...props }: ImgHTMLAttributes<HTMLImageElement> & { priority?: boolean }) => {
    void priority;
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img {...props} alt={props.alt || ""} />
    );
  },
}));

const sessionId = "11111111-1111-4111-8111-111111111111";

function productPresentation(): ProductCardsPresentation {
  return {
    type: "product_cards",
    title: "1200 AZN büdcəyə uyğun 5 məhsul tapdım",
    total: 5,
    shown_count: 3,
    recommended_product_id: "prd_televisions_008",
    items: [
      {
        product_id: "prd_televisions_008",
        name: "Samsung Q70D QLED 4K",
        sku: "SYN-TV-SMS-008",
        price: 569.99,
        currency: "AZN",
        stock_status: "in_stock",
        rating: 5,
        warranty_months: 36,
        highlights: ['55\" ekran QLED', "8K UHD", "120 Hz", "HDR yoxdur", "Tizen"],
        budget_remaining: 630.01,
      },
      {
        product_id: "prd_televisions_001",
        name: "Samsung QN900D Neo QLED 8K",
        sku: "SYN-TV-SMS-001",
        price: 1186.79,
        currency: "AZN",
        stock_status: "in_stock",
        rating: 4.1,
        warranty_months: 24,
        highlights: ['55\" ekran QLED', "4K UHD", "120 Hz", "HDR", "Tizen"],
        budget_remaining: 13.21,
      },
      {
        product_id: "prd_televisions_003",
        name: "LG OLED evo C4",
        sku: "SYN-TV-LGE-003",
        price: 1199,
        currency: "AZN",
        stock_status: "out_of_stock",
        rating: 4.6,
        warranty_months: 24,
        highlights: ['55\" ekran OLED', "4K UHD", "144 Hz", "HDR", "webOS"],
        budget_remaining: 1,
      },
    ],
  };
}

function alternativePresentation(): ProductCardsPresentation {
  const presentation = productPresentation();
  return {
    ...presentation,
    result_kind: "alternatives",
    requested_label: "Samsung Future TV",
    title: "Samsung Future TV tapılmadı — yaxın alternativlər",
    relaxed_fields: ["color_code"],
    items: presentation.items.slice(0, 2).map((item) => ({
      ...item,
      differences: ["Rəng fərqlidir: Qara"],
    })),
    total: 2,
    shown_count: 2,
  };
}

function mockSuccessfulApi() {
  vi.spyOn(salesBotApi, "createSession").mockResolvedValue({
    session_id: sessionId,
    status: "active",
    expires_at: "2026-08-12T10:00:00Z",
  });
  vi.spyOn(salesBotApi, "sendMessage")
    .mockResolvedValueOnce({
      request_id: "22222222-2222-4222-8222-222222222222",
      session_id: sessionId,
      message_id: "33333333-3333-4333-8333-333333333333",
      answer:
        "Universitet üçün uyğun laptoplar tapdım. İlk seçim **büdcənizə uyğundur**.\n\nHDR sizin üçün vacibdir?",
      used_tools: ["product_search"],
      finish_reason: "completed",
      presentation: productPresentation(),
    })
    .mockResolvedValueOnce({
      request_id: "44444444-4444-4444-8444-444444444444",
      session_id: sessionId,
      message_id: "55555555-5555-4555-8555-555555555555",
      answer: "Ən münasib variant birinci məhsuldur.",
      used_tools: ["product_search"],
      finish_reason: "completed",
    });
}

describe("ChatApp", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("renders the sales-focused empty state and suggestions", () => {
    render(<ChatApp />);

    expect(screen.getByRole("heading", { name: "Bu gün sənə necə kömək edə bilərəm?" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Qara 128 GB iPhone göstər" })).toBeTruthy();
    expect(screen.getByLabelText("Məhsul haqqında sual")).toBeTruthy();
    expect(screen.getByText("0/4000")).toBeTruthy();
  });

  it("creates one session and reuses it for following messages", async () => {
    mockSuccessfulApi();
    render(<ChatApp />);

    fireEvent.click(screen.getByRole("button", { name: "Universitet üçün laptop lazımdır" }));
    expect(await screen.findByText(/Universitet üçün uyğun laptoplar tapdım/)).toBeTruthy();

    const textarea = screen.getByLabelText("Məhsul haqqında sual");
    fireEvent.change(textarea, { target: { value: "Ən ucuzu hansıdır?" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(await screen.findByText("Ən münasib variant birinci məhsuldur.")).toBeTruthy();

    expect(salesBotApi.createSession).toHaveBeenCalledTimes(1);
    expect(salesBotApi.sendMessage).toHaveBeenNthCalledWith(1, sessionId, "Universitet üçün laptop lazımdır");
    expect(salesBotApi.sendMessage).toHaveBeenNthCalledWith(2, sessionId, "Ən ucuzu hansıdır?");
  });

  it("renders structured product cards without product images", async () => {
    mockSuccessfulApi();
    render(<ChatApp />);

    fireEvent.click(screen.getByRole("button", { name: "Universitet üçün laptop lazımdır" }));

    const results = await screen.findByRole("region", { name: "Məhsul seçimləri" });
    const answer = results.querySelector(".product-answer");
    const firstCard = within(results).getByText("Samsung Q70D QLED 4K").closest(".product-card");
    const followUp = results.querySelector(".product-follow-up");
    expect(answer?.textContent).toBe(
      "Universitet üçün uyğun laptoplar tapdım. İlk seçim büdcənizə uyğundur.",
    );
    expect(answer?.querySelector("strong")?.textContent).toBe("büdcənizə uyğundur");
    expect(followUp?.textContent).toBe("HDR sizin üçün vacibdir?");
    expect(answer?.compareDocumentPosition(firstCard as Node) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(firstCard?.compareDocumentPosition(followUp as Node) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(results).queryByText("Bəzi uyğun məhsullar tapıldı.")).toBeNull();
    expect(within(results).queryByText("Uyğun məhsul tapıldı.")).toBeNull();
    expect(within(results).queryByText("Tövsiyəm:")).toBeNull();
    expect(within(results).queryByText("1200 AZN büdcəyə uyğun 5 məhsul tapdım")).toBeNull();
    expect(within(results).queryByText("İlk 3 seçim göstərilir")).toBeNull();
    expect(within(results).queryByText("Ən uyğun seçim")).toBeNull();
    expect(firstCard).toBeTruthy();
    expect(within(results).getByText("Samsung QN900D Neo QLED 8K")).toBeTruthy();
    expect(within(results).getByText("LG OLED evo C4")).toBeTruthy();
    expect(within(results).getByText("569,99 AZN")).toBeTruthy();
    expect(within(results).getByText("Stokda yoxdur")).toBeTruthy();
    expect(within(results).getByText("630,01 AZN")).toBeTruthy();
    expect(within(results).getAllByText("reytinq")).toHaveLength(3);
    expect(within(results).getAllByText("zəmanət")).toHaveLength(3);
    expect(within(results).queryByRole("img")).toBeNull();
  });

  it("renders missing-product alternatives before the card details", async () => {
    vi.spyOn(salesBotApi, "createSession").mockResolvedValue({
      session_id: sessionId,
      status: "active",
      expires_at: "2026-08-12T10:00:00Z",
    });
    vi.spyOn(salesBotApi, "sendMessage").mockResolvedValue({
      request_id: "22222222-2222-4222-8222-222222222222",
      session_id: sessionId,
      message_id: "33333333-3333-4333-8333-333333333333",
      answer: "Samsung Future TV kataloqda tapılmadı. Ən yaxın seçim kimi Samsung Q70D QLED 4K göstərilir.",
      used_tools: ["product_search"],
      finish_reason: "completed",
      presentation: alternativePresentation(),
    });
    render(<ChatApp />);

    fireEvent.click(screen.getByRole("button", { name: "Universitet üçün laptop lazımdır" }));

    const results = await screen.findByRole("region", { name: "Yaxın məhsul alternativləri" });
    expect(within(results).queryByText("Samsung Future TV tapılmadı — yaxın alternativlər")).toBeNull();
    expect(within(results).getByText("Ən yaxın alternativ")).toBeTruthy();
    expect(within(results).getAllByText("Rəng fərqlidir: Qara")).toHaveLength(2);
    expect(within(results).getByText(/Samsung Future TV kataloqda tapılmadı/)).toBeTruthy();
    expect(results.querySelector(".product-follow-up")).toBeNull();
  });

  it("keeps Shift+Enter local and blocks oversized submission", () => {
    mockSuccessfulApi();
    render(<ChatApp />);
    const textarea = screen.getByLabelText("Məhsul haqqında sual");

    fireEvent.change(textarea, { target: { value: "Birinci sətir" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(salesBotApi.createSession).not.toHaveBeenCalled();

    fireEvent.change(textarea, { target: { value: "x".repeat(4001) } });
    fireEvent.submit(textarea.closest("form") as HTMLFormElement);
    expect(salesBotApi.createSession).not.toHaveBeenCalled();
  });

  it("blocks parallel submissions while a reply is pending", async () => {
    vi.spyOn(salesBotApi, "createSession").mockResolvedValue({
      session_id: sessionId,
      status: "active",
      expires_at: "2026-08-12T10:00:00Z",
    });
    const sendMessage = vi
      .spyOn(salesBotApi, "sendMessage")
      .mockReturnValue(new Promise(() => undefined));
    render(<ChatApp />);

    fireEvent.click(screen.getByRole("button", { name: "Universitet üçün laptop lazımdır" }));
    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));

    const pendingButton = screen.getByRole("button", { name: "Gözləyin" });
    expect(pendingButton).toHaveProperty("disabled", true);
    fireEvent.click(pendingButton);
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it("stores completed chats, starts a new chat and restores recents", async () => {
    mockSuccessfulApi();
    render(<ChatApp />);

    fireEvent.click(screen.getByRole("button", { name: "Universitet üçün laptop lazımdır" }));
    await screen.findByText(/Universitet üçün uyğun laptoplar tapdım/);
    fireEvent.click(screen.getByRole("button", { name: "Yeni söhbət" }));

    expect(screen.getByRole("heading", { name: "Bu gün sənə necə kömək edə bilərəm?" })).toBeTruthy();
    const recent = screen.getByTitle("Universitet üçün laptop lazımdır");
    fireEvent.click(recent);
    expect(await screen.findByText(/Universitet üçün uyğun laptoplar tapdım/)).toBeTruthy();
  });

  it("shows an explicit new-chat action for expired sessions", async () => {
    vi.spyOn(salesBotApi, "createSession").mockResolvedValue({
      session_id: sessionId,
      status: "active",
      expires_at: "2026-08-12T10:00:00Z",
    });
    vi.spyOn(salesBotApi, "sendMessage").mockRejectedValue(
      new SalesBotApiError(410, "session_expired", "Sessiyanın vaxtı bitib."),
    );
    render(<ChatApp />);

    const textarea = screen.getByLabelText("Məhsul haqqında sual");
    fireEvent.change(textarea, { target: { value: "Bu məhsul stokdadır?" } });
    fireEvent.click(screen.getByRole("button", { name: "Göndər" }));

    await waitFor(() => {
      expect(screen.getByText("Bu söhbətin vaxtı bitib. Davam etmək üçün yeni söhbət açın.")).toBeTruthy();
    });
    expect(screen.getByRole("button", { name: "Yeni söhbət aç" })).toBeTruthy();
  });

  it.each([
    [422, "validation_error", "Mesaj qəbul edilmədi. Mətni yoxlayıb yenidən göndərin."],
    [409, "request_conflict", "Bu söhbətdə başqa sorğu işlənir. Bir az sonra yenidən cəhd edin."],
    [503, "assistant_temporarily_unavailable", "Köməkçi hazırda əlçatan deyil. Bir qədər sonra yenidən cəhd edin."],
    [0, "network_error", "Serverlə əlaqə yaratmaq mümkün olmadı."],
  ])("shows the Azerbaijani error state for status %s", async (status, code, expectedMessage) => {
    vi.spyOn(salesBotApi, "createSession").mockResolvedValue({
      session_id: sessionId,
      status: "active",
      expires_at: "2026-08-12T10:00:00Z",
    });
    vi.spyOn(salesBotApi, "sendMessage").mockRejectedValue(
      new SalesBotApiError(status, code, expectedMessage),
    );
    render(<ChatApp />);

    fireEvent.click(screen.getByRole("button", { name: "Qara 128 GB iPhone göstər" }));

    expect(await screen.findByText(expectedMessage)).toBeTruthy();
  });

  it("opens the developer trace for the selected assistant answer", async () => {
    vi.stubEnv("NEXT_PUBLIC_DEBUG_PANEL", "true");
    mockSuccessfulApi();
    vi.spyOn(salesBotApi, "getDebugTrace").mockResolvedValue({
      trace_version: 1,
      detail_level: "full",
      request_id: "22222222-2222-4222-8222-222222222222",
      session_id: sessionId,
      message_id: "33333333-3333-4333-8333-333333333333",
      status: "completed",
      model: { provider: "azure_openai", deployment: "gpt-test" },
      diagnosis: {
        code: "products_found",
        title: "Uyğun məhsullar tapıldı",
        detail: "Axtarış tamamlandı.",
        catalog_checked: true,
        data_status: "Tapılıb: 3",
        result_count: 3,
      },
      data_sources: {
        product_catalog: { configured: true, product_count: 300 },
        semantic_qdrant: { configured: true },
        documents: { configured: false, detail: "Document məlumat mənbəyi yoxdur." },
      },
      timeline: [
        { stage: "input_validation", status: "completed" },
        {
          stage: "model_round",
          status: "completed",
          round: 1,
          tools_allowed: true,
          decision: "tool_call",
          tool_name: "product_search",
        },
      ],
      warnings: [],
      metrics: { model_rounds: 2, tool_count: 1, latency_ms: 1200 },
    });
    render(<ChatApp />);

    fireEvent.click(screen.getByRole("button", { name: "Universitet üçün laptop lazımdır" }));
    await screen.findByText(/Universitet üçün uyğun laptoplar tapdım/);
    fireEvent.click(screen.getByRole("button", { name: "Debug" }));

    expect(await screen.findByRole("dialog", { name: "Cavabın debug axını" })).toBeTruthy();
    expect(screen.getByText("Tapılıb: 3")).toBeTruthy();
    expect(screen.getByText("product_catalog")).toBeTruthy();
    expect(salesBotApi.getDebugTrace).toHaveBeenCalledWith(sessionId, {
      requestId: "22222222-2222-4222-8222-222222222222",
    });
  });
});
