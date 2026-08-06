import type { ImgHTMLAttributes } from "react";

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ChatApp } from "@/components/ChatApp";
import { SalesBotApiError, salesBotApi } from "@/lib/api";

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
      answer: "Universitet üçün uyğun laptoplar tapdım.",
      used_tools: ["product_search"],
      finish_reason: "completed",
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
    expect(await screen.findByText("Universitet üçün uyğun laptoplar tapdım.")).toBeTruthy();

    const textarea = screen.getByLabelText("Məhsul haqqında sual");
    fireEvent.change(textarea, { target: { value: "Ən ucuzu hansıdır?" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(await screen.findByText("Ən münasib variant birinci məhsuldur.")).toBeTruthy();

    expect(salesBotApi.createSession).toHaveBeenCalledTimes(1);
    expect(salesBotApi.sendMessage).toHaveBeenNthCalledWith(1, sessionId, "Universitet üçün laptop lazımdır");
    expect(salesBotApi.sendMessage).toHaveBeenNthCalledWith(2, sessionId, "Ən ucuzu hansıdır?");
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
    await screen.findByText("Universitet üçün uyğun laptoplar tapdım.");
    fireEvent.click(screen.getByRole("button", { name: "Yeni söhbət" }));

    expect(screen.getByRole("heading", { name: "Bu gün sənə necə kömək edə bilərəm?" })).toBeTruthy();
    const recent = screen.getByTitle("Universitet üçün laptop lazımdır");
    fireEvent.click(recent);
    expect(await screen.findByText("Universitet üçün uyğun laptoplar tapdım.")).toBeTruthy();
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
});
