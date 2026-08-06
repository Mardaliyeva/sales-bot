import { afterEach, describe, expect, it, vi } from "vitest";

import { salesBotApi } from "@/lib/api";

describe("salesBotApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates a backend session through the same-origin proxy", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "11111111-1111-4111-8111-111111111111",
          status: "active",
          expires_at: "2026-08-12T10:00:00Z",
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await salesBotApi.createSession();

    expect(result.status).toBe("active");
    expect(fetchMock).toHaveBeenCalledWith(
      "/backend/v1/sessions",
      expect.objectContaining({ method: "POST", body: "{}" }),
    );
  });

  it("sends the current session id and message", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          request_id: "22222222-2222-4222-8222-222222222222",
          session_id: "11111111-1111-4111-8111-111111111111",
          message_id: "33333333-3333-4333-8333-333333333333",
          answer: "Üç laptop tapdım.",
          used_tools: ["product_search"],
          finish_reason: "completed",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await salesBotApi.sendMessage("11111111-1111-4111-8111-111111111111", "Laptop göstər");

    expect(fetchMock).toHaveBeenCalledWith(
      "/backend/v1/chat",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          session_id: "11111111-1111-4111-8111-111111111111",
          message: "Laptop göstər",
        }),
      }),
    );
  });

  it("maps API and network failures without exposing response internals", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(
          new Response(
            JSON.stringify({
              request_id: null,
              error: { code: "database_unavailable", message: "Məlumat bazası hazır deyil." },
            }),
            { status: 503, headers: { "Content-Type": "application/json" } },
          ),
        )
        .mockRejectedValueOnce(new Error("provider-internal-detail")),
    );

    await expect(salesBotApi.createSession()).rejects.toMatchObject({
      status: 503,
      code: "database_unavailable",
    });
    await expect(salesBotApi.createSession()).rejects.toMatchObject({
      status: 0,
      code: "network_error",
      message: "Serverlə əlaqə yaratmaq mümkün olmadı.",
    });
  });
});
