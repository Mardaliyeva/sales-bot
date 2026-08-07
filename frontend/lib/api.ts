import type {
  ChatResponse,
  DebugTraceResponse,
  ErrorResponse,
  SessionCreateResponse,
} from "@/lib/types";

const API_PREFIX = "/backend";

export class SalesBotApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly requestId: string | null = null,
  ) {
    super(message);
    this.name = "SalesBotApiError";
  }
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init.headers,
      },
      cache: "no-store",
    });
  } catch {
    throw new SalesBotApiError(0, "network_error", "Serverlə əlaqə yaratmaq mümkün olmadı.");
  }

  if (!response.ok) {
    let payload: ErrorResponse | null = null;
    try {
      payload = (await response.json()) as ErrorResponse;
    } catch {
      // A provider or proxy can return a non-JSON error page.
    }
    throw new SalesBotApiError(
      response.status,
      payload?.error?.code || "request_failed",
      payload?.error?.message || "Sorğu tamamlanmadı.",
      payload?.request_id || null,
    );
  }
  return response.json() as Promise<T>;
}

export const salesBotApi = {
  createSession: () =>
    request<SessionCreateResponse>("/v1/sessions", {
      method: "POST",
      body: "{}",
    }),
  sendMessage: (sessionId: string, message: string) =>
    request<ChatResponse>("/v1/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    }),
  getDebugTrace: (
    sessionId: string,
    lookup: { requestId?: string; messageId?: string },
  ) => {
    const params = new URLSearchParams({ session_id: sessionId });
    if (lookup.requestId) params.set("request_id", lookup.requestId);
    else if (lookup.messageId) params.set("message_id", lookup.messageId);
    return request<DebugTraceResponse>(`/v1/debug/traces?${params.toString()}`, {
      method: "GET",
    });
  },
};
