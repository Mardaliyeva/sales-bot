import type {
  ChatMessage,
  LocalChatSession,
  ProductCardItem,
  ProductCardsPresentation,
} from "@/lib/types";

export const CHAT_STORAGE_KEY = "sales-bot.chat-history.v1";
export const MAX_STORED_SESSIONS = 20;
export const MAX_MESSAGES_PER_SESSION = 100;
const STORAGE_VERSION = 1;

type StoredChatState = {
  version: number;
  sessions: LocalChatSession[];
};

function isProductCardItem(value: unknown): value is ProductCardItem {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<ProductCardItem>;
  return (
    typeof item.product_id === "string" &&
    typeof item.name === "string" &&
    typeof item.sku === "string" &&
    typeof item.price === "number" &&
    typeof item.currency === "string" &&
    (item.stock_status === "in_stock" || item.stock_status === "out_of_stock") &&
    typeof item.rating === "number" &&
    typeof item.warranty_months === "number" &&
    Array.isArray(item.highlights) &&
    item.highlights.every((highlight) => typeof highlight === "string") &&
    (item.differences === undefined ||
      (Array.isArray(item.differences) && item.differences.every((difference) => typeof difference === "string"))) &&
    (item.budget_remaining === undefined || typeof item.budget_remaining === "number")
  );
}

function isProductCardsPresentation(value: unknown): value is ProductCardsPresentation {
  if (!value || typeof value !== "object") return false;
  const presentation = value as Partial<ProductCardsPresentation>;
  return (
    presentation.type === "product_cards" &&
    (presentation.result_kind === undefined ||
      presentation.result_kind === "matches" ||
      presentation.result_kind === "alternatives") &&
    (presentation.requested_label === undefined ||
      presentation.requested_label === null ||
      typeof presentation.requested_label === "string") &&
    typeof presentation.title === "string" &&
    typeof presentation.total === "number" &&
    typeof presentation.shown_count === "number" &&
    typeof presentation.recommended_product_id === "string" &&
    (presentation.relaxed_fields === undefined ||
      (Array.isArray(presentation.relaxed_fields) &&
        presentation.relaxed_fields.every((field) => typeof field === "string"))) &&
    Array.isArray(presentation.items) &&
    presentation.items.every(isProductCardItem)
  );
}

function normalizePresentation(presentation: ProductCardsPresentation): ProductCardsPresentation {
  return {
    ...presentation,
    result_kind: presentation.result_kind ?? "matches",
    relaxed_fields: presentation.relaxed_fields ?? [],
    items: presentation.items.map((item) => ({
      ...item,
      differences: item.differences ?? [],
    })),
  };
}

function isMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<ChatMessage>;
  return (
    typeof message.id === "string" &&
    (message.role === "user" || message.role === "assistant") &&
    typeof message.content === "string" &&
    typeof message.createdAt === "string" &&
    (message.requestId === undefined || typeof message.requestId === "string") &&
    (message.usedTools === undefined ||
      (Array.isArray(message.usedTools) && message.usedTools.every((item) => typeof item === "string"))) &&
    (message.presentation === undefined || isProductCardsPresentation(message.presentation))
  );
}

function isSession(value: unknown): value is LocalChatSession {
  if (!value || typeof value !== "object") return false;
  const session = value as Partial<LocalChatSession>;
  return (
    typeof session.localId === "string" &&
    (typeof session.backendSessionId === "string" || session.backendSessionId === null) &&
    typeof session.title === "string" &&
    typeof session.createdAt === "string" &&
    typeof session.updatedAt === "string" &&
    Array.isArray(session.messages) &&
    session.messages.every(isMessage)
  );
}

export function loadSessions(storage: Storage = window.localStorage): LocalChatSession[] {
  try {
    const raw = storage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Partial<StoredChatState>;
    if (parsed.version !== STORAGE_VERSION || !Array.isArray(parsed.sessions)) return [];
    return parsed.sessions
      .filter(isSession)
      .map((session) => ({
        ...session,
        messages: session.messages.slice(-MAX_MESSAGES_PER_SESSION).map((message) => ({
          ...message,
          presentation: message.presentation
            ? normalizePresentation(message.presentation)
            : undefined,
        })),
      }))
      .filter((session) => session.messages.length > 0)
      .sort((left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt))
      .slice(0, MAX_STORED_SESSIONS);
  } catch {
    return [];
  }
}

export function saveSessions(
  sessions: LocalChatSession[],
  storage: Storage = window.localStorage,
): LocalChatSession[] {
  const normalized = sessions
    .filter((session) => session.messages.length > 0)
    .map((session) => ({
      ...session,
      messages: session.messages.slice(-MAX_MESSAGES_PER_SESSION),
    }))
    .sort((left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt))
    .slice(0, MAX_STORED_SESSIONS);
  storage.setItem(
    CHAT_STORAGE_KEY,
    JSON.stringify({ version: STORAGE_VERSION, sessions: normalized } satisfies StoredChatState),
  );
  return normalized;
}
