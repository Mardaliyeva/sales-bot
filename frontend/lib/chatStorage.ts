import type { ChatMessage, LocalChatSession } from "@/lib/types";

export const CHAT_STORAGE_KEY = "sales-bot.chat-history.v1";
export const MAX_STORED_SESSIONS = 20;
export const MAX_MESSAGES_PER_SESSION = 100;
const STORAGE_VERSION = 1;

type StoredChatState = {
  version: number;
  sessions: LocalChatSession[];
};

function isMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<ChatMessage>;
  return (
    typeof message.id === "string" &&
    (message.role === "user" || message.role === "assistant") &&
    typeof message.content === "string" &&
    typeof message.createdAt === "string"
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
        messages: session.messages.slice(-MAX_MESSAGES_PER_SESSION),
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
