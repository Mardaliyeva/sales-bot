import { describe, expect, it } from "vitest";

import {
  CHAT_STORAGE_KEY,
  loadSessions,
  MAX_MESSAGES_PER_SESSION,
  MAX_STORED_SESSIONS,
  saveSessions,
} from "@/lib/chatStorage";
import type { LocalChatSession } from "@/lib/types";

function session(index: number, messageCount = 1): LocalChatSession {
  const timestamp = new Date(Date.UTC(2026, 7, 1, 0, index)).toISOString();
  return {
    localId: `local-${index}`,
    backendSessionId: `backend-${index}`,
    title: `Söhbət ${index}`,
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: Array.from({ length: messageCount }, (_, messageIndex) => ({
      id: `message-${index}-${messageIndex}`,
      role: messageIndex % 2 ? ("assistant" as const) : ("user" as const),
      content: `Mətn ${messageIndex}`,
      createdAt: timestamp,
    })),
  };
}

describe("chatStorage", () => {
  it("returns an empty list for invalid or old storage", () => {
    window.localStorage.setItem(CHAT_STORAGE_KEY, "not-json");
    expect(loadSessions()).toEqual([]);

    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ version: 0, sessions: [session(1)] }));
    expect(loadSessions()).toEqual([]);
  });

  it("sorts recent sessions and enforces storage limits", () => {
    const stored = saveSessions(
      Array.from({ length: MAX_STORED_SESSIONS + 4 }, (_, index) =>
        session(index, MAX_MESSAGES_PER_SESSION + 5),
      ),
    );

    expect(stored).toHaveLength(MAX_STORED_SESSIONS);
    expect(stored[0].localId).toBe(`local-${MAX_STORED_SESSIONS + 3}`);
    expect(stored[0].messages).toHaveLength(MAX_MESSAGES_PER_SESSION);
    expect(loadSessions()).toEqual(stored);
  });

  it("does not persist unused empty chats", () => {
    expect(saveSessions([{ ...session(1), messages: [] }])).toEqual([]);
    expect(loadSessions()).toEqual([]);
  });
});
