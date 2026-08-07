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

  it("preserves optional debug identifiers without changing the storage version", () => {
    const item = session(1);
    item.messages[0] = {
      ...item.messages[0],
      role: "assistant",
      requestId: "22222222-2222-4222-8222-222222222222",
      usedTools: ["product_search"],
    };

    saveSessions([item]);

    expect(loadSessions()[0].messages[0]).toMatchObject({
      requestId: "22222222-2222-4222-8222-222222222222",
      usedTools: ["product_search"],
    });
    expect(JSON.parse(window.localStorage.getItem(CHAT_STORAGE_KEY) || "{}").version).toBe(1);
  });

  it("preserves product card presentations without changing the storage version", () => {
    const item = session(1);
    item.messages[0] = {
      ...item.messages[0],
      role: "assistant",
      presentation: {
        type: "product_cards",
        title: "2 uyğun məhsul tapdım",
        total: 2,
        shown_count: 1,
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
            highlights: ["QLED", "8K UHD"],
            budget_remaining: 630.01,
          },
        ],
      },
    };

    saveSessions([item]);

    expect(loadSessions()[0].messages[0].presentation).toMatchObject({
      type: "product_cards",
      result_kind: "matches",
      relaxed_fields: [],
      recommended_product_id: "prd_televisions_008",
      items: [{ sku: "SYN-TV-SMS-008", differences: [] }],
    });
    expect(JSON.parse(window.localStorage.getItem(CHAT_STORAGE_KEY) || "{}").version).toBe(1);
  });

  it("drops sessions containing invalid product card presentations", () => {
    const item = session(1) as LocalChatSession & {
      messages: Array<LocalChatSession["messages"][number] & { presentation?: unknown }>;
    };
    item.messages[0].presentation = {
      type: "product_cards",
      title: "Etibarsız",
      items: [{ price: "569,99" }],
    };
    window.localStorage.setItem(
      CHAT_STORAGE_KEY,
      JSON.stringify({ version: 1, sessions: [item] }),
    );

    expect(loadSessions()).toEqual([]);
  });
});
