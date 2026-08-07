"use client";

import Image from "next/image";
import {
  AlertCircle,
  Bug,
  CheckCircle2,
  ChevronDown,
  CircleX,
  LoaderCircle,
  Menu,
  PanelLeftClose,
  PlusCircle,
  SendHorizontal,
  ShieldCheck,
  Star,
} from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { SalesBotApiError, salesBotApi } from "@/lib/api";
import { loadSessions, MAX_MESSAGES_PER_SESSION, saveSessions } from "@/lib/chatStorage";
import type {
  ChatMessage,
  DebugTraceResponse,
  LocalChatSession,
  MessageState,
  ProductCardsPresentation,
} from "@/lib/types";
import { DebugDrawer } from "@/components/DebugDrawer";

const MAX_MESSAGE_LENGTH = 4000;
const EMPTY_TITLE = "Yeni söhbət";
const SUGGESTIONS = [
  "Universitet üçün laptop lazımdır",
  "Qara 128 GB iPhone göstər",
  "12000 BTU kondisioner varmı?",
];

function debugPanelEnabled() {
  return process.env.NEXT_PUBLIC_DEBUG_PANEL === "true";
}

function createId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createEmptySession(): LocalChatSession {
  const now = new Date().toISOString();
  return {
    localId: createId(),
    backendSessionId: null,
    title: EMPTY_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function titleFromMessage(message: string) {
  return message.replace(/\s+/g, " ").trim().slice(0, 80) || EMPTY_TITLE;
}

function visibleError(error: unknown): { content: string; state: MessageState } {
  if (!(error instanceof SalesBotApiError)) {
    return { content: "Gözlənilməyən xəta baş verdi. Yenidən cəhd edin.", state: "error" };
  }
  if (error.status === 404 || error.status === 410) {
    return {
      content: "Bu söhbətin vaxtı bitib. Davam etmək üçün yeni söhbət açın.",
      state: "session_expired",
    };
  }
  if (error.status === 409) {
    return { content: "Bu söhbətdə başqa sorğu işlənir. Bir az sonra yenidən cəhd edin.", state: "error" };
  }
  if (error.status === 422) {
    return { content: "Mesaj qəbul edilmədi. Mətni yoxlayıb yenidən göndərin.", state: "error" };
  }
  if (error.status === 503) {
    return { content: "Köməkçi hazırda əlçatan deyil. Bir qədər sonra yenidən cəhd edin.", state: "error" };
  }
  return { content: error.message || "Sorğu tamamlanmadı. Yenidən cəhd edin.", state: "error" };
}

const priceFormatter = new Intl.NumberFormat("az-AZ", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatPrice(value: number, currency: string) {
  const formatted = priceFormatter
    .formatToParts(value)
    .map((part) => {
      if (part.type === "decimal") return ",";
      if (part.type === "group") return "\u00a0";
      return part.value;
    })
    .join("");
  return `${formatted} ${currency}`;
}

function InlineMarkdown({ text }: { text: string }) {
  return (
    <>
      {text
        .split(/(\*\*[^*\n]+\*\*)/g)
        .filter(Boolean)
        .map((part, index) =>
          part.startsWith("**") && part.endsWith("**") ? (
            <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>
          ) : (
            <span key={`${part}-${index}`}>{part}</span>
          ),
        )}
    </>
  );
}

function splitProductAnswer(answer: string) {
  const paragraphs = answer
    .trim()
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
  const lastParagraph = paragraphs.at(-1);
  const hasFollowUp = paragraphs.length > 1 && Boolean(lastParagraph?.endsWith("?"));

  return {
    summary: (hasFollowUp ? paragraphs.slice(0, -1) : paragraphs).join("\n\n"),
    followUp: hasFollowUp ? lastParagraph : undefined,
  };
}

function ProductCardsMessage({
  presentation,
  answer,
}: {
  presentation: ProductCardsPresentation;
  answer: string;
}) {
  const alternatives = presentation.result_kind === "alternatives";
  const response = splitProductAnswer(answer);

  return (
    <section
      className={`message-bubble product-results${alternatives ? " alternatives" : ""}`}
      aria-label={alternatives ? "Yaxın məhsul alternativləri" : "Məhsul seçimləri"}
    >
      <div className="product-answer">
        <InlineMarkdown text={response.summary} />
      </div>

      <ul className="product-card-list">
        {presentation.items.map((item) => {
          const recommended = item.product_id === presentation.recommended_product_id;
          const inStock = item.stock_status === "in_stock";
          return (
            <li className={`product-card${recommended ? " recommended" : ""}`} key={item.product_id}>
              <div className="product-card-heading">
                <div className="product-card-title">
                  {recommended && alternatives ? (
                    <span className="product-recommended-label">
                      <CheckCircle2 size={14} aria-hidden="true" />
                      Ən yaxın alternativ
                    </span>
                  ) : null}
                  <h3>{item.name}</h3>
                  <code>{item.sku}</code>
                </div>
                <div className="product-card-price">
                  <strong>{formatPrice(item.price, item.currency)}</strong>
                  <span className={inStock ? "in-stock" : "out-of-stock"}>
                    {inStock ? (
                      <CheckCircle2 size={15} aria-hidden="true" />
                    ) : (
                      <CircleX size={15} aria-hidden="true" />
                    )}
                    {inStock ? "Stokda" : "Stokda yoxdur"}
                  </span>
                </div>
              </div>

              {item.highlights.length ? (
                <ul className="product-highlights" aria-label={`${item.name} xüsusiyyətləri`}>
                  {item.highlights.map((highlight) => (
                    <li key={highlight}>{highlight}</li>
                  ))}
                </ul>
              ) : null}

              {alternatives && item.differences?.length ? (
                <ul className="product-differences" aria-label={`${item.name} fərqləri`}>
                  {item.differences.map((difference) => (
                    <li key={difference}>{difference}</li>
                  ))}
                </ul>
              ) : null}

              <div className="product-card-meta">
                <span>
                  <Star size={15} aria-hidden="true" />
                  <strong>{item.rating.toFixed(1)}</strong>
                  <span className="product-meta-label">reytinq</span>
                </span>
                <span>
                  <ShieldCheck size={15} aria-hidden="true" />
                  <strong>{item.warranty_months} ay</strong>
                  <span className="product-meta-label">zəmanət</span>
                </span>
                {item.budget_remaining !== undefined ? (
                  <span className="budget-remaining">
                    Büdcənizdə <strong>{formatPrice(item.budget_remaining, item.currency)}</strong> qalır
                  </span>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>

      {response.followUp ? (
        <div className="product-follow-up">
          <InlineMarkdown text={response.followUp} />
        </div>
      ) : null}
    </section>
  );
}

function MessageBubble({
  message,
  onNewChat,
  onDebug,
}: {
  message: ChatMessage;
  onNewChat: () => void;
  onDebug?: (message: ChatMessage) => void;
}) {
  const assistant = message.role === "assistant";
  const productPresentation = assistant && !message.state ? message.presentation : undefined;
  return (
    <article className={`message-row ${message.role}`} data-message-id={message.id}>
      {assistant ? (
        <span className="message-avatar" aria-hidden="true">
          <Image src="/kontakt-robot.png" alt="" width={64} height={64} />
        </span>
      ) : null}
      <div className={`message-content${productPresentation ? " product-message-content" : ""}`}>
        {productPresentation ? (
          <ProductCardsMessage presentation={productPresentation} answer={message.content} />
        ) : (
          <div className={`message-bubble${message.state ? ` ${message.state}` : ""}`}>
            {message.state ? <AlertCircle size={17} aria-hidden="true" /> : null}
            <div className="message-text">
              <InlineMarkdown text={message.content} />
            </div>
            {message.state === "session_expired" ? (
              <button className="inline-new-chat" type="button" onClick={onNewChat}>
                Yeni söhbət aç
              </button>
            ) : null}
          </div>
        )}
        {assistant && !message.state && onDebug ? (
          <button className="message-debug-button" type="button" onClick={() => onDebug(message)}>
            <Bug size={14} aria-hidden="true" />
            Debug
          </button>
        ) : null}
      </div>
    </article>
  );
}

export function ChatApp() {
  const [sessions, setSessions] = useState<LocalChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<LocalChatSession>(() => createEmptySession());
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [recentsOpen, setRecentsOpen] = useState(true);
  const [debugOpen, setDebugOpen] = useState(false);
  const [debugLoading, setDebugLoading] = useState(false);
  const [debugError, setDebugError] = useState<string | null>(null);
  const [debugTrace, setDebugTrace] = useState<DebugTraceResponse | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Browser storage is intentionally loaded after hydration.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSessions(loadSessions());
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeSession.messages.length, busy]);

  const recentSessions = useMemo(
    () => sessions.filter((session) => session.messages.length > 0),
    [sessions],
  );

  function commitSession(nextSession: LocalChatSession) {
    setActiveSession(nextSession);
    setSessions((current) => {
      const merged = [nextSession, ...current.filter((item) => item.localId !== nextSession.localId)];
      return saveSessions(merged);
    });
  }

  function handleNewChat() {
    if (busy) return;
    setActiveSession(createEmptySession());
    setDraft("");
    setSidebarOpen(false);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function selectSession(session: LocalChatSession) {
    if (busy) return;
    setActiveSession(session);
    setDraft("");
    setSidebarOpen(false);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  async function openDebug(message: ChatMessage) {
    if (!activeSession.backendSessionId) return;
    setDebugOpen(true);
    setDebugLoading(true);
    setDebugError(null);
    setDebugTrace(null);
    try {
      const trace = await salesBotApi.getDebugTrace(
        activeSession.backendSessionId,
        message.requestId ? { requestId: message.requestId } : { messageId: message.id },
      );
      setDebugTrace(trace);
    } catch (error) {
      setDebugError(
        error instanceof SalesBotApiError
          ? error.message
          : "Debug trace-i yükləmək mümkün olmadı.",
      );
    } finally {
      setDebugLoading(false);
    }
  }

  async function submitMessage(value = draft) {
    const message = value.trim();
    if (!message || message.length > MAX_MESSAGE_LENGTH || busy) return;

    setDraft("");
    setBusy(true);
    const now = new Date().toISOString();
    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: message,
      createdAt: now,
    };
    let workingSession: LocalChatSession = {
      ...activeSession,
      title: activeSession.messages.length ? activeSession.title : titleFromMessage(message),
      updatedAt: now,
      messages: [...activeSession.messages, userMessage].slice(-MAX_MESSAGES_PER_SESSION),
    };
    commitSession(workingSession);

    try {
      let backendSessionId = workingSession.backendSessionId;
      if (!backendSessionId) {
        const created = await salesBotApi.createSession();
        backendSessionId = created.session_id;
        workingSession = { ...workingSession, backendSessionId };
        commitSession(workingSession);
      }
      const response = await salesBotApi.sendMessage(backendSessionId, message);
      const assistantMessage: ChatMessage = {
        id: response.message_id,
        role: "assistant",
        content: response.answer,
        createdAt: new Date().toISOString(),
        requestId: response.request_id,
        usedTools: response.used_tools,
        presentation: response.presentation,
      };
      workingSession = {
        ...workingSession,
        updatedAt: assistantMessage.createdAt,
        messages: [...workingSession.messages, assistantMessage].slice(-MAX_MESSAGES_PER_SESSION),
      };
      commitSession(workingSession);
    } catch (error) {
      const visible = visibleError(error);
      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        content: visible.content,
        createdAt: new Date().toISOString(),
        state: visible.state,
        requestId: error instanceof SalesBotApiError ? error.requestId || undefined : undefined,
      };
      workingSession = {
        ...workingSession,
        updatedAt: assistantMessage.createdAt,
        messages: [...workingSession.messages, assistantMessage].slice(-MAX_MESSAGES_PER_SESSION),
      };
      commitSession(workingSession);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitMessage();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitMessage();
    }
  }

  const hasMessages = activeSession.messages.length > 0;
  return (
    <div className={`sales-chat${sidebarOpen ? "" : " sidebar-hidden"}`}>
      <button
        className="sidebar-open-button"
        type="button"
        aria-label="Söhbət panelini aç"
        onClick={() => setSidebarOpen(true)}
      >
        <Menu size={20} aria-hidden="true" />
      </button>

      <button
        className={`sidebar-backdrop${sidebarOpen ? " open" : ""}`}
        type="button"
        aria-label="Söhbət panelini bağla"
        onClick={() => setSidebarOpen(false)}
      />

      <aside className={`sidebar${sidebarOpen ? " open" : ""}`} aria-label="Söhbət naviqasiyası">
        <header className="sidebar-brand">
          <div className="brand-lockup">
            <Image src="/kontakt-robot-head.png" alt="" width={44} height={44} priority />
            <strong>Kontakt Satış Köməkçisi</strong>
          </div>
          <button
            className="sidebar-icon-button"
            type="button"
            aria-label="Söhbət panelini bağla"
            onClick={() => setSidebarOpen(false)}
          >
            <PanelLeftClose size={19} aria-hidden="true" />
          </button>
        </header>

        <button className="new-chat-button" type="button" onClick={handleNewChat} disabled={busy}>
          <PlusCircle size={18} aria-hidden="true" />
          <span>Yeni söhbət</span>
        </button>

        <section className="recent-section" aria-label="Son söhbətlər">
          <button
            className="recent-heading"
            type="button"
            aria-expanded={recentsOpen}
            onClick={() => setRecentsOpen((current) => !current)}
          >
            <span>Son söhbətlər</span>
            <ChevronDown className={recentsOpen ? "" : "collapsed"} size={16} aria-hidden="true" />
          </button>
          {recentsOpen ? (
            <div className="recent-list">
              {recentSessions.length ? (
                recentSessions.map((session) => (
                  <button
                    className={`recent-item${session.localId === activeSession.localId ? " active" : ""}`}
                    type="button"
                    key={session.localId}
                    title={session.title}
                    disabled={busy}
                    onClick={() => selectSession(session)}
                  >
                    {session.title}
                  </button>
                ))
              ) : (
                <p className="recent-empty">Hələ söhbət yoxdur.</p>
              )}
            </div>
          ) : null}
        </section>
      </aside>

      <main className="chat-area">
        <section className={`chat-thread${hasMessages ? " has-messages" : ""}`} aria-live="polite">
          {!hasMessages ? (
            <div className="empty-state">
              <Image src="/kontakt-robot.png" alt="Kontakt satış köməkçisi" width={92} height={92} priority />
              <h1>Bu gün sənə necə kömək edə bilərəm?</h1>
            </div>
          ) : (
            <div className="message-list">
              {activeSession.messages.map((message) => (
                <MessageBubble
                  message={message}
                  key={message.id}
                  onNewChat={handleNewChat}
                  onDebug={
                    debugPanelEnabled() && activeSession.backendSessionId
                      ? (selected) => void openDebug(selected)
                      : undefined
                  }
                />
              ))}
              {busy ? (
                <article className="message-row assistant loading" aria-label="Cavab hazırlanır">
                  <span className="message-avatar" aria-hidden="true">
                    <Image src="/kontakt-robot.png" alt="" width={64} height={64} />
                  </span>
                  <div className="message-bubble loading-bubble">
                    <LoaderCircle className="spinner" size={19} aria-hidden="true" />
                    <span>Cavab hazırlanır...</span>
                  </div>
                </article>
              ) : null}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <div className="composer-wrap">
          {!hasMessages ? (
            <div className="suggestions" aria-label="Təklif olunan suallar">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  type="button"
                  key={suggestion}
                  disabled={busy}
                  onClick={() => void submitMessage(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          ) : null}
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              value={draft}
              rows={2}
              maxLength={MAX_MESSAGE_LENGTH}
              aria-label="Məhsul haqqında sual"
              placeholder="Məhsul haqqında sualınızı yazın..."
              disabled={busy}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button type="submit" disabled={busy || !draft.trim() || draft.length > MAX_MESSAGE_LENGTH}>
              {busy ? <LoaderCircle className="spinner" size={18} aria-hidden="true" /> : <SendHorizontal size={18} aria-hidden="true" />}
              <span>{busy ? "Gözləyin" : "Göndər"}</span>
            </button>
          </form>
          <div className="composer-meta">
            <span>Cavablar cari qiymət, stok və kataloq məlumatlarına əsaslanır.</span>
            <span>{draft.length}/{MAX_MESSAGE_LENGTH}</span>
          </div>
        </div>
      </main>
      <DebugDrawer
        open={debugOpen}
        trace={debugTrace}
        loading={debugLoading}
        error={debugError}
        onClose={() => setDebugOpen(false)}
      />
    </div>
  );
}
