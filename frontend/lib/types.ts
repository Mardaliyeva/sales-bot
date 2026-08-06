export type ChatRole = "user" | "assistant";
export type MessageState = "normal" | "error" | "session_expired";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  state?: MessageState;
};

export type LocalChatSession = {
  localId: string;
  backendSessionId: string | null;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
};

export type SessionCreateResponse = {
  session_id: string;
  status: "active";
  expires_at: string;
};

export type ChatResponse = {
  request_id: string;
  session_id: string;
  message_id: string;
  answer: string;
  used_tools: string[];
  finish_reason: "completed";
};

export type ErrorResponse = {
  request_id: string | null;
  error: {
    code: string;
    message: string;
  };
};
