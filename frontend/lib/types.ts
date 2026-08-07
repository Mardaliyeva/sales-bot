export type ChatRole = "user" | "assistant";
export type MessageState = "normal" | "error" | "session_expired";

export type ProductCardItem = {
  product_id: string;
  name: string;
  sku: string;
  price: number;
  currency: string;
  stock_status: "in_stock" | "out_of_stock";
  rating: number;
  warranty_months: number;
  highlights: string[];
  differences?: string[];
  budget_remaining?: number;
};

export type ProductCardsPresentation = {
  type: "product_cards";
  result_kind?: "matches" | "alternatives";
  requested_label?: string | null;
  title: string;
  total: number;
  shown_count: number;
  recommended_product_id: string;
  relaxed_fields?: string[];
  items: ProductCardItem[];
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  state?: MessageState;
  requestId?: string;
  usedTools?: string[];
  presentation?: ProductCardsPresentation;
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
  presentation?: ProductCardsPresentation;
};

export type ErrorResponse = {
  request_id: string | null;
  error: {
    code: string;
    message: string;
  };
};

export type DebugDiagnosis = {
  code: string;
  title: string;
  detail: string;
  catalog_checked: boolean;
  data_status: string;
  result_count: number | null;
  error_type?: string | null;
  observed_outcome?: string;
  match_status?: "exact_match" | "matching_products" | "alternatives" | "not_found";
  strict_total?: number;
  relaxed_fields?: string[];
};

export type DebugCandidate = {
  product_id: string;
  name?: string | null;
  rank?: number;
  score?: number;
  exact?: boolean;
  sale_price?: number | null;
  rating?: number | null;
  selected?: boolean;
};

export type DebugRetrieval = {
  mode: string;
  query: string;
  filters: Record<string, unknown>;
  sort: string;
  qdrant_checked: boolean;
  filtered_count: number;
  exact_product_ids: string[];
  matching_exact_product_ids?: string[];
  exact_filter_conflict: boolean;
  exact_candidates: DebugCandidate[];
  semantic_candidates: DebugCandidate[];
  semantic_state: string;
  match_status?: "exact_match" | "matching_products" | "alternatives" | "not_found";
  requested_label?: string | null;
  strict_total?: number;
  relaxed_fields?: string[];
  alternative_stages?: Array<{
    relaxed_fields: string[];
    filters: Record<string, unknown>;
    filtered_count: number;
    candidate_count: number;
    selected_count: number;
  }>;
  sorted_candidates: DebugCandidate[];
  hydrated_product_ids: string[];
  returned_product_ids: string[];
  total: number;
};

export type DebugTimelineEvent = {
  stage: string;
  status: string;
  round?: number;
  tools_allowed?: boolean;
  decision?: string;
  tool_name?: string | null;
  arguments?: Record<string, unknown>;
  result?: {
    status?: string;
    code?: string | null;
    total?: number | null;
    match_status?: string | null;
    strict_total?: number | null;
    relaxed_fields?: string[];
    returned_products?: Array<{ product_id?: string; name?: string }>;
  };
  retrieval?: DebugRetrieval | null;
  [key: string]: unknown;
};

export type DebugTraceResponse = {
  trace_version: number;
  detail_level: "full" | "legacy_partial";
  request_id: string;
  session_id: string;
  message_id: string | null;
  status: "running" | "completed" | "failed";
  model: Record<string, unknown>;
  diagnosis: DebugDiagnosis | null;
  data_sources: Record<string, Record<string, unknown>>;
  timeline: DebugTimelineEvent[];
  warnings: Array<{ code: string; detail: string }>;
  metrics: Record<string, number | string | null>;
};
