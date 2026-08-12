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
  result_kind?: "matches" | "alternatives" | "exact_conflict" | "comparison";
  requested_label?: string | null;
  title: string;
  total: number;
  shown_count: number;
  recommended_product_id: string | null;
  requested_item?: ProductCardItem | null;
  constraint_conflicts?: string[];
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
  documents_checked?: boolean;
  data_status: string;
  result_count: number | null;
  error_type?: string | null;
  observed_outcome?: string;
  match_status?: "exact_match" | "exact_conflict" | "matching_products" | "alternatives" | "clarification_required" | "found" | "not_found";
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
  retrieval_executed?: boolean;
  filtered_count: number;
  exact_product_ids: string[];
  matching_exact_product_ids?: string[];
  exact_filter_conflict: boolean;
  exact_candidates: DebugCandidate[];
  semantic_candidates: DebugCandidate[];
  semantic_state: string;
  match_status?: "exact_match" | "exact_conflict" | "matching_products" | "alternatives" | "clarification_required" | "not_found";
  original_arguments?: Record<string, unknown>;
  canonical_arguments?: Record<string, unknown>;
  argument_corrections?: Array<Record<string, unknown>>;
  facet_mapping?: Array<Record<string, unknown>>;
  unavailable_requested_values?: Array<Record<string, unknown>>;
  constraint_conflicts?: string[];
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

export type DebugDocumentCandidate = {
  chunk_id: string;
  document_id?: string | null;
  filename?: string | null;
  title?: string | null;
  heading?: string | null;
  score?: number;
  selected?: boolean;
  text_preview?: string;
};

export type DebugDocumentRetrieval = {
  mode: "document_qdrant_v1";
  query: string;
  qdrant_checked: boolean;
  semantic_state: string;
  min_score?: number | null;
  source_checksum?: string | null;
  candidate_count: number;
  selected_chunk_ids: string[];
  candidates: DebugDocumentCandidate[];
  error_type?: string | null;
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
  retrieval?: DebugRetrieval | DebugDocumentRetrieval | null;
  [key: string]: unknown;
};

export type DebugDecisionExplanation = {
  version: number;
  basis: string;
  summary: string;
  narrative?: string;
  understood_request: Record<string, unknown>;
  context_used: Array<{
    source: string;
    detail: string;
    memory_ids?: string[];
    revision?: number;
  }>;
  decision_path: Array<{ code: string; status: string; detail: string }>;
  evidence: Record<string, unknown>;
  outcome: Record<string, unknown>;
  memory_effect: DebugMemoryTransition;
  limitations: string[];
};

export type DebugMemoryTransition = {
  revision_before: number;
  revision_after: number;
  action: "preserve" | "replace" | "merge";
  changed_ids: string[];
  removed_ids: string[];
  size_bytes: number;
  context_source?: "confirmed" | "pending" | "none";
  summary_replaced?: boolean;
  summary_size_chars?: number;
  cache_expires_at?: string | null;
  confirmed_state_changed?: boolean;
  pending_state_changed?: boolean;
};

export type DebugTraceResponse = {
  trace_version: number;
  detail_level: "full" | "legacy_partial";
  request_id: string;
  session_id: string;
  message_id: string | null;
  status: "running" | "completed" | "failed";
  model: Record<string, unknown>;
  runtime?: Record<string, unknown>;
  prompt?: {
    mode?: "modular" | "legacy";
    active_phase?: "tool" | "response" | "safe_final";
    versions?: Record<string, string>;
    phase_hashes?: Record<string, string>;
    phase_chars?: Record<string, number>;
    phase_token_estimates?: Record<string, number>;
    legacy_chars?: number;
    tool_char_reduction_percent?: number;
    response_char_reduction_percent?: number;
  };
  diagnosis: DebugDiagnosis | null;
  data_sources: Record<string, Record<string, unknown>>;
  timeline: DebugTimelineEvent[];
  warnings: Array<{ code: string; detail: string }>;
  metrics: Record<string, number | string | null>;
  decision_explanation?: DebugDecisionExplanation | null;
  memory_transition?: DebugMemoryTransition | null;
  continuation_context_before?: string | null;
  continuation_context_after?: string | null;
};
