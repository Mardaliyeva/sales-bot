"use client";

import {
  AlertTriangle,
  Bug,
  Check,
  ChevronDown,
  Copy,
  Database,
  Route,
  Search,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type {
  DebugCandidate,
  DebugDecisionExplanation,
  DebugDocumentCandidate,
  DebugDocumentRetrieval,
  DebugMemoryTransition,
  DebugRetrieval,
  DebugTimelineEvent,
  DebugTraceResponse,
} from "@/lib/types";

type DebugDrawerProps = {
  open: boolean;
  trace: DebugTraceResponse | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
};

const STAGE_LABELS: Record<string, string> = {
  input_validation: "Mesaj yoxlanıldı",
  context_build: "Söhbət konteksti hazırlandı",
  model_round: "Azure model mərhələsi",
  product_search: "Məhsul axtarışı",
  document_search: "Sənəd axtarışı",
  final_answer: "Yekun cavab",
  run_error: "Run xətası",
  legacy_run_summary: "Köhnə run məlumatı",
  session_memory_update: "Sessiya yaddaşı yeniləndi",
};

function displayValue(value: unknown): string {
  if (value === true) return "bəli";
  if (value === false) return "xeyr";
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  return String(value);
}

function formatScore(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(4) : "—";
}

function candidateMeta(candidate: DebugCandidate): string[] {
  const parts: string[] = [];
  if (candidate.exact) parts.push("exact");
  if (candidate.score !== undefined) parts.push(`score ${formatScore(candidate.score)}`);
  if (candidate.sale_price != null) parts.push(`${candidate.sale_price} AZN`);
  if (candidate.rating != null) parts.push(`reytinq ${candidate.rating}`);
  if (candidate.selected) parts.push("cavaba seçilib");
  return parts;
}

function CandidateList({
  title,
  candidates,
  showAll,
}: {
  title: string;
  candidates: DebugCandidate[];
  showAll: boolean;
}) {
  const visible = showAll ? candidates.slice(0, 20) : candidates.slice(0, 10);
  return (
    <div className="debug-candidate-group">
      <div className="debug-section-heading compact">
        <span>{title}</span>
        <span className="debug-count">{candidates.length}</span>
      </div>
      {visible.length ? (
        <ol className="debug-candidate-list">
          {visible.map((candidate, index) => (
            <li key={`${title}-${candidate.product_id}-${index}`}>
              <span className="candidate-rank">#{candidate.rank ?? index + 1}</span>
              <span className="candidate-content">
                <strong>{candidate.name || candidate.product_id}</strong>
                <code>{candidate.product_id}</code>
                <small>{candidateMeta(candidate).join(" · ") || "rank məlumatı yoxdur"}</small>
              </span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="debug-empty-line">Namizəd yoxdur.</p>
      )}
    </div>
  );
}

function timelineDetail(event: DebugTimelineEvent): string {
  if (event.stage === "model_round") {
    const decision = event.decision === "tool_call"
      ? `${event.tool_name || "tool"} seçildi`
      : event.decision === "forced_final"
        ? "Tool-suz yekun cavab tələb edildi"
        : "Birbaşa cavab verildi";
    return `Raund ${event.round}: ${decision}`;
  }
  if (event.stage === "product_search") {
    const total = event.result?.total;
    return event.status === "completed"
      ? `Axtarış tamamlandı${typeof total === "number" ? `, total ${total}` : ""}`
      : `Axtarış xətası: ${event.result?.code || "naməlum"}`;
  }
  if (event.stage === "document_search") {
    const total = event.result?.total;
    return event.status === "completed"
      ? `Sənəd axtarışı tamamlandı${typeof total === "number" ? `, total ${total}` : ""}`
      : `Sənəd axtarışı xətası: ${event.result?.code || "naməlum"}`;
  }
  if (typeof event.detail === "string") return event.detail;
  return event.status === "completed" ? "Tamamlandı" : event.status;
}

function DocumentCandidateList({ candidates }: { candidates: DebugDocumentCandidate[] }) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? candidates.slice(0, 20) : candidates.slice(0, 10);
  return (
    <>
      <ol className="debug-candidate-list">
        {visible.map((candidate, index) => (
          <li key={`${candidate.chunk_id}-${index}`}>
            <span className="candidate-rank">#{index + 1}</span>
            <span className="candidate-content">
              <strong>{candidate.title || candidate.document_id || candidate.chunk_id}</strong>
              <code>{candidate.filename || candidate.chunk_id}</code>
              <small>
                {candidate.heading || "Başlıq yoxdur"} · score {formatScore(candidate.score)}
                {candidate.selected ? " · modelə seçilib" : ""}
              </small>
              {candidate.text_preview ? <p>{candidate.text_preview}</p> : null}
            </span>
          </li>
        ))}
      </ol>
      {!visible.length ? <p className="debug-empty-line">Sənəd namizədi yoxdur.</p> : null}
      {candidates.length > 10 ? (
        <button className="debug-show-all" type="button" onClick={() => setShowAll((value) => !value)}>
          <ChevronDown className={showAll ? "expanded" : ""} size={16} aria-hidden="true" />
          {showAll ? "İlk 10-u göstər" : "20 sənəd hissəsinə qədər göstər"}
        </button>
      ) : null}
    </>
  );
}

function DocumentRetrievalDetails({ retrieval }: { retrieval: DebugDocumentRetrieval }) {
  return (
    <section className="debug-section">
      <div className="debug-section-heading">
        <Search size={17} aria-hidden="true" />
        <h3>Document retrieval nəticələri</h3>
      </div>
      <dl className="debug-kv-grid">
        <div><dt>Rejim</dt><dd>{retrieval.mode}</dd></div>
        <div><dt>Semantic vəziyyət</dt><dd>{retrieval.semantic_state}</dd></div>
        <div><dt>Qdrant yoxlanılıb</dt><dd>{retrieval.qdrant_checked ? "bəli" : "xeyr"}</dd></div>
        <div><dt>Namizəd sayı</dt><dd>{retrieval.candidate_count}</dd></div>
        <div><dt>Seçilmiş chunk</dt><dd>{retrieval.selected_chunk_ids.length}</dd></div>
        <div><dt>Minimum score</dt><dd>{formatScore(retrieval.min_score)}</dd></div>
      </dl>
      <div className="debug-code-block">
        <span>Tool sorğusu</span>
        <code>{retrieval.query}</code>
      </div>
      {retrieval.error_type ? (
        <div className="debug-warning"><AlertTriangle size={16} />{retrieval.error_type}</div>
      ) : null}
      <DocumentCandidateList candidates={retrieval.candidates ?? []} />
    </section>
  );
}

function RetrievalDetails({ retrieval }: { retrieval: DebugRetrieval }) {
  const [showAll, setShowAll] = useState(false);
  const exactCandidates = retrieval.exact_candidates ?? [];
  const semanticCandidates = retrieval.semantic_candidates ?? [];
  const sortedCandidates = retrieval.sorted_candidates ?? [];
  const hydratedProductIds = retrieval.hydrated_product_ids ?? retrieval.returned_product_ids ?? [];
  const hasMore = Math.max(
    exactCandidates.length,
    semanticCandidates.length,
    sortedCandidates.length,
  ) > 10;
  return (
    <section className="debug-section">
      <div className="debug-section-heading">
        <Search size={17} aria-hidden="true" />
        <h3>Retrieval nəticələri</h3>
      </div>
      <dl className="debug-kv-grid">
        <div><dt>Rejim</dt><dd>{retrieval.mode}</dd></div>
        <div><dt>Semantic vəziyyət</dt><dd>{retrieval.semantic_state}</dd></div>
        <div><dt>Uyğunluq statusu</dt><dd>{retrieval.match_status || "—"}</dd></div>
        <div><dt>Sərt nəticə sayı</dt><dd>{retrieval.strict_total ?? "—"}</dd></div>
        <div><dt>Qdrant yoxlanılıb</dt><dd>{retrieval.qdrant_checked ? "bəli" : "xeyr"}</dd></div>
        <div><dt>Filterdən keçən</dt><dd>{retrieval.filtered_count}</dd></div>
        <div><dt>Yekun total</dt><dd>{retrieval.total}</dd></div>
        <div><dt>JSON-dan götürülən</dt><dd>{hydratedProductIds.length}</dd></div>
      </dl>
      <div className="debug-code-block">
        <span>Tool sorğusu</span>
        <code>{retrieval.query}</code>
      </div>
      <div className="debug-code-block">
        <span>Tətbiq olunan filterlər</span>
        <pre>{JSON.stringify(retrieval.filters, null, 2)}</pre>
      </div>
      {retrieval.relaxed_fields?.length ? (
        <div className="debug-code-block">
          <span>Yumşaldılmış field-lər</span>
          <code>{retrieval.relaxed_fields.join(", ")}</code>
        </div>
      ) : null}
      {retrieval.alternative_stages?.length ? (
        <div className="debug-code-block">
          <span>Alternativ mərhələləri</span>
          <pre>{JSON.stringify(retrieval.alternative_stages, null, 2)}</pre>
        </div>
      ) : null}
      {retrieval.exact_filter_conflict ? (
        <div className="debug-warning">
          <AlertTriangle size={16} aria-hidden="true" />
          Exact identifier tapıldı, lakin filterlə ziddiyyət yarandı.
        </div>
      ) : null}
      <CandidateList title="Qdrant exact" candidates={exactCandidates} showAll={showAll} />
      <CandidateList title="Qdrant semantic" candidates={semanticCandidates} showAll={showAll} />
      <CandidateList title="Sıralanmış namizədlər" candidates={sortedCandidates} showAll={showAll} />
      {hasMore ? (
        <button className="debug-show-all" type="button" onClick={() => setShowAll((value) => !value)}>
          <ChevronDown className={showAll ? "expanded" : ""} size={16} aria-hidden="true" />
          {showAll ? "İlk 10-u göstər" : "20 namizədə qədər göstər"}
        </button>
      ) : null}
    </section>
  );
}

function DecisionExplanationDetails({
  explanation,
  transition,
}: {
  explanation: DebugDecisionExplanation;
  transition: DebugMemoryTransition | null;
}) {
  const memory = transition ?? explanation.memory_effect;
  return (
    <section className="debug-section">
      <div className="debug-section-heading">
        <Route size={17} aria-hidden="true" />
        <h3>Cavabın əsaslandırması</h3>
      </div>
      <div className="debug-warning">
        Bu bölmə modelin gizli düşüncəsi deyil; yoxlanılmış plan, mənbə və runtime qərarlarının xülasəsidir.
      </div>
      <p>{explanation.narrative ?? explanation.summary}</p>
      <details className="debug-code-block">
        <summary>Texniki detallar</summary>
      <dl className="debug-kv-grid">
        <div><dt>Əsas</dt><dd>{explanation.basis}</dd></div>
        <div><dt>Memory action</dt><dd>{memory.action}</dd></div>
        <div><dt>Revision</dt><dd>{memory.revision_before} → {memory.revision_after}</dd></div>
        <div><dt>Memory ölçüsü</dt><dd>{memory.size_bytes} bayt</dd></div>
      </dl>
      <div className="debug-candidate-group">
        <div className="debug-section-heading compact"><span>İstifadə olunan kontekst</span></div>
        <ol className="debug-candidate-list">
          {explanation.context_used.map((item, index) => (
            <li key={`${item.source}-${index}`}>
              <span className="candidate-rank">#{index + 1}</span>
              <span className="candidate-content">
                <strong>{item.source}</strong>
                <small>{item.detail}</small>
                {item.memory_ids?.length ? <code>{item.memory_ids.join(", ")}</code> : null}
              </span>
            </li>
          ))}
        </ol>
      </div>
      <div className="debug-candidate-group">
        <div className="debug-section-heading compact"><span>Qərar yolu</span></div>
        <ol className="debug-candidate-list">
          {explanation.decision_path.map((item, index) => (
            <li key={`${item.code}-${index}`}>
              <span className="candidate-rank">#{index + 1}</span>
              <span className="candidate-content">
                <strong>{item.code}</strong>
                <small>{item.status} · {item.detail}</small>
              </span>
            </li>
          ))}
        </ol>
      </div>
      <details className="debug-code-block">
        <summary>Sorğunun strukturlaşdırılmış mənası</summary>
        <pre>{JSON.stringify(explanation.understood_request, null, 2)}</pre>
      </details>
      <details className="debug-code-block">
        <summary>Sübutlar və nəticə</summary>
        <pre>{JSON.stringify({ evidence: explanation.evidence, outcome: explanation.outcome }, null, 2)}</pre>
      </details>
      <details className="debug-code-block">
        <summary>Yaddaş fərqi</summary>
        <pre>{JSON.stringify(memory, null, 2)}</pre>
      </details>
      </details>
      {explanation.limitations.length ? (
        <div className="debug-warning">
          <AlertTriangle size={16} aria-hidden="true" />
          {explanation.limitations.join(" ")}
        </div>
      ) : null}
    </section>
  );
}

export function DebugDrawer({ open, trace, loading, error, onClose }: DebugDrawerProps) {
  const [copiedRequestId, setCopiedRequestId] = useState(false);
  const [copiedTrace, setCopiedTrace] = useState(false);
  const productRetrieval = useMemo(
    () => (
      trace?.timeline.find(
        (event) => event.retrieval && event.retrieval.mode !== "document_qdrant_v1",
      )?.retrieval as DebugRetrieval | undefined
    ) ?? null,
    [trace],
  );
  const documentRetrieval = useMemo(
    () => (
      trace?.timeline.find((event) => event.retrieval?.mode === "document_qdrant_v1")
        ?.retrieval as DebugDocumentRetrieval | undefined
    ) ?? null,
    [trace],
  );

  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose, open]);

  async function copyRequestId() {
    if (!trace) return;
    await navigator.clipboard.writeText(trace.request_id);
    setCopiedRequestId(true);
    window.setTimeout(() => setCopiedRequestId(false), 1600);
  }

  async function copyDebugTrace() {
    if (!trace) return;
    await navigator.clipboard.writeText(JSON.stringify(trace, null, 2));
    setCopiedTrace(true);
    window.setTimeout(() => setCopiedTrace(false), 1600);
  }

  if (!open) return null;
  return (
    <div className="debug-layer" role="presentation">
      <button className="debug-backdrop" type="button" aria-label="Debug panelini bağla" onClick={onClose} />
      <aside className="debug-drawer" role="dialog" aria-modal="true" aria-labelledby="debug-title">
        <header className="debug-header">
          <div>
            <span className="debug-eyebrow"><Bug size={14} aria-hidden="true" /> Developer trace</span>
            <h2 id="debug-title">Cavabın debug axını</h2>
          </div>
          <div className="debug-header-actions">
            {trace ? (
              <button
                className="debug-copy debug-copy-all"
                type="button"
                title="Bütün debug trace-ni JSON formatında kopyala"
                onClick={() => void copyDebugTrace()}
              >
                {copiedTrace ? <Check size={15} /> : <Copy size={15} />}
                {copiedTrace ? "Debug kopyalandı" : "Debug-i kopyala"}
              </button>
            ) : null}
            <button className="debug-close" type="button" aria-label="Debug panelini bağla" onClick={onClose}>
              <X size={20} aria-hidden="true" />
            </button>
          </div>
        </header>

        <div className="debug-body">
          {loading ? <div className="debug-state">Trace yüklənir...</div> : null}
          {error ? <div className="debug-state error"><AlertTriangle size={18} />{error}</div> : null}
          {trace ? (
            <>
              {trace.detail_level === "legacy_partial" ? (
                <div className="debug-warning"><AlertTriangle size={16} />Bu köhnə run-dır; candidate rank və score-ları mövcud deyil.</div>
              ) : null}

              <section className={`debug-diagnosis ${trace.diagnosis?.code || "unknown"}`}>
                <span className="diagnosis-icon"><Database size={18} aria-hidden="true" /></span>
                <div>
                  <span className="debug-eyebrow">Datada varmı?</span>
                  <strong>{trace.diagnosis?.data_status || "Müəyyən deyil"}</strong>
                  <h3>{trace.diagnosis?.title || "Run davam edir"}</h3>
                  <p>{trace.diagnosis?.detail}</p>
                </div>
              </section>

              {trace.decision_explanation ? (
                <DecisionExplanationDetails
                  explanation={trace.decision_explanation}
                  transition={trace.memory_transition ?? null}
                />
              ) : (
                <section className="debug-section">
                  <div className="debug-section-heading"><Route size={17} /><h3>Cavabın əsaslandırması</h3></div>
                  <p className="debug-empty-line">Bu run üçün əsaslandırma mövcud deyil.</p>
                </section>
              )}

              {trace.continuation_context_after ? (
                <section className="debug-section">
                  <div className="debug-section-heading">
                    <Route size={17} aria-hidden="true" />
                    <h3>Sonrakı mesaj üçün söhbət konteksti</h3>
                  </div>
                  <div className="debug-warning">
                    Bu mətn modelin gizli düşüncəsi deyil; yoxlanılmış sessiya vəziyyətinin qısa xülasəsidir.
                  </div>
                  <p>{trace.continuation_context_after}</p>
                  {trace.continuation_context_before &&
                  trace.continuation_context_before !== trace.continuation_context_after ? (
                    <details className="debug-code-block">
                      <summary>Əvvəlki kontekst</summary>
                      <p>{trace.continuation_context_before}</p>
                    </details>
                  ) : null}
                  {trace.memory_transition ? (
                    <dl className="debug-kv-grid">
                      <div><dt>Mənbə</dt><dd>{trace.memory_transition.context_source ?? "none"}</dd></div>
                      <div><dt>Simvol sayı</dt><dd>{trace.memory_transition.summary_size_chars ?? 0}</dd></div>
                      <div><dt>Əvəz edildi</dt><dd>{trace.memory_transition.summary_replaced ? "bəli" : "xeyr"}</dd></div>
                      <div><dt>Cache bitmə vaxtı</dt><dd>{trace.memory_transition.cache_expires_at ?? "—"}</dd></div>
                    </dl>
                  ) : null}
                </section>
              ) : null}

              {trace.prompt ? (
                <section className="debug-section">
                  <div className="debug-section-heading">
                    <Route size={17} aria-hidden="true" />
                    <h3>Prompt modulları</h3>
                  </div>
                  <div className="debug-warning">
                    Burada yalnız versiya, hash və ölçü metrikləri göstərilir; system prompt mətni göstərilmir.
                  </div>
                  <dl className="debug-kv-grid">
                    <div><dt>Rejim</dt><dd>{trace.prompt.mode ?? "legacy"}</dd></div>
                    <div><dt>Aktiv phase</dt><dd>{trace.prompt.active_phase ?? "—"}</dd></div>
                    <div><dt>Tool ölçüsü</dt><dd>{trace.prompt.phase_chars?.tool ?? "—"} simvol</dd></div>
                    <div><dt>Final ölçüsü</dt><dd>{trace.prompt.phase_chars?.response ?? "—"} simvol</dd></div>
                    <div><dt>Tool azalma</dt><dd>{trace.prompt.tool_char_reduction_percent ?? 0}%</dd></div>
                    <div><dt>Final azalma</dt><dd>{trace.prompt.response_char_reduction_percent ?? 0}%</dd></div>
                  </dl>
                  <details className="debug-code-block">
                    <summary>Versiyalar və hash-lər</summary>
                    <pre>{JSON.stringify({
                      versions: trace.prompt.versions,
                      phase_hashes: trace.prompt.phase_hashes,
                      token_estimates: trace.prompt.phase_token_estimates,
                    }, null, 2)}</pre>
                  </details>
                </section>
              ) : null}

              <section className="debug-section">
                <div className="debug-section-heading"><Database size={17} /><h3>Məlumat mənbələri</h3></div>
                <div className="debug-source-list">
                  {Object.entries(trace.data_sources).map(([name, source]) => (
                    <div key={name}>
                      <strong>{name}</strong>
                      <span>{source.configured ? "aktiv" : "mövcud deyil"}</span>
                      {typeof source.product_count === "number" ? <small>{source.product_count} məhsul</small> : null}
                      {typeof source.document_count === "number" ? <small>{source.document_count} sənəd</small> : null}
                      {typeof source.chunk_count === "number" ? <small>{source.chunk_count} chunk</small> : null}
                      {typeof source.detail === "string" ? <small>{source.detail}</small> : null}
                    </div>
                  ))}
                </div>
              </section>

              <section className="debug-section">
                <div className="debug-section-heading"><Route size={17} /><h3>Keçilən mərhələlər</h3></div>
                <ol className="debug-timeline">
                  {trace.timeline.map((event, index) => (
                    <li key={`${event.stage}-${index}`}>
                      <span className={`timeline-dot ${event.status}`}><Check size={12} /></span>
                      <div><strong>{STAGE_LABELS[event.stage] || event.stage}</strong><p>{timelineDetail(event)}</p></div>
                    </li>
                  ))}
                </ol>
              </section>

              {productRetrieval ? <RetrievalDetails retrieval={productRetrieval} /> : null}
              {documentRetrieval ? <DocumentRetrievalDetails retrieval={documentRetrieval} /> : null}

              {trace.warnings.length ? (
                <section className="debug-section">
                  <div className="debug-section-heading"><AlertTriangle size={17} /><h3>Xəbərdarlıqlar</h3></div>
                  {trace.warnings.map((warning) => <div className="debug-warning" key={warning.code}>{warning.detail}</div>)}
                </section>
              ) : null}

              <section className="debug-section">
                <div className="debug-section-heading"><Bug size={17} /><h3>Texniki metriklər</h3></div>
                <dl className="debug-kv-grid">
                  {Object.entries(trace.model).map(([key, value]) => (
                    <div key={`model-${key}`}><dt>{`model.${key}`}</dt><dd>{displayValue(value)}</dd></div>
                  ))}
                  {Object.entries(trace.metrics).map(([key, value]) => (
                    <div key={key}><dt>{key}</dt><dd>{displayValue(value)}</dd></div>
                  ))}
                </dl>
                <button className="debug-copy" type="button" onClick={() => void copyRequestId()}>
                  {copiedRequestId ? <Check size={15} /> : <Copy size={15} />}
                  {copiedRequestId ? "Kopyalandı" : "Request ID-ni kopyala"}
                </button>
                <code className="debug-request-id">{trace.request_id}</code>
              </section>
            </>
          ) : null}
        </div>
      </aside>
    </div>
  );
}
