from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CategoryId = Literal[
    "smartphones",
    "tablets",
    "laptops",
    "air_conditioners",
    "televisions",
    "headphones",
]
ColorCode = Literal["black", "white", "blue", "green", "gold", "gray", "silver", "pink"]
SortOption = Literal["relevance", "price_asc", "price_desc", "rating_desc"]
SearchIntent = Literal["lookup", "discover"]
SemanticOperation = Literal["lookup", "discover", "compare"]
SemanticOperator = Literal["eq", "not_eq", "in", "not_in", "lt", "lte", "gt", "gte"]
PredicateStrength = Literal["hard", "preference"]
EntityState = Literal["selected", "superseded"]
IdentifierType = Literal["auto", "product_id", "sku", "model", "model_family"]
MemoryAction = Literal["replace", "merge", "preserve"]
RankingDirection = Literal["maximize", "minimize"]
RankingPriority = Literal["primary", "normal", "inferred"]
RankingOrigin = Literal["explicit", "inferred", "memory"]
ValueProvenance = Literal["current_message", "memory", "catalog_attribute"]
AttributeOperator = Literal["eq", "gte", "lte", "in", "contains_any"]
AttributeField = Literal[
    "active_noise_cancellation",
    "battery_hours",
    "battery_mah",
    "btu",
    "cellular",
    "charging_type",
    "connectivity",
    "coverage_max_m2",
    "coverage_min_m2",
    "cpu_brand",
    "cpu_model",
    "display_size_in",
    "energy_class",
    "form_factor",
    "gpu",
    "hdmi_count",
    "hdr",
    "indoor_unit_dimensions_mm",
    "inverter",
    "main_camera_mp",
    "microphone",
    "modes",
    "network",
    "noise_level_db",
    "operating_system",
    "outdoor_unit_dimensions_mm",
    "panel_type",
    "pen_support",
    "ram_gb",
    "refresh_rate_hz",
    "refrigerant",
    "resolution",
    "screen_size_in",
    "sim_type",
    "smart_tv_os",
    "storage_gb",
    "storage_type",
    "system_type",
    "water_resistance",
    "weight_kg",
    "wifi",
]
TopLevelPreferenceField = Literal[
    "brand",
    "model_family",
    "color_code",
    "min_price",
    "storage_gb",
    "ram_gb",
    "btu",
    "screen_size_in",
    "connectivity",
    "active_noise_cancellation",
]
RequiredFilterField = TopLevelPreferenceField | AttributeField
RequestedField = Literal[
    "name",
    "sku",
    "model",
    "brand",
    "model_family",
    "category_id",
    "price",
    "stock_status",
    "color",
    "warranty_months",
    "rating",
] | AttributeField
MatchStatus = Literal[
    "exact_match",
    "exact_conflict",
    "matching_products",
    "alternatives",
    "clarification_required",
    "not_found",
]


class ProductEntity(BaseModel):
    """A product mention extracted by the model, before catalog resolution."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1, max_length=40)
    raw_text: str = Field(min_length=1, max_length=200)
    state: EntityState = "selected"
    supersedes_entity_id: str | None = Field(default=None, max_length=40)
    evidence_text: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Shortest exact span from the current message that identifies this product mention"
        ),
    )
    identifier_type: IdentifierType = Field(
        default="auto",
        description=(
            "Catalog namespace, not a free-form label: exact product ID, SKU, exact model, "
            "model family, or auto when the namespace is not explicit. model_family is only a "
            "named product line/family; manufacturer brands use auto"
        ),
    )
    context_product_id: str | None = Field(
        default=None,
        max_length=120,
        description="Only a product ID copied from the server-provided session context",
    )
    memory_refs: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Opaque memory IDs copied from server-provided session memory when this entity "
            "continues, revises, or derives meaning from an earlier confirmed entity"
        ),
    )


class SemanticPredicate(BaseModel):
    """A typed catalog predicate. Natural-language interpretation happens upstream."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Catalog field at the same semantic granularity explicitly requested by the user. "
            "brand is only a manufacturer/company name, model_family is a named product line or "
            "family, model is an exact model identifier, and category_id is a broad product type. "
            "Never infer brand merely from knowing who makes a named family/model"
        ),
    )
    operator: SemanticOperator
    value: str | int | float | bool | list[str] | list[int] | list[float] = Field(
        description=(
            "Value supported by evidence_text. Preserve the user's named value; catalog aliases "
            "and canonical values are resolved by the backend"
        )
    )
    strength: PredicateStrength
    unit: str | None = Field(default=None, max_length=30)
    value_provenance: ValueProvenance | None = Field(
        default=None,
        description=(
            "Required for concrete numeric values: current_message for an explicit literal, "
            "memory for a typed inherited predicate, or catalog_attribute for a verified "
            "context product fact"
        ),
    )
    value_source_product_id: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "Server-provided context product ID used only when value_provenance=catalog_attribute"
        ),
    )
    evidence_text: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Shortest exact span from the current message that supports this specific field, "
            "operator, and value; never use the whole message merely to justify an unstated default"
        ),
    )
    memory_refs: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Opaque server-provided memory IDs that ground inherited constraints or catalog "
            "facts. Empty for meaning supported only by the current message"
        ),
    )


class SemanticExpression(BaseModel):
    """Recursive, language-independent selection/filter/ranking expression."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "predicate",
        "all_of",
        "any_of",
        "not",
        "fallback",
        "prefer",
        "entity_ref",
    ] = Field(
        description=(
            "Shape discriminator: predicate uses predicate; all_of/any_of use expressions; "
            "not/prefer use expression; fallback uses primary and secondary; entity_ref uses entity_id"
        )
    )
    predicate: SemanticPredicate | None = Field(
        default=None,
        description="Present only when kind=predicate",
    )
    expressions: list[SemanticExpression] | None = Field(
        default=None,
        max_length=20,
        description="Non-empty child list, present only when kind=all_of or kind=any_of",
    )
    expression: SemanticExpression | None = Field(
        default=None,
        description="Single child, present only when kind=not or kind=prefer",
    )
    primary: SemanticExpression | None = Field(
        default=None,
        description="First ordered branch, present only when kind=fallback",
    )
    secondary: SemanticExpression | None = Field(
        default=None,
        description="Second ordered branch, present only when kind=fallback",
    )
    entity_id: str | None = Field(
        default=None,
        max_length=40,
        description="Referenced entity ID, present only when kind=entity_ref",
    )

    @model_validator(mode="after")
    def validate_shape(self) -> SemanticExpression:
        populated = {
            "predicate": self.predicate is not None,
            "expressions": self.expressions is not None,
            "expression": self.expression is not None,
            "primary": self.primary is not None,
            "secondary": self.secondary is not None,
            "entity_id": self.entity_id is not None,
        }
        required: dict[str, set[str]] = {
            "predicate": {"predicate"},
            "all_of": {"expressions"},
            "any_of": {"expressions"},
            "not": {"expression"},
            "prefer": {"expression"},
            "fallback": {"primary", "secondary"},
            "entity_ref": {"entity_id"},
        }
        expected = required[self.kind]
        actual = {name for name, is_set in populated.items() if is_set}
        if actual != expected:
            raise ValueError(f"{self.kind} expression fields must be exactly: {sorted(expected)}")
        if self.kind in {"all_of", "any_of"} and not self.expressions:
            raise ValueError(f"{self.kind} requires at least one child expression")
        return self

    def depth(self) -> int:
        if self.kind in {"predicate", "entity_ref"}:
            return 1
        if self.kind in {"all_of", "any_of"}:
            return 1 + max(child.depth() for child in self.expressions or [])
        if self.kind in {"not", "prefer"}:
            return 1 + (self.expression.depth() if self.expression else 0)
        return 1 + max(
            self.primary.depth() if self.primary else 0,
            self.secondary.depth() if self.secondary else 0,
        )

    def predicate_count(self) -> int:
        if self.kind == "predicate":
            return 1
        if self.kind in {"all_of", "any_of"}:
            return sum(child.predicate_count() for child in self.expressions or [])
        if self.kind in {"not", "prefer"}:
            return self.expression.predicate_count() if self.expression else 0
        if self.kind == "fallback":
            return (self.primary.predicate_count() if self.primary else 0) + (
                self.secondary.predicate_count() if self.secondary else 0
            )
        return 0


class FactQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "One canonical catalog field explicitly asked about. Do not emit aliases or a second "
            "field for the same fact"
        ),
    )
    operator: SemanticOperator | None = Field(
        default=None,
        description=(
            "Comparison operator when the user asks whether the fact satisfies a concrete "
            "proposition; otherwise null"
        ),
    )
    value: str | int | float | bool | list[str] | list[int] | list[float] | None = Field(
        default=None,
        description=(
            "Compared value explicitly present in the question. It must accompany operator; "
            "otherwise null"
        ),
    )
    unit: str | None = Field(
        default=None,
        max_length=30,
        description="Explicit unit for the compared value, if any",
    )
    value_provenance: ValueProvenance | None = Field(
        default=None,
        description=(
            "Source of an explicit numeric comparison value. Omit when the question asks only "
            "for the field value"
        ),
    )
    value_source_product_id: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "Server-provided context product ID used only when value_provenance=catalog_attribute"
        ),
    )
    evidence_text: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Shortest unchanged current-message span supporting the field and any operator/value"
        ),
    )
    memory_refs: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Only typed fact_question IDs from verified session memory",
    )


class RankingObjective(BaseModel):
    """A value-free directional preference interpreted by the model."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "One numeric catalog field from the supplied field-capability metadata. The field "
            "must be sortable for the relevant product category"
        ),
    )
    direction: RankingDirection
    priority: RankingPriority = "normal"
    origin: RankingOrigin = "explicit"
    evidence_text: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Shortest unchanged current-message span supporting the direction, or a concise "
            "description when origin=inferred"
        ),
    )
    memory_refs: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Only typed ranking_objective IDs from verified session memory",
    )

    @model_validator(mode="after")
    def validate_origin_and_priority(self) -> RankingObjective:
        if self.origin == "inferred" and self.priority != "inferred":
            raise ValueError("inferred ranking objectives require priority=inferred")
        if self.origin != "inferred" and self.priority == "inferred":
            raise ValueError("priority=inferred requires origin=inferred")
        return self


class MemoryRemoval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1, max_length=100)
    evidence_text: str = Field(
        min_length=1,
        max_length=300,
        description="Exact current-message span that supports removing this memory element",
    )


class ProductQueryPlan(BaseModel):
    """Semantic tool contract emitted by the first model round."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=500,
        description="The complete current user product request in its original language",
    )
    operation: SemanticOperation = Field(
        description=(
            "lookup for a concrete item/fact, discover for finding candidates including multiple "
            "allowed selections, compare only for an explicit comparison between entities"
        )
    )
    entities: list[ProductEntity] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Concrete catalog-selection mentions whose identity or relationship matters. Broad "
            "allowed facet values may instead be predicates in filter_expression"
        ),
    )
    selection_expression: SemanticExpression | None = Field(
        default=None,
        description=(
            "Relationships among entities only. When entities are present, every selection branch "
            "is composed only of entity_ref nodes; never place catalog predicates beside entity_ref"
        ),
    )
    filter_expression: SemanticExpression | None = Field(
        default=None,
        description=(
            "Hard catalog constraints shared by the applicable selection branches. Contains "
            "predicate logic, not duplicate entity selection or ranking preferences"
        ),
    )
    preference_expression: SemanticExpression | None = Field(
        default=None,
        description=(
            "Soft ranking preferences only. It must not contain required conditions or duplicate "
            "selection/filter meaning"
        ),
    )
    ranking_objectives: list[RankingObjective] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Value-free directional goals such as maximizing a numeric feature or minimizing "
            "price/weight. Never invent a threshold to represent a qualitative request"
        ),
    )
    fact_questions: list[FactQuestion] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Only catalog facts explicitly requested about an item. A proposition question keeps "
            "its operator/value/unit here and does not become a filter"
        ),
    )
    recommendation_requested: bool = Field(
        default=False,
        description=(
            "True only when the user explicitly asks the assistant to choose or recommend, not for "
            "plain lookup, browsing, or comparison"
        ),
    )
    memory_action: MemoryAction = Field(
        default="replace",
        description=(
            "replace starts a new independent product state; merge revises or continues the "
            "confirmed session memory; preserve answers a read-only fact without changing the "
            "active semantic state"
        ),
    )
    referenced_memory_ids: list[str] = Field(default_factory=list, max_length=20)
    removed_memory_ids: list[str] = Field(default_factory=list, max_length=20)
    memory_removals: list[MemoryRemoval] = Field(default_factory=list, max_length=20)
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True only when the semantic meaning or referent cannot be selected safely. False for a "
            "valid broad/filter-only discovery that merely lacks optional category, budget, or preferences"
        ),
    )
    clarification_question: str | None = Field(default=None, max_length=300)
    sort: SortOption = "relevance"
    limit: int = Field(default=3, ge=1, le=3)
    context_product_ids: list[str] = Field(default_factory=list, max_length=3, exclude=True)
    context_memory: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def validate_plan(self) -> ProductQueryPlan:
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity_id values must be unique")
        known_ids = set(ids)
        for entity in self.entities:
            if entity.supersedes_entity_id and entity.supersedes_entity_id not in known_ids:
                raise ValueError("supersedes_entity_id must refer to an entity in this plan")
            if entity.supersedes_entity_id == entity.entity_id:
                raise ValueError("an entity cannot supersede itself")
        expressions = [
            expression
            for expression in (
                self.selection_expression,
                self.filter_expression,
                self.preference_expression,
            )
            if expression is not None
        ]
        if any(expression.depth() > 6 for expression in expressions):
            raise ValueError("semantic expression depth cannot exceed 6")
        if sum(expression.predicate_count() for expression in expressions) > 20:
            raise ValueError("semantic plan cannot contain more than 20 predicates")
        if sum(item.origin == "inferred" for item in self.ranking_objectives) > 2:
            raise ValueError("semantic plan cannot contain more than two inferred objectives")

        def validate_refs(expression: SemanticExpression) -> None:
            if expression.kind == "entity_ref" and expression.entity_id not in known_ids:
                raise ValueError("entity_ref must refer to an entity in this plan")
            for child in expression.expressions or []:
                validate_refs(child)
            for child in (expression.expression, expression.primary, expression.secondary):
                if child is not None:
                    validate_refs(child)

        for expression in expressions:
            validate_refs(expression)
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("clarification_question is required when needs_clarification is true")
        if len(self.referenced_memory_ids) != len(set(self.referenced_memory_ids)):
            raise ValueError("referenced_memory_ids values must be unique")
        if len(self.removed_memory_ids) != len(set(self.removed_memory_ids)):
            raise ValueError("removed_memory_ids values must be unique")
        removal_evidence_ids = [item.memory_id for item in self.memory_removals]
        if len(removal_evidence_ids) != len(set(removal_evidence_ids)):
            raise ValueError("memory_removals memory_id values must be unique")
        if set(removal_evidence_ids) != set(self.removed_memory_ids):
            raise ValueError(
                "memory_removals must provide current-message evidence for every removed_memory_id"
            )
        return self

NUMERIC_ATTRIBUTE_FIELDS = frozenset(
    {
        "battery_hours",
        "battery_mah",
        "btu",
        "coverage_max_m2",
        "coverage_min_m2",
        "display_size_in",
        "hdmi_count",
        "main_camera_mp",
        "noise_level_db",
        "ram_gb",
        "refresh_rate_hz",
        "screen_size_in",
        "storage_gb",
        "weight_kg",
    }
)
BOOLEAN_ATTRIBUTE_FIELDS = frozenset(
    {
        "active_noise_cancellation",
        "cellular",
        "hdr",
        "inverter",
        "microphone",
        "pen_support",
        "wifi",
    }
)
TEXT_ATTRIBUTE_FIELDS = frozenset(
    {
        "charging_type",
        "connectivity",
        "cpu_brand",
        "cpu_model",
        "energy_class",
        "form_factor",
        "gpu",
        "network",
        "operating_system",
        "panel_type",
        "refrigerant",
        "resolution",
        "sim_type",
        "smart_tv_os",
        "storage_type",
        "system_type",
        "water_resistance",
    }
)
LIST_ATTRIBUTE_FIELDS = frozenset({"modes"})
DIMENSION_ATTRIBUTE_FIELDS = frozenset(
    {"indoor_unit_dimensions_mm", "outdoor_unit_dimensions_mm"}
)


class AttributeFilter(BaseModel):
    """Validated filter for a category-specific Qdrant payload field."""

    model_config = ConfigDict(extra="forbid")

    field: AttributeField = Field(description="Kataloq attributes daxilindəki dəqiq payload field-i")
    operator: AttributeOperator = Field(
        description="Rəqəm üçün eq/gte/lte, mətn üçün eq/in, boolean üçün eq, siyahı üçün contains_any"
    )
    value: str | int | float | bool | list[str] | list[int] = Field(
        description="Field tipinə və operatora uyğun filter dəyəri"
    )

    @model_validator(mode="after")
    def validate_field_operator_and_value(self) -> AttributeFilter:
        field = self.field
        operator = self.operator
        value = self.value
        if field in NUMERIC_ATTRIBUTE_FIELDS:
            if operator not in {"eq", "gte", "lte"}:
                raise ValueError(f"{field} üçün operator eq, gte və ya lte olmalıdır")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} üçün rəqəm dəyəri tələb olunur")
        elif field in BOOLEAN_ATTRIBUTE_FIELDS:
            if operator != "eq" or not isinstance(value, bool):
                raise ValueError(f"{field} üçün eq və boolean dəyəri tələb olunur")
        elif field in TEXT_ATTRIBUTE_FIELDS:
            if operator == "eq" and isinstance(value, str):
                return self
            if operator == "in" and isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                return self
            raise ValueError(f"{field} üçün eq mətn və ya in mətn siyahısı tələb olunur")
        elif field in LIST_ATTRIBUTE_FIELDS:
            if operator != "contains_any" or not isinstance(value, list) or not value:
                raise ValueError(f"{field} üçün contains_any və boş olmayan siyahı tələb olunur")
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"{field} siyahısında yalnız mətn ola bilər")
        elif field in DIMENSION_ATTRIBUTE_FIELDS:
            if operator != "eq" or not isinstance(value, str):
                raise ValueError(f"{field} üçün eq və 'en x hündürlük x dərinlik' mətni tələb olunur")
        return self


class ProductSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    search_intent: SearchIntent = Field(
        default="discover",
        description=(
            "lookup konkret məhsul haqqında məlumat üçündür; discover uyğun məhsul seçimi üçündür"
        ),
    )
    requested_fields: list[RequestedField] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "İstifadəçinin qiymət, stok və ya texniki xüsusiyyət kimi öyrənmək istədiyi sahələr; "
            "bunlar filtr deyil"
        ),
    )
    product_id: str | None = Field(
        default=None,
        max_length=120,
        description="İstifadəçi tam product_id veribsə həmin dəyər",
    )
    sku: str | None = Field(
        default=None,
        max_length=120,
        description="İstifadəçi tam SKU veribsə həmin dəyər",
    )
    model: str | None = Field(
        default=None,
        max_length=120,
        description="İstifadəçinin dəqiq dediyi model; brand adını modelə əlavə etmə",
    )
    category_id: CategoryId | None = None
    brand: str | None = Field(default=None, max_length=80)
    model_family: str | None = Field(default=None, max_length=80)
    color_code: ColorCode | None = None
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    in_stock: bool | None = None
    storage_gb: int | None = Field(default=None, gt=0)
    ram_gb: int | None = Field(default=None, gt=0)
    btu: int | None = Field(default=None, gt=0)
    screen_size_in: float | None = Field(default=None, gt=0)
    connectivity: str | None = Field(default=None, max_length=80)
    active_noise_cancellation: bool | None = None
    attribute_filters: list[AttributeFilter] = Field(
        default_factory=list,
        max_length=20,
        description="Kateqoriyaya məxsus texniki parametrlər üçün type-safe Qdrant filterləri",
    )
    required_filter_fields: list[RequiredFilterField] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "İstifadəçinin 'mütləq' və ya 'yalnız' dediyi filter field-ləri. "
            "Kateqoriya, maksimum büdcə və stok onsuz da həmişə sərtdir."
        ),
    )
    sort: SortOption = "relevance"
    limit: int = Field(default=5, ge=1, le=5)
    semantic_filter_expression: SemanticExpression | None = Field(default=None, exclude=True)
    semantic_preference_expression: SemanticExpression | None = Field(default=None, exclude=True)
    semantic_ranking_objectives: list[RankingObjective] = Field(
        default_factory=list,
        max_length=3,
        exclude=True,
    )
    excluded_product_ids: list[str] = Field(default_factory=list, max_length=3, exclude=True)
    semantic_plan_compiled: bool = Field(default=False, exclude=True)

    @model_validator(mode="after")
    def validate_filters(self) -> ProductSearchArguments:
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError("min_price max_price-dan böyük ola bilməz")
        seen = {(item.field, item.operator) for item in self.attribute_filters}
        if len(seen) != len(self.attribute_filters):
            raise ValueError("Eyni attribute field və operator təkrar verilə bilməz")
        if len(self.required_filter_fields) != len(set(self.required_filter_fields)):
            raise ValueError("required_filter_fields təkrarlanmamalıdır")
        if len(self.requested_fields) != len(set(self.requested_fields)):
            raise ValueError("requested_fields təkrarlanmamalıdır")
        provided_fields = {
            field
            for field in (
                "brand",
                "model_family",
                "color_code",
                "min_price",
                "storage_gb",
                "ram_gb",
                "btu",
                "screen_size_in",
                "connectivity",
                "active_noise_cancellation",
            )
            if getattr(self, field) is not None
        }
        provided_fields.update(item.field for item in self.attribute_filters)
        missing_required = sorted(set(self.required_filter_fields) - provided_fields)
        if missing_required:
            raise ValueError(
                "required_filter_fields dəyəri uyğun filter ilə birlikdə verilməlidir: "
                + ", ".join(missing_required)
            )
        return self

    @property
    def has_exact_identifier(self) -> bool:
        return any((self.product_id, self.sku, self.model))


class ProductSearchItem(BaseModel):
    product_id: str
    sku: str
    name: str
    category_id: str
    category_name: str
    brand: str
    model_family: str
    color_code: str
    color_name: str
    sale_price: float
    currency: str
    stock_status: str
    warranty_months: int
    rating: float
    attributes: dict[str, Any]
    short_description: str
    differences: list[str] = Field(default_factory=list)
    ranking_reasons: list[str] = Field(default_factory=list)


class ProductSearchResult(BaseModel):
    status: Literal["success"] = "success"
    match_status: MatchStatus
    requested_label: str | None = None
    strict_total: int = Field(ge=0)
    total: int
    applied_filters: dict[str, Any]
    relaxed_fields: list[str] = Field(default_factory=list)
    items: list[ProductSearchItem]
    requested_item: ProductSearchItem | None = None
    constraint_conflicts: list[str] = Field(default_factory=list)
    argument_corrections: list[dict[str, Any]] = Field(default_factory=list)
    recommended_product_id: str | None = None
    display_product_ids: list[str] = Field(default_factory=list)
    operation: SemanticOperation = "discover"
    resolved_entities: list[dict[str, Any]] = Field(default_factory=list)
    entity_results: list[dict[str, Any]] = Field(default_factory=list)
    canonical_query_hash: str | None = None
    clarification: dict[str, Any] | None = None
    unavailable_requested_values: list[dict[str, Any]] = Field(default_factory=list)
    ranking_applied: bool = False
    ranking_objectives: list[RankingObjective] = Field(default_factory=list)
    plan_corrections: list[dict[str, Any]] = Field(default_factory=list)


SemanticExpression.model_rebuild()


class DocumentSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=5)


class DocumentSearchChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    heading: str
    text: str
    score: float


class DocumentSearchResult(BaseModel):
    status: Literal["success"] = "success"
    match_status: Literal["found", "not_found"]
    total: int = Field(ge=0)
    min_score: float = Field(ge=0, le=1)
    chunks: list[DocumentSearchChunk]
