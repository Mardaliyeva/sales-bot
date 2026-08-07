# Sales Bot

Azərbaycan dilində danışan e-commerce assistant üçün ilk vertical slice. Sistem smalltalk-a birbaşa
cavab verir, məhsul sorğularında Azure embedding və Qdrant üzərindən `product_search` çağırır, tam
məhsul məlumatını 300 məhsullu JSON kataloqdan götürür və sessiya tarixçəsini PostgreSQL-də saxlayır.

## Tələblər

- Python 3.14+
- Node.js 20.9+
- Supabase PostgreSQL Session Pooler və ya native PostgreSQL
- Mətn və embedding deployment-ləri olan Azure OpenAI resursu

Docker tələb olunmur.

## Lokal quraşdırma

PowerShell-də virtual environment və dependency-ləri hazırlayın:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
Copy-Item .env.example .env
```

`.env.example` Supabase Shared Pooler-in session mode URL-ni ehtiva edir. `.env` daxilində
`YOUR_PASSWORD_URL_ENCODED` hissəsini Supabase database parolunun URL-encoded forması ilə, həmçinin
`CUSTOMER_AZURE_OPENAI_ENDPOINT` və `CUSTOMER_AZURE_OPENAI_API_KEY` dəyərlərini real Azure
məlumatları ilə əvəz edin. `AZURE_TEXT_MODEL` və `AZURE_EMBEDDING_MODEL` sahələrinə Azure portalında
yaratdığınız deployment adlarını yazın. `.env` Git-ə daxil edilmir.

`TEST_DATABASE_URL` yalnız ayrıca test database olduqda açılmalıdır. Integration testlərini əsas
Supabase `postgres` database-inə yönəltməyin. Supabase bağlantı rejimləri barədə rəsmi məlumat:
https://supabase.com/docs/guides/database/connecting-to-postgres

Migration və server:

```powershell
python -m alembic upgrade head
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## API

Sessiya yaradın:

```http
POST /v1/sessions
Content-Type: application/json

{}
```

Mesaj göndərin:

```http
POST /v1/chat
Content-Type: application/json

{
  "session_id": "SESSION_UUID",
  "message": "Qara rəngdə 128 GB iPhone göstər"
}
```

Health endpoint-ləri: `GET /health/live` və `GET /health/ready`.

## Frontend

Backend `127.0.0.1:8000` ünvanında işləyərkən ayrıca PowerShell pəncərəsində frontend-i başladın:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm.cmd install
npm.cmd run dev
```

Brauzerdə `http://127.0.0.1:3000` ünvanını açın. Frontend `/backend/*` sorğularını
`SALES_BOT_API_URL` ilə göstərilən FastAPI serverinə proxy edir. Söhbət tarixçəsi həmin brauzerin
`localStorage` yaddaşında saxlanılır.

Development debug panelini aktivləşdirmək üçün backend `.env` faylına
`DEBUG_PANEL_ENABLED=true`, frontend `.env.local` faylına isə
`NEXT_PUBLIC_DEBUG_PANEL=true` əlavə edin. Panel yalnız `APP_ENV=development`
olduqda işləyir. Hər assistant cavabındakı `Debug` düyməsi model mərhələlərini,
tool arqumentlərini, Qdrant exact/semantic namizədlərini və JSON hydration nəticəsini göstərir;
API açarları, system prompt və gizli reasoning göstərilmir.

Frontend yoxlamaları:

```powershell
npm.cmd test
npm.cmd run lint
npm.cmd run build
```

## Yoxlama

```powershell
python -m ruff check .
python -m pytest -m "not integration"
python -m pytest -m integration
```

Default testlər real Azure və Qdrant çağırışı etmir. Integration testlər yalnız adı
`_test` ilə bitən `TEST_DATABASE_URL` bazasında işləyir.

## Azure embedding və Qdrant baseline

`.env` faylında `CUSTOMER_AZURE_OPENAI_ENDPOINT`, `CUSTOMER_AZURE_OPENAI_API_KEY`,
`AZURE_EMBEDDING_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY` və `QDRANT_COLLECTION_NAME` dəyərlərini
konfiqurasiya edin. `ALTERNATIVE_MIN_SCORE` cari dataset və embedding eval nəticəsinə uyğun
alternativ relevance həddidir; deployment və ya embedding text versiyası dəyişəndə yenidən
kalibrasiya edilməlidir. Real açarları `.env.example` faylına yazmayın.

300 məhsulu indeksləmək və vəziyyəti yoxlamaq üçün:

```powershell
python -m app.indexing.products index
python -m app.indexing.products status
```

Embedding cache-ni bilərəkdən keçərək yenidən indeksləmək üçün:

```powershell
python -m app.indexing.products index --refresh-embeddings
```

Qdrant semantic retrieval nəticəsini 30 canonical və 37 challenge sorğusu ilə ölçmək üçün:

```powershell
python -m app.evals.product_semantic --update-baseline
python -m app.evals.product_semantic
```

Semantic runner Azure mətn modelini və Supabase-i çağırmır; query embedding üçün Azure embedding
deployment-indən istifadə edir.

## Qdrant-only product search runtime

`product_search` söz uyğunluğu və lokal score hesablamır. Azure modeli struktur filterləri çıxarır;
SKU, `product_id` və model Qdrant payload field-lərində exact yoxlanılır, digər sorğular isə yalnız
`name + description` embedding-i ilə Qdrant-da semantic axtarılır. Bütün kateqoriya parametrləri
Qdrant payload field-ləri kimi filterlənə bilir.

Exact məhsul və ya bütün şərtlərə uyğun nəticə tapılmadıqda backend `match_status` vasitəsilə bunu
normal nəticədən ayırır və maksimum üç alternativ seçir. Kateqoriya, maksimum büdcə, stok və
`required_filter_fields` daxilindəki tələblər yumşalmır; rəng, texniki üstünlüklər və brand/model
ailəsi mərhələli yumşaldılır. Semantic score exact uyğunluq sayılmır, yalnız relevance həddini keçən
alternativləri sıralayır.

Qdrant yalnız uyğun `product_id`-ləri və sıralama məlumatını qaytarır. İstifadəçiyə göstərilən tam
məhsul JSON kataloqdan hydrate edilir. Qiymət və reytinq sıralaması ilk 50 semantic namizədə tətbiq
olunur. Azure embedding və ya Qdrant əlçatan deyilsə söz əsaslı fallback edilmir; tool açıq
`product_search_unavailable` xətası qaytarır.

Köhnə `data/evals/baselines/lexical_v1.json` yalnız tarixi snapshot-dır və runtime/eval tərəfindən
istifadə edilmir. Aktiv baseline `data/evals/baselines/semantic_qdrant_v2.json` faylıdır.

## Hazırkı sərhəd

Sənəd RAG, operator handoff, streaming və auth hazırkı mərhələyə daxil deyil. Frontend məhsul
cavablarını məhsul kartları ilə göstərə bilir. Public chat müqaviləsinin `presentation` hissəsinə
alternativ statusu və kart fərqləri additive əlavə olunub; daxili `product_search` schema-sı exact
identifier, sərt filter və uyğunluq statusları ilə genişləndirilib.
