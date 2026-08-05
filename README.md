# Sales Bot

Azərbaycan dilində danışan e-commerce assistant üçün ilk vertical slice. Sistem smalltalk-a birbaşa
cavab verir, məhsul sorğularında lokal 300 məhsullu katalog üzərində `product_search` çağırır və
sessiya tarixçəsini PostgreSQL-də saxlayır.

## Tələblər

- Python 3.14+
- Supabase PostgreSQL Session Pooler və ya native PostgreSQL
- OpenRouter API key

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
`OPENROUTER_API_KEY` dəyərini real açarla əvəz edin. `.env` Git-ə daxil edilmir.

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

## Yoxlama

```powershell
python -m ruff check .
python -m pytest -m "not integration"
python -m pytest -m integration
```

Default testlər real OpenRouter çağırışı etmir. Integration testlər yalnız adı `_test` ilə bitən
`TEST_DATABASE_URL` bazasında işləyir.

## Azure embedding və Qdrant baseline

`.env` faylında `CUSTOMER_AZURE_OPENAI_ENDPOINT`, `CUSTOMER_AZURE_OPENAI_API_KEY`,
`AZURE_EMBEDDING_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY` və `QDRANT_COLLECTION_NAME` dəyərlərini
konfiqurasiya edin. Real açarları `.env.example` faylına yazmayın.

300 məhsulu indeksləmək və vəziyyəti yoxlamaq üçün:

```powershell
python -m app.indexing.products index
python -m app.indexing.products status
```

Embedding cache-ni bilərəkdən keçərək yenidən indeksləmək üçün:

```powershell
python -m app.indexing.products index --refresh-embeddings
```

Qdrant semantic retrieval nəticəsini eyni 60 sorğu ilə ölçmək üçün:

```powershell
python -m app.evals.product_semantic --update-baseline
python -m app.evals.product_semantic
```

Semantic runner OpenRouter və Supabase çağırmır. Bu mərhələdə botun runtime `product_search` davranışı
hələ lexical olaraq qalır; Qdrant yalnız offline indeksləmə və eval komandalarında işləyir.

## Product retrieval baseline

Lokal söz və metadata əsaslı `product_search` nəticəsini 30 canonical və 30 challenge sorğusu ilə
yoxlamaq üçün:

```powershell
python -m app.evals.product_retrieval
```

Komanda cari nəticəni `data/evals/baselines/lexical_v1.json` ilə müqayisə edir və fərq olduqda
uğursuz exit code qaytarır. Yalnız axtarış dəyişikliyi nəzərdən keçirildikdən sonra baseline-ı bilərəkdən
yeniləyin:

```powershell
python -m app.evals.product_retrieval --update-baseline
```

Bu eval OpenRouter, Supabase və şəbəkə çağırışı etmir; birbaşa lokal kataloq axtarışını ölçür.

## Mərhələ 1 sərhədi

Qdrant, həqiqi semantic retrieval, sənəd RAG, operator handoff, frontend, streaming və auth bu
mərhələyə daxil deyil. Lokal lexical search sonrakı mərhələdə eyni tool müqaviləsini saxlayan Qdrant
adapteri ilə əvəz ediləcək.
