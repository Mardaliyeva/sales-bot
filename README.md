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

## Mərhələ 1 sərhədi

Qdrant, həqiqi semantic retrieval, sənəd RAG, operator handoff, frontend, streaming və auth bu
mərhələyə daxil deyil. Lokal lexical search sonrakı mərhələdə eyni tool müqaviləsini saxlayan Qdrant
adapteri ilə əvəz ediləcək.
