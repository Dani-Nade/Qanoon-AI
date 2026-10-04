# Initial Setup

```powershell
Set-Location G:\Qanoon-AI
py -3.12 --version
py -3.12 -m venv .\backend\.venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\backend\requirements-dev.txt
```

# PostgreSQL

```powershell
Set-Location G:\Qanoon-AI
docker compose -f .\infra\docker-compose.yml up -d postgres
docker compose -f .\infra\docker-compose.yml ps postgres
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'pg_trgm')
ORDER BY extname;
```

# Build Index

```powershell
Set-Location G:\Qanoon-AI
$env:QANOON_DATABASE_URL="postgresql://qanoon:qanoon@127.0.0.1:5432/qanoon"
$env:QANOON_RETRIEVAL_MODE="postgres"
.\backend\.venv\Scripts\Activate.ps1
Set-Location .\backend
python -m qanoon_ai.cli build-system --limit 25
python -m qanoon_ai.cli system-status
python -m qanoon_ai.cli build-system
python -m qanoon_ai.cli system-status
```

# Start Backend

```powershell
Set-Location G:\Qanoon-AI
$env:QANOON_DATABASE_URL="postgresql://qanoon:qanoon@127.0.0.1:5432/qanoon"
$env:QANOON_RETRIEVAL_MODE="postgres"
.\backend\.venv\Scripts\Activate.ps1
Set-Location .\backend
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

# Start Frontend

```powershell
Set-Location G:\Qanoon-AI\frontend
npm ci
npm run dev
```

# Daily Start

```powershell
Set-Location G:\Qanoon-AI
docker compose -f .\infra\docker-compose.yml up -d postgres
```

```powershell
Set-Location G:\Qanoon-AI
$env:QANOON_DATABASE_URL="postgresql://qanoon:qanoon@127.0.0.1:5432/qanoon"
$env:QANOON_RETRIEVAL_MODE="postgres"
.\backend\.venv\Scripts\Activate.ps1
Set-Location .\backend
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

```powershell
Set-Location G:\Qanoon-AI\frontend
npm run dev
```

# Stop PostgreSQL

```powershell
Set-Location G:\Qanoon-AI
docker compose -f .\infra\docker-compose.yml stop postgres
```
