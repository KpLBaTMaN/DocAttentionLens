FROM python:3.12.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock setuptools==80.9.0
COPY library/pyproject.toml /app/library/pyproject.toml
COPY library/src /app/library/src
COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/src /app/backend/src
RUN pip install --no-cache-dir --no-build-isolation --no-deps /app/library /app/backend \
    && useradd --create-home --uid 10001 docattentionlens \
    && mkdir /artifacts /model-cache && chown docattentionlens:docattentionlens /artifacts /model-cache
USER docattentionlens
CMD ["uvicorn", "docattentionlens_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
