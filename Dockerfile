FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY data/fixtures ./data/fixtures

RUN python -m pip install --no-cache-dir '.[dev]'

CMD ["pytest"]
