FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY cpp ./cpp
COPY CMakeLists.txt ./
COPY configs ./configs
COPY data/fixtures ./data/fixtures

RUN python -m pip install --no-cache-dir '.[dev]'

CMD ["pytest"]
