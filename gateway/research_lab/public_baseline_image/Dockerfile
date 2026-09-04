FROM node:20-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/baseline-venv
ENV PATH="/opt/baseline-venv/bin:/baseline/experiments/harness_bakeoff/deepline/node_modules/.bin:${PATH}"

ARG BASELINE_REPOSITORY=https://github.com/leadpoet/pydantic-harness.git
ARG BASELINE_BRANCH=main
RUN git clone --depth 1 --branch "${BASELINE_BRANCH}" "${BASELINE_REPOSITORY}" /baseline \
    && pip install --no-cache-dir \
        -r /baseline/requirements.txt \
        -r /baseline/experiments/harness_bakeoff/adapters/requirements-pydantic-ai.txt \
    && npm ci --omit=dev --prefix /baseline/experiments/harness_bakeoff/deepline

WORKDIR /baseline
ENTRYPOINT ["python", "production_runner.py"]
