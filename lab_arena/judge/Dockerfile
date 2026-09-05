# Lab Arena judge image (lab_arena/RUNBOOK.md section 6).
#
# Validators run this image, pinned by digest in every round configuration,
# for scoring assignments. Build and publish it from the repository root:
#
#   bash scripts/build_lab_arena_judge_image.sh <registry>/<repository>
#
# The script prints the digest-pinned reference for LAB_ARENA_SCORER_IMAGE.
#
# The image holds this repository's evaluator packages at their repository
# paths under /model, the shim installed into every Python process through
# sitecustomize, and the scorer entrypoint as the image ENTRYPOINT. A runner
# mounts /input, /output, and the worker socket exactly as for a miner's
# image. The judge has no network and reaches the web only through the
# broker with the organizer's shared provider keys.

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG LEADPOET_BUILD_COMMIT=unknown
LABEL org.opencontainers.image.revision="${LEADPOET_BUILD_COMMIT}" \
      org.opencontainers.image.title="leadpoet-lab-arena-judge"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    PYTHONPATH=/model

WORKDIR /model

# Build tools for the pinned dependency set (the same set the validator image
# installs); the apt lists are removed in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /model/requirements.txt
RUN pip install --no-cache-dir -r /model/requirements.txt

# The evaluator's packages at their repository paths (RUNBOOK section 6): the
# import closure of lab_arena, qualification, gateway, and leadpoet_verifier,
# which reaches the support packages below as well.
COPY lab_arena/ /model/lab_arena/
COPY qualification/ /model/qualification/
COPY gateway/ /model/gateway/
COPY leadpoet_verifier/ /model/leadpoet_verifier/
COPY leadpoet_canonical/ /model/leadpoet_canonical/
COPY leadpoet_observability/ /model/leadpoet_observability/
COPY Leadpoet/ /model/Leadpoet/
COPY validator_models/ /model/validator_models/
COPY validator_tee/ /model/validator_tee/
COPY lab_arena/scorer_entrypoint.py /model/scorer_entrypoint.py

# Every Python process in the image installs the shim before a provider client loads.
RUN printf 'import lab_arena.shim as _lab_arena_shim\n_lab_arena_shim.install()\n' > /model/sitecustomize.py \
    && find /model -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /model \
    && python3 -c "import lab_arena.shim, lab_arena.scorer_entrypoint, qualification.scoring.competition, qualification.scoring.lead_scorer"

USER 65534:65534
ENTRYPOINT ["python3", "/model/scorer_entrypoint.py"]
