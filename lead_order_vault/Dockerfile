# Lead-order key vault. Built reproducibly and run on Phala Cloud (Intel TDX).
# Only envelope.py, tools.py and vault.py go in the image; the verifier and CLI stay out.
FROM python:3.12.15-slim-bookworm@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY lead_order_vault/requirements.txt /tmp/requirements.txt
RUN pip install --no-compile --require-hashes --only-binary=:all: --no-deps -r /tmp/requirements.txt && rm /tmp/requirements.txt
WORKDIR /app
COPY lead_order_vault/__init__.py lead_order_vault/envelope.py lead_order_vault/tools.py lead_order_vault/vault.py /app/lead_order_vault/
EXPOSE 8080
# -E -s: ignore PYTHON* env vars and user site-packages. (-I would also drop
# /app from the import path.) Runs as root: the dstack socket is root-owned.
CMD ["python", "-E", "-s", "-m", "lead_order_vault.vault"]
