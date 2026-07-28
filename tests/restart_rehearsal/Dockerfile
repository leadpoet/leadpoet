ARG REHEARSAL_BASE_IMAGE
FROM ${REHEARSAL_BASE_IMAGE}

RUN dnf install -y \
      diffutils \
      findutils \
      gawk \
      gcc \
      gcc-c++ \
      git \
      gzip \
      iproute \
      jq \
      libffi-devel \
      make \
      openssl-devel \
      procps-ng \
      python3.11 \
      python3.11-devel \
      python3.11-pip \
      rsync \
      shadow-utils \
      sudo \
      tar \
      util-linux \
      which \
    && dnf clean all

COPY requirements.txt /opt/leadpoet/requirements.txt
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.11 -m pip install --no-cache-dir \
      --requirement /opt/leadpoet/requirements.txt \
      minio \
      awscli

COPY requirements-scoring-py39.in /opt/leadpoet/requirements-scoring-py39.in
COPY requirements-scoring-py39.lock /opt/leadpoet/requirements-scoring-py39.lock
RUN mkdir -p /opt/leadpoet/scoring-wheelhouse \
    && python3.11 -m pip download \
      --require-hashes \
      --no-deps \
      --only-binary=:all: \
      --platform manylinux2014_x86_64 \
      --python-version 3.9 \
      --implementation cp \
      --abi cp39 \
      --dest /opt/leadpoet/scoring-wheelhouse \
      --requirement /opt/leadpoet/requirements-scoring-py39.lock

COPY runsc-runtime.lock.json /opt/leadpoet/runsc-runtime.lock.json
COPY runtime-artifacts-v2.lock.json /opt/leadpoet/runtime-artifacts-v2.lock.json
COPY harness/prepare_external_artifacts.py /opt/leadpoet/prepare_external_artifacts.py
RUN python3.11 /opt/leadpoet/prepare_external_artifacts.py \
      --runsc-lock /opt/leadpoet/runsc-runtime.lock.json \
      --validator-runtime-lock /opt/leadpoet/runtime-artifacts-v2.lock.json \
      --output-root /opt/leadpoet/external-artifacts

COPY harness/ /harness/
RUN chmod 755 \
      /harness/run_inside.sh \
      /harness/contract_adapter.py \
      /harness/gateway_boundary_service.py \
      /harness/gateway_enclave_service.py \
      /harness/join_evidence.py \
      /harness/local_services.py \
      /harness/prepare_external_artifacts.py \
      /harness/prepare_host_fixtures.py \
      /harness/production_workflow_runner.py \
      /harness/tls_connect_proxy_service.py \
      /harness/validator_enclave_service.py

ENV HOME=/home/ec2-user
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_INDEX=1

ENTRYPOINT ["/harness/run_inside.sh"]
