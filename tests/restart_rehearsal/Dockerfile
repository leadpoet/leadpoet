FROM amazonlinux@sha256:0c74e9fbba754003cfa179fd5cc65be7790d7248443276948704b8a858b298e5

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

COPY harness/ /harness/
RUN chmod 755 \
      /harness/run_inside.sh \
      /harness/contract_adapter.py \
      /harness/weight_readiness_runner.py

ENV HOME=/home/ec2-user
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_INDEX=1

ENTRYPOINT ["/harness/run_inside.sh"]
