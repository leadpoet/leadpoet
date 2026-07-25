FROM amazonlinux@sha256:2d3d063f7fc9fed492a1a7286a88f9852e9b391aa7a6fe91dedbb8381e0bb5b0

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
