#!/bin/bash
# Rebuild the pinned bittensor-drand 1.0.0 C ABI without Python bindings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_TEE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$VALIDATOR_TEE_DIR")"
SOURCE_ARCHIVE="${1:-$REPO_ROOT/.validator-tee-artifacts/bittensor_drand-1.0.0.tar.gz}"
OUTPUT="${2:-$REPO_ROOT/.validator-tee-artifacts/libbittensor_drand_v2.so}"
EXPECTED_HASH_FILE="${3:-$VALIDATOR_TEE_DIR/enclave/libbittensor_drand_v2.sha256}"
BUILDER_IMAGE="rust@sha256:d9c3c6f1264a547d84560e06ffd79ed7a799ce0bff0980b26cf10d29af888377"
CACHE_DIR="${VALIDATOR_DRAND_CARGO_CACHE_DIR:-$HOME/.cache/leadpoet/drand-cabi-v2}"

test -s "$SOURCE_ARCHIVE" || {
  echo "ERROR: pinned bittensor-drand source archive is unavailable" >&2
  exit 1
}
test -s "$EXPECTED_HASH_FILE" || {
  echo "ERROR: pinned drand C ABI output hash is unavailable" >&2
  exit 1
}
EXPECTED_HASH="$(tr -d '[:space:]' < "$EXPECTED_HASH_FILE")"
[[ "$EXPECTED_HASH" =~ ^[0-9a-f]{64}$ ]] || {
  echo "ERROR: pinned drand C ABI output hash is invalid" >&2
  exit 1
}

WORK_DIR="$(mktemp -d "$REPO_ROOT/.drand-cabi-v2.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
cp "$SOURCE_ARCHIVE" "$WORK_DIR/source.tar.gz"
cp "$VALIDATOR_TEE_DIR/enclave/Cargo.drand-cabi-v2.lock" \
  "$WORK_DIR/Cargo.drand-cabi-v2.lock"
mkdir -p "$CACHE_DIR/home" "$CACHE_DIR/target"

cat > "$WORK_DIR/build.sh" <<'BUILD'
#!/bin/bash
set -euo pipefail
mkdir -p /work/source /work/output /work/cargo
tar -xzf /work/source.tar.gz -C /work/source --strip-components=1
cat > /work/source/src/lib.rs <<'RS'
mod drand;
mod ffi;
RS
python3 - <<'PY'
from pathlib import Path

path = Path('/work/source/Cargo.toml')
lines = path.read_text().splitlines()
output = []
in_features = False
for line in lines:
    stripped = line.strip()
    if stripped == '[features]':
        in_features = True
        continue
    if in_features and stripped.startswith('['):
        in_features = False
    if in_features or stripped.startswith('pyo3 ='):
        continue
    output.append(line)
path.write_text('\n'.join(output) + '\n')

PY
cp /work/Cargo.drand-cabi-v2.lock /work/source/Cargo.lock
cd /work/source
export CARGO_HOME=/cargo-cache/home
export CARGO_TARGET_DIR=/cargo-cache/target
export SOURCE_DATE_EPOCH=0
export RUSTFLAGS="--remap-path-prefix=/work/source=/usr/src/bittensor-drand -C strip=symbols -C link-arg=-Wl,--build-id=none"
cargo build --release --locked --no-default-features
cp /cargo-cache/target/release/libbittensor_drand.so /work/output/libbittensor_drand_v2.so
BUILD
chmod 700 "$WORK_DIR/build.sh"

docker run --rm --platform linux/amd64 \
  --network bridge \
  -v "$WORK_DIR:/work" \
  -v "$CACHE_DIR:/cargo-cache" \
  "$BUILDER_IMAGE" \
  bash /work/build.sh

ACTUAL_HASH="$(sha256sum "$WORK_DIR/output/libbittensor_drand_v2.so" | awk '{print $1}')"
if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
  echo "ERROR: drand C ABI hash differs: expected=$EXPECTED_HASH actual=$ACTUAL_HASH" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT")"
install -m 0644 "$WORK_DIR/output/libbittensor_drand_v2.so" "$OUTPUT"
echo "drand_cabi_sha256=$ACTUAL_HASH"
