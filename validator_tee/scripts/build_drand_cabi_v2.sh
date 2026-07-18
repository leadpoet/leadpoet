#!/bin/bash
# Rebuild the pinned bittensor-drand 2.0.0 C ABI without Python bindings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_TEE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$VALIDATOR_TEE_DIR")"
SOURCE_ARCHIVE="${1:-$REPO_ROOT/.validator-tee-artifacts/bittensor_drand-2.0.0.tar.gz}"
OUTPUT="${2:-$REPO_ROOT/.validator-tee-artifacts/libbittensor_drand_v2.so}"
EXPECTED_HASH_FILE="${3:-$VALIDATOR_TEE_DIR/enclave/libbittensor_drand_v2.sha256}"
BUILDER_DOCKERFILE="$VALIDATOR_TEE_DIR/Dockerfile.drand-builder"
BUILDER_IMAGE="validator-drand-builder:v2"
CACHE_DIR="${VALIDATOR_DRAND_CARGO_CACHE_DIR:-$HOME/.cache/leadpoet/drand-cabi-v2}"

test -s "$SOURCE_ARCHIVE" || {
  echo "ERROR: pinned bittensor-drand source archive is unavailable" >&2
  exit 1
}
test -s "$EXPECTED_HASH_FILE" || {
  echo "ERROR: pinned drand C ABI output hash is unavailable" >&2
  exit 1
}
test -s "$BUILDER_DOCKERFILE" || {
  echo "ERROR: pinned drand C ABI builder definition is unavailable" >&2
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
mkdir -p "$CACHE_DIR/home" "$CACHE_DIR/target-al2-glibc226"

docker build --platform linux/amd64 \
  -f "$BUILDER_DOCKERFILE" \
  -t "$BUILDER_IMAGE" \
  "$REPO_ROOT" >/dev/null

cat > "$WORK_DIR/build.sh" <<'BUILD'
#!/bin/bash
set -euo pipefail
mkdir -p /work/source /work/output /work/cargo
tar -xzf /work/source.tar.gz -C /work/source --strip-components=1
cat > /work/source/src/lib.rs <<'RS'
mod constants;
mod drand;
mod epoch_schedule;
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
export CARGO_TARGET_DIR=/cargo-cache/target-al2-glibc226
export SOURCE_DATE_EPOCH=0
export RUSTFLAGS="--remap-path-prefix=/work/source=/usr/src/bittensor-drand -C strip=symbols -C link-arg=-Wl,--build-id=none"
cargo build --release --locked --no-default-features
LIBRARY=/cargo-cache/target-al2-glibc226/release/libbittensor_drand.so

cat > /work/load_test.c <<'C'
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        fprintf(stderr, "drand C ABI load failed: %s\n", dlerror());
        return 1;
    }
    const char *symbols[] = {"cr_generate_commit_v2", "cr_free", "cr_free_str"};
    for (unsigned int i = 0; i < sizeof(symbols) / sizeof(symbols[0]); ++i) {
        if (dlsym(handle, symbols[i]) == NULL) {
            fprintf(stderr, "drand C ABI symbol is unavailable: %s\n", symbols[i]);
            return 1;
        }
    }
    dlclose(handle);
    return 0;
}
C
gcc -O2 -o /work/load_test /work/load_test.c -ldl
/work/load_test "$LIBRARY"
readelf --version-info "$LIBRARY" | python3 -c '
import re
import sys

versions = {
    tuple(map(int, match))
    for match in re.findall(r"GLIBC_(\d+)\.(\d+)", sys.stdin.read())
}
if versions and max(versions) > (2, 26):
    rendered = ".".join(map(str, max(versions)))
    raise SystemExit(
        f"ERROR: drand C ABI requires GLIBC_{rendered}; enclave maximum is GLIBC_2.26"
    )
print(".".join(map(str, max(versions))) if versions else "none")
' > /work/output/max-glibc.txt
cp "$LIBRARY" /work/output/libbittensor_drand_v2.so
BUILD
chmod 700 "$WORK_DIR/build.sh"

docker run --rm --platform linux/amd64 \
  --network bridge \
  -e "HOST_UID=$(id -u)" \
  -e "HOST_GID=$(id -g)" \
  -v "$WORK_DIR:/work" \
  -v "$CACHE_DIR:/cargo-cache" \
  "$BUILDER_IMAGE" \
  bash -c 'trap '\''chown -R "$HOST_UID:$HOST_GID" /work'\'' EXIT; bash /work/build.sh'

echo "drand_cabi_max_glibc=$(cat "$WORK_DIR/output/max-glibc.txt")"

ACTUAL_HASH="$(sha256sum "$WORK_DIR/output/libbittensor_drand_v2.so" | awk '{print $1}')"
if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
  echo "ERROR: drand C ABI hash differs: expected=$EXPECTED_HASH actual=$ACTUAL_HASH" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT")"
install -m 0644 "$WORK_DIR/output/libbittensor_drand_v2.so" "$OUTPUT"
echo "drand_cabi_sha256=$ACTUAL_HASH"
