# AL2023 host certificate distrust

This is one reviewed, public OpenSSL `TRUSTED CERTIFICATE` source for
`/etc/pki/ca-trust/source/leadpoet-emsign-ecc-g3.trusted.pem`. It is explicit
host provisioning, separate from service deployment. Do not put it in
`anchors/` or `blocklist/`, install AAA, or edit generated bundles.

- G3 certificate DER SHA256: `86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b`.
- Source file SHA256: `661798181d9251cd062f8ece4712bfce980138ac3ac085b4a3c77b12065b892b`.
- AUX has no trusted uses. It rejects `anyExtendedKeyUsage`, `serverAuth`,
  `clientAuth`, `emailProtection`, `codeSigning`, `timeStamping`, `OCSPSigning`.

The unchanged X.509 certificate comes from the [official certifi 2024.08.30 bundle](https://raw.githubusercontent.com/certifi/python-certifi/2024.08.30/certifi/cacert.pem)
(bundle SHA256 `94edeb66e91774fcae93a05650914e29096259a5c7e871a1f65d461ab5201b47`).
That old bundle is an identity source only. The [p11-kit 0.24.1 manual](https://raw.githubusercontent.com/p11-glue/p11-kit/0.24.1/doc/manual/p11-kit-trust.xml)
documents TRUSTED CERTIFICATE policy input; installed `update-ca-trust(8)`
documents the neutral source directory. The source can be reproduced from the
public G3 PEM with OpenSSL `x509 -clrtrust -clrreject`, seven `-addreject` options
using the OIDs above in that order, and `-trustout`. No private key is involved.

## Measured policy and limits

The October 22 private-namespace test passed with installed p11-kit 0.24.1 and
OpenSSL 3.2.2, including its `codesign` purpose. The exact test script SHA256 was
`da8454f896392530438193d8b81b3802400e4f8bc938ea1a85a53d45295950f1`.
It measured all ten purposes plus default before/after, with no new acceptance.
G3 was explicitly rejected (error 28) for `sslserver`, `sslclient`,
`nssslserver`, `smimesign`, `smimeencrypt`, `timestampsign`, `codesign`, `any`,
and default. Its `crlsign` and `ocsphelper` compatibility acceptance stayed
unchanged. This is not an all-purpose or all-consumer fix.

AAA DER SHA256 is `d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4`.
It was already excluded from TLS and rejected for client/server TLS and
`any`/default. Its full purpose vector and AUX bytes remain unchanged; its
existing email/code-signing and compatibility acceptance is not remediated here.
Plain-PEM blocklisting widened generic trust for both roots, so that proposal
was rejected. No AAA file is installed by this procedure.

The passing test changed TLS anchors 143 to 142, removed only G3 from TLS,
email and code-signing bundles, and preserved every other anchor/AUX value.
Fresh OS-default TLS and same-endpoint wrong-hostname controls passed for
`openrouter.ai`, `api.scrapingdog.com`, `pypi.org`, `files.pythonhosted.org`.
Logical rollback passed; the outside whole-store manifest stayed
`4d1b064e8292a4d83d3cef4a3dec7bd521f9d3116363c8900e9475840642c29b`.
The manifest covers all three trust input/output trees, aliases, default CA
paths and extraction-script hash, including file contents, links, mode/owner.

## Apply on each host after review

Do not overlap an active release or another trust writer. Use the exact
reviewed commit. First record OS/tool versions and the complete trust manifest,
all TLS/email/code-signing DER sets and OpenSSL certificate/AUX values, and both
root verification vectors. Keep that evidence outside the trust trees. Run the
reviewed private-namespace test against this host's current inputs first; stop
if it differs from the reviewed policy or cannot prove isolation, retained
anchors, TLS controls and rollback. The count/hash above is gateway evidence,
not an assumed baseline for the validator or a later vendor update.

Confirm `/etc/pki/ca-trust/source` is the real root-owned directory. The one
destination must be absent, including dangling symlinks. Verify the installed
`/usr/bin/update-ca-trust` SHA256 is
`d321db9aa915ea4fc54d796badc869c3654ef0ce35dc8e35a4e8f59f0f62fa25`.
Stop for review if it differs. This pinned vendor script ignores arguments and
writes only its fixed extracted outputs. Verify the source checksum:

```bash
sha256sum --check deploy/ca-trust/SHA256SUMS
sha256sum /usr/bin/update-ca-trust
```

Before installation, create a new root-owned mode-0700 backup directory and
record its exact path. Copy the complete three trees, then verify the backup
against the before manifest:

```bash
CA_BACKUP=$(sudo -n mktemp -d /var/tmp/leadpoet-ca-backup.XXXXXXXX)
sudo -n cp -a --parents /etc/pki/ca-trust /usr/share/pki/ca-trust-source /usr/share/pki/ca-trust-legacy "$CA_BACKUP"
```

The backup is evidence and recovery input; do not overwrite live vendor trees
from it during ordinary rollback.

Run the following as sequential checked operations. Stop on any failure.
The shell's noclobber mode refuses an existing destination; do not replace it.

```bash
sudo -n /bin/sh -eu -c 'set -C; umask 022; cat "$1" > "$2"; chown 0:0 "$2"; chmod 0644 "$2"' -- deploy/ca-trust/source/leadpoet-emsign-ecc-g3.trusted.pem /etc/pki/ca-trust/source/leadpoet-emsign-ecc-g3.trusted.pem
sudo -n cmp deploy/ca-trust/source/leadpoet-emsign-ecc-g3.trusted.pem /etc/pki/ca-trust/source/leadpoet-emsign-ecc-g3.trusted.pem
sudo -n /usr/bin/update-ca-trust
```

No package or service action is part of this procedure. After extraction,
require the measured source SHA256, root ownership and mode 0644. Require
TLS/email/code-signing DER sets equal baseline minus G3 and all unrelated AUX
bytes unchanged, including AAA. Fresh Python OS-default roots must equal the
candidate TLS set. Repeat both full verification vectors and reject any new
acceptance; require G3 error 28 for the nine modes listed above. Use explicit
`-CAfile /etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt`,
`-no-CApath -no-CAstore`, and each `-purpose` (omit it for default). These are
root self-verification controls, not forged-chain tests. Repeat the four fresh
OS-default TLS/hostname controls before declaring the host ready.

## Activation, rollback and retirement

Fresh OS-default contexts read the regenerated store. Existing contexts/pools
can retain old roots until recreated. Apply before the next planned validator
startup when possible. Any required process refresh uses its existing
canonical activation path; this procedure does not restart services. The
cached gateway Arena proxy and provider HTTPX transport use patched certifi
(`trust_env=False`), separate from OS trust. aiohttp can cache an OS context at
import. Container/enclave stores and custom CA overrides are separate.

For rollback, confirm the one destination is still a regular, root-owned file
with the exact reviewed source SHA256 and mode. Stop on any mismatch. Remove
only `/etc/pki/ca-trust/source/leadpoet-emsign-ecc-g3.trusted.pem`, then run the
pinned `/usr/bin/update-ca-trust`. Require the original DER sets, all AUX values,
both full verification vectors and fresh OS contexts to match baseline.
Do not restore old generated bundles or remove unrelated source entries.
Rollback restores baseline G3 trust, so keep the security consequence explicit.
Existing contexts may still need separate activation after rollback.

Retire this local override only after an approved vendor bundle, tested without
this exact local source in an isolated copy, rejects G3 for the same nine modes,
preserves AAA/unrelated policy, introduces no acceptance in any measured mode,
and passes fresh-context and four-host TLS controls. Then remove only the
matching local file, regenerate and verify. Keep it across vendor updates until
those exit criteria pass. Do not delete the recorded backup as part of apply.
