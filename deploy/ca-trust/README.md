# AL2023 host certificate distrust

These two public certificates are configuration for the installed AL2023
`p11-kit`/`update-ca-trust` blocklist. Apply them separately on the gateway and
normal-validator hosts after operator review. This is explicit host provisioning;
the gateway and validator restart scripts do not install it.

| File | Certificate DER SHA256 |
| --- | --- |
| `leadpoet-emsign-ecc-g3.pem` | `86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b` |
| `leadpoet-comodo-aaa.pem` | `d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4` |

The bytes come from the [official certifi 2024.08.30 certificate bundle](https://raw.githubusercontent.com/certifi/python-certifi/2024.08.30/certifi/cacert.pem).
That old bundle is an identity source only; it is never installed or trusted.
`SHA256SUMS` pins the two PEM files. The [Red Hat incident bulletin](https://access.redhat.com/security/vulnerabilities/RHSB-2026-012)
and installed `update-ca-trust(8)` describe the vendor blocklist mechanism.

## Apply

Do not apply during an active release. On each host, first record the installed
OS/tool versions, existing TLS/email/code-signing anchor fingerprints, and
unrelated OpenSSL trust attributes. Confirm the vendor's real directory
`/etc/pki/ca-trust/source/blocklist` exists. Both destination names below must
be absent, including dangling symlinks. Stop on any collision; do not replace
an existing file. Retain the exact source commit and the before-state evidence.

From the exact reviewed checkout, verify the source files:

```bash
sha256sum --check deploy/ca-trust/SHA256SUMS
```

Create only the two new files, without overwriting a destination, then verify
their bytes before regeneration:

```bash
sudo -n cp --no-clobber deploy/ca-trust/blocklist/leadpoet-emsign-ecc-g3.pem /etc/pki/ca-trust/source/blocklist/
sudo -n cp --no-clobber deploy/ca-trust/blocklist/leadpoet-comodo-aaa.pem /etc/pki/ca-trust/source/blocklist/
sudo -n cmp deploy/ca-trust/blocklist/leadpoet-emsign-ecc-g3.pem /etc/pki/ca-trust/source/blocklist/leadpoet-emsign-ecc-g3.pem
sudo -n cmp deploy/ca-trust/blocklist/leadpoet-comodo-aaa.pem /etc/pki/ca-trust/source/blocklist/leadpoet-comodo-aaa.pem
sudo -n chmod 0644 /etc/pki/ca-trust/source/blocklist/leadpoet-emsign-ecc-g3.pem /etc/pki/ca-trust/source/blocklist/leadpoet-comodo-aaa.pem
sudo -n /usr/bin/update-ca-trust
```

Run these as sequential checked operations. Stop immediately if any operation
fails; do not continue to regeneration after a failed copy or comparison.
The copies made by `sudo` are root-owned. No package, service, or generated
trust-output file is edited directly. The vendor command ignores destination
arguments and regenerates its fixed extracted outputs.

## Verify and activate

Verify that TLS, email and code-signing anchor sets equal their before-state
minus these two DER fingerprints, and that every unrelated OpenSSL trust
attribute is unchanged. Do not assume both hosts have the same initial count.
The October 22 gateway private-namespace test measured TLS 143 -> 142; AAA was
already absent from TLS. It preserved all other anchors and trust attributes,
passed real OS-default TLS and wrong-hostname controls for `openrouter.ai`,
`api.scrapingdog.com`, `pypi.org` and `files.pythonhosted.org`, and restored the
initial logical trust on rollback without changing the live store.

Use the generated `extracted/openssl/ca-bundle.trust.crt` with explicit CAfile,
`-no-CApath -no-CAstore`, and `openssl verify -purpose` for `sslserver`,
`sslclient`, `smimesign` and `smimeencrypt`. Each root must fail with error 28,
`certificate rejected`. These are root self-verification controls, not forged
certificate chains. Also inspect freshly created Python OS-default contexts
and repeat the four-host TLS/hostname controls before declaring host readiness.

Generic OpenSSL `-purpose any` is supported but is not covered by this vendor
named-purpose distrust export. Both roots remained accepted in that mode in
the isolated test. This procedure does not claim to fix every purpose or
consumer and does not rewrite vendor output to change generic verification.

Fresh OS-default contexts read the regenerated store. Existing SSL contexts
and connection pools can keep their old roots until recreated. Apply before
the next normal validator startup when possible. Any needed process refresh
must use its existing canonical activation path; this procedure does not
restart services. Certifi-backed clients and pip's vendored certifi use their
own bundles and require their separate patched versions. The gateway's cached
Arena proxy context is certifi-backed (`trust_env=False`); OS regeneration does
not change that context. Host processes that use aiohttp's cached OS context
need recreation to observe this change. Container/enclave stores are separate.

## Rollback and retirement

Rollback only files newly created by this application. First compare both
installed files to the exact reviewed source and confirm they remain regular,
root-owned files. Stop if their identity or contents changed. Remove only these
two named blocklist files and run `/usr/bin/update-ca-trust` again. Verify the
initial logical anchor sets and unrelated trust attributes are restored.
Do not restore old generated bundles or remove unrelated blocklist entries.
Existing contexts still require separate activation after a rollback.

Retire this local override only after an approved vendor trust-bundle update
itself rejects both exact certificates for the required named purposes.
First prove this with only these two local entries removed in an isolated
copy, preserve all other anchors, and repeat fresh-context and TLS controls.
Then remove only the matching owned files, regenerate, and verify again.
Until that evidence exists, retain the blocklist across vendor updates.
