# Attested release isolation

The production gateway and validator are not release builders. The legacy
`gateway` and `validator` evidence domain names remain stable schema identifiers.
They do not identify hosts or grant access. Each domain still needs three clean
builds of each physical role, and all six measurements must agree.

The workflow keeps source admission, independent builds, verification and
publication in separate jobs. Only the fresh GitHub-hosted publisher can request
OIDC, and only after protected-environment approval. Builders receive no stored
secrets or OIDC permission. Source configuration alone does not prove that the
external runner, environment and IAM controls are configured.

To reproduce an existing serving commit, dispatch this workflow on `main` with
`reproduce_commit` set to its full lowercase 40-character hash. It must be an
ancestor of `main`. An optional `expected_pcr0` checks the coordinator measurement
across both domains and all repetitions. Every reproduction skips publication,
including a reproduction of the current main commit. Inputs enter shell scripts
through environment values and are validated before use. Workflow code and
toolchain selection always come from the exact current main commit.

## Release admission

Release publication must remain unavailable until all of these checks pass:

1. A repository administrator configures the `attested-release` environment.
   It permits only `main` and requires manual approval from a configured reviewer.
   The initiating user may approve when listed as a reviewer. Administrator
   bypass remains disabled.
2. Builder A runs on a fresh GitHub-hosted Linux x64 runner. Prove its actual
   memory and free disk are sufficient with the complete build below. Standard
   runner specifications are not a completed build or a peak-usage measurement.
3. Builder B runs on a fresh RunsOn Linux x64 runner in the existing dedicated
   `leadpoet-ci` account `899952150628`, separate from production `493765492819`.
   Its release runner label is
   `runs-on=<run-id>-attested-v2-b-<attempt>/runner=attested-v2-isolated/spot=false/retry=false`.
   The repository's `.github/runs-on.yml` defines the release-only runner with
   no cache extras, persistent disk, SSH or inherited runner/image hooks.
   It uses no `pool=` selector. Runtime guards reject warm-pool/sticky-disk/EFS
   markers, external cache brokers and Docker registry mirrors before a build.
   Check stack mandatory extras and the inherited `ubuntu24-full-x64` image
   definition separately; local configuration does not prove their absence.
   Check the installation, main-workflow restriction, repository access and
   network/IAM boundaries. The existing leadpoet-site installation does not
   prove access for this repository. The build helper checks its AWS account
   before historical source runs, then clears inherited AWS credential settings.
   Do not substitute either production host or a GitHub runner for Builder B.
4. Both builders use Nitro CLI 1.4.2 and identical approved x64 bootstrap blobs.
   Verify the source/package identity and file hashes before execution. The
   version string alone does not identify the kernel, init, or command line.
5. A separate fresh GitHub-hosted publisher uses a short-lived GitHub OIDC role.
   Its trust permits only this repository's approved release workflow on `main`
   and the protected environment. Environment-only default OIDC subjects omit
   the branch. Enforce the branch in environment rules and bind the workflow/ref
   in a reviewed custom subject before enabling publication.
6. The role can read and create only
   `s3://leadpoet-attested-v2-artifacts-493765492819/attested-v2/releases/*`.
   Its only required object actions are `s3:GetObject`, `s3:PutObject`, and
   `s3:PutObjectRetention` for the existing COMPLIANCE retention headers.
   It has no production host, secret, IAM, deletion or bucket-policy authority.
   Check bucket Object Lock and policy requirements before granting access.
7. Remove the production hosts' corresponding publication authority after the
   replacement path is proved. An inactive runner service alone is not IAM
   isolation.

Do not enable publication through a repository variable alone. The external
environment, OIDC trust, runner restrictions and actual reproduction must be
verified first. Do not add a production-host fallback.

## Reproduction inputs and command

Record the exact clean commit, Docker engine/BuildKit versions and architecture,
Nitro CLI version/package or source revision, executable SHA-256, and every file
under the selected `NITRO_CLI_BLOBS` directory. Nitro CLI 1.4.2's upstream tag
currently resolves to `6e8512e4d9e1ba82e0e5c4c9d9d0b3987724f217`.
The selected approved bootstrap blob hashes must match production evidence.
Do not assume that rebuilding the CLI produces the same executable hash.
The October 28 checks found identical CLI/blob hashes on both production hosts.
They are pinned in `.github/scripts/attested-nitro-toolchain.lock.json`.
The installed `init`, `linuxkit` and `nsm.ko` differ from the upstream 1.4.2 Git
tag. Do not replace them with the upstream tag's files during reproduction.
Run `.github/scripts/verify_attested_nitro_toolchain.py --nitro-cli PATH --blobs DIR`
before any build; it checks the binary and bootstrap hashes before executing
the pinned CLI version check.
The acceptance signer public-key hash remains
`sha256:cdc20e17d9ecee5da04b30604ac72bc015326640ab98501646551f075d60ef36`.

The Python build orchestrator imports only the standard library. It needs
Linux x64, Git, Python 3, pip, rsync, Docker/BuildKit, flock, and the Nitro CLI.
The enclave image base is digest pinned in `gateway/tee/Dockerfile.enclave`.
The four enclave wheels are hash pinned in
`gateway/tee/requirements-enclave-py312.lock`; the preparation script downloads
and verifies them. A publisher additionally needs boto3. No production secret
or full production Python environment is needed to reproduce the image.

`.github/scripts/install_attested_nitro_toolchain.py` downloads the two official
Amazon Linux RPMs at the pinned immutable CDN URLs. It verifies both package
hashes, extracts only the fixed CLI/bootstrap members through `bsdtar`, verifies
their hashes, and checks the CLI version. It does not install the RPM, execute
package hooks, load a driver, start an allocator or change host CPU/RAM settings.
The workflow installs only `libarchive-tools` and `rsync` as CI prerequisites.
Nitro build artifacts, logs and sockets use writable runner-temporary paths.

On each approved isolated builder, in an exact clean checkout:

```bash
export GATEWAY_V2_BUILD_REVISION="$(git rev-parse HEAD)"
export GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$RUNNER_TEMP/offline-artifacts"
export GATEWAY_V2_BUILD_EVIDENCE_DIR="$RUNNER_TEMP/gateway-evidence"
export GATEWAY_V2_BUILD_WORK_ROOT="$RUNNER_TEMP/gateway-build-work"
export GATEWAY_V2_PCR0_CACHE_FILE="$RUNNER_TEMP/gateway-pcr0-cache.json"
export LEADPOET_DOCKER_OPERATION_LOCK_FILE="$RUNNER_TEMP/docker-operation.lock"
# Set PATH and NITRO_CLI_BLOBS to the verified toolchain explicitly.
# Do not source Nitro env.sh: it attempts to load a kernel driver.
# Assign the stable schema domain and a distinct same-run builder identity.
export GATEWAY_V2_BUILDER_DOMAIN=gateway
export GATEWAY_V2_BUILDER_ID="gateway-parent-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
bash gateway/tee/prepare_offline_enclave_dependencies.sh
bash gateway/tee/build_release_evidence_v2.sh
```

Builder B uses domain `validator` and identity
`validator-parent-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}`. These values are
publisher expectations, not claims trusted from the downloaded JSON.

The builder removes images and EIFs after each repetition. Image normalization
also creates Docker save, extracted-layer, and normalized archives. Measure
peak disk and memory through the complete three-build run; image size alone
does not bound temporary disk use. Never run production reclamation or memory
cleanup scripts on an isolated builder as a substitute for sizing it.

For the live-source reproduction, use the source identity proved by the serving
EIF/attestation, not merely the current parent checkout. Compare PCR0 and the
deterministic build identity. A raw EIF hash can differ because Nitro CLI adds
build-time metadata outside PCR0. Do not restart or replace the serving enclave
to make its measurement match the reproduction.

## Untrusted artifact boundary

Download the two exact same-run/attempt artifact names into separate directories;
never use a wildcard merge. Check out trusted exact `main` with persisted Git
credentials disabled. Run `.github/scripts/require_attested_release_source.sh`
before processing artifacts and again before credentials or publication.

Run `.github/scripts/verify_attested_release_evidence.py` with the expected full
commit and distinct builder identities before manifest assembly or AWS OIDC.
It accepts exactly one bounded JSON file per domain, checks roles/ordinals and
every build's expected identity, and runs the existing full consensus gate.
It rejects stale commits, mixed domains, extra files, symlinks and duplicate JSON
keys. Never run downloaded scripts or use downloaded workflow/manifest code.

After assembly, verify the channel's expected commit with the existing
`gateway.tee.release_channel_v2 --verify ... --expected-commit ...` command.
Only then obtain the narrow OIDC role and publish with existing COMPLIANCE
retention and conditional creation. Read back the exact channel and confirm the
published commit and measured roles. Artifact JSON and a green workflow do not
prove that either builder was isolated; retain external runner/account evidence.

References: [AWS build-enclave](https://docs.aws.amazon.com/enclaves/latest/user/cmd-nitro-build-enclave.html),
[Nitro CLI 1.4.2 source](https://github.com/aws/aws-nitro-enclaves-cli/tree/6e8512e4d9e1ba82e0e5c4c9d9d0b3987724f217),
[GitHub runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[GitHub OIDC in AWS](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws),
[OIDC subject customization](https://docs.github.com/en/actions/reference/security/oidc),
[RunsOn runner configuration](https://runs-on.com/docs/runners/labels/),
[RunsOn feature markers](https://runs-on.com/docs/runners/environment/).
