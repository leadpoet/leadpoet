# Production access through SSM

The October 27 cutover is verified for the three hosts below. AWS inventory
confirmed one attached instance and one security group per interface. Public
TCP 22 is removed from all three groups; public TCP 3000 is removed from
`subnet71-security`. Ports 80, 443, gateway 8000, and all egress rules are unchanged.
Fresh SSH over SSM, sudo, exact binary stdin, SCP, session termination/reconnect,
and restricted emergency SSH were tested. The canonical paired restart and
exact-owner recovery completed through SSM. Dashboard HTTPS and Arena published
and open-round state remained intact. Recheck live inventory before later changes.

Use `leadpoet-lab` for routine operations in account `493765492819`, region
`us-east-1`. Its SSO role is `LabOperator`, which cannot change IAM. For an
explicitly authorized administrative change, use the existing `leadpoet-admin`
profile with the Identity Center `pranav-admin` authority. Reuse valid cached
temporary credentials. Start a new session only when no valid cached credentials
remain. Never use root recovery. Check STS identity before each write.

The validator credential KMS guard no longer reads AWS credentials from
`gateway.env`. Run it from the checkout with temporary SSO authority:

```bash
python3 -m scripts.configure_lab_arena_production --validator-credential-kms-guard --check --allowed-account 493765492819
```

The default profile is explicitly `leadpoet-lab`. It can inspect this guard but
cannot change IAM. For an explicitly authorized policy change, select the
existing administrative authority with `--aws-profile leadpoet-admin`, `--apply`,
and `LEADPOET_LAB_ARENA_PRODUCTION_APPLY=1`. Reuse its valid cached credentials.
An already-correct policy causes no IAM write. Root authentication and AWS CLI
credential export are not supported by this helper.

For a host with an appropriately permitted EC2 instance role, select
`--aws-instance-role`. It accepts only the instance metadata credential provider.
Instance-role mode applies only to this guard. The helper rejects inherited AWS authority
overrides, long-lived credentials, IAM user callers and other accounts. STS and
IAM clients ignore configured endpoint URLs. Root callers and the retired root
profile are rejected. No role permissions change.
Gateway, Arena and canonical restart authentication continue to use their
existing instance roles. Keep obsolete static keys out of both the cache and its
authoritative Secrets Manager document.

Arena environment configuration now runs locally with the named temporary
`leadpoet-lab` operator profile. `--aws-profile` can select another approved
temporary operator profile. Select `leadpoet-admin` only for an explicitly
authorized administrative change.
The two fixed environment documents and scoped service key stay in process
memory. They do not pass through SSH, command arguments or temporary files.
EC2 and container credential providers are rejected for secret configuration.
Only `--prepare-runner` still uses SSH and needs the private key and host STS
check; the validator IAM guard keeps its separate authority rules above.

For example, inspect a scoped configuration without changing it:

```bash
python3 -m scripts.configure_lab_arena_production --openrouter-max-concurrency 4 --check --allowed-account 493765492819 --aws-profile leadpoet-lab
```

`--apply` still requires `LEADPOET_LAB_ARENA_PRODUCTION_APPLY=1`. Full gateway and
validator configuration checks both documents before its first write, rechecks
the version and document immediately before each write, and verifies readback.
A partial-apply error reports how many targets completed; inspect version stages
before retrying. Secrets Manager has no atomic compare-and-swap operation, so
these checks do not make concurrent operator writes transactional.

The live `leadpoet-lab-operator` policy's own-session condition was corrected
from the literal `${!aws:userid}` to `${aws:userid}`. All other permissions stayed
unchanged. AWS simulation covered both attached principals, including the IAM
user's permissions boundary: own-session termination/resume are allowed;
other-owner and other-account cases are denied. Live termination and fresh
connections passed on all three hosts. The AWS SSH session document has no
maximum duration, no account session-preferences document exists, and the
tested SSO role credentials last 12 hours. Recheck expiry before a long release.

| SSH alias and existing IP | SSM target | AWS scope |
| --- | --- | --- |
| `leadpoet-gateway-ssm`, `52.91.135.79` | `i-07e945bb2653c2e8f` | `493765492819`, `us-east-1` |
| `leadpoet-validator-ssm`, `100.59.201.156` | `i-07ab2b90c537d0f28` | `493765492819`, `us-east-1` |
| `leadpoet-dashboard-ssm`, `13.218.210.185` | `i-0e428b150faa9d20f` | `493765492819`, `us-east-1` |

SSM carries the normal SSH connection. The host still requires the approved
private key and the existing trusted host key. AWS CLI v2 and the Session Manager
plugin must already be installed. The instance needs a working SSM agent, instance
role, SSM network access, and a caller permitted to start `AWS-StartSSHSession` on
the exact target. Do not grant broad access to fix a failed check.

## Install without losing trust or other SSH settings

Keep public SSH access until both a new SSM connection and independent AWS
control-plane recovery work. Before installation, save private, dated copies of
`~/.ssh/config` (if present), `~/.ssh/known_hosts`, and the affected security-group
rules. Keep these files outside Git. Record rule IDs, group IDs, protocol, ports,
sources, descriptions, attached resources, and the account/region used to read them.

Use `ssh-keygen -F 52.91.135.79 -f ~/.ssh/known_hosts` and the same command for
`100.59.201.156` and `13.218.210.185` to check the existing entries, including hashed entries. Verify
their fingerprints against the already trusted record. If trust is absent or
uncertain, obtain the key through an independently authenticated channel first.
Do not accept a new key because it arrived through SSM. Do not use `ssh-keyscan`
as proof of identity or change host-key checking to accept a replacement.

The inspected operator config already has SSM routes for `gateway`, `validator`,
and `subnet71` through `leadpoet-lab`. Preserve those routes, the existing
ProxyCommand environment handling, and the approved local key path. Prefer adding
the explicit IP and new alias to the three verified Host patterns. Do not replace
the file or switch routine SSH to an administrator profile. Preserve the
existing `subnet71` mapping; its direct host identity is now verified.

For an operator without these routes, copy [the include](../deploy/ssh-ssm.conf) to `~/.ssh/leadpoet-ssm.conf` with mode
`600`. Check the operator-specific `IdentityFile` path and permissions. Edit
`~/.ssh/config` to put these lines **before** existing options and Host blocks;
keep all existing contents below them:

```sshconfig
Include ~/.ssh/leadpoet-ssm.conf
Host *
```

The `Host *` line restores the scope before existing global options. OpenSSH uses
the first value for most options, so an include below a general rule can leave
direct SSH active. Inspect `ssh -G leadpoet-gateway-ssm`, `ssh -G 52.91.135.79`,
all validator targets, and all dashboard targets. Require the exact instance `hostname`, IP
`hostkeyalias`, named profile/region `proxycommand`, `stricthostkeychecking true`,
and `forwardagent no`. Also inspect one unrelated host to check its settings.
Resolve any conflicting identity files or later Match blocks before continuing.

## Refresh authorization and prove new connections

Refresh SSO with `aws sso login --profile leadpoet-lab`. The profile must use the
approved SSO account and role. Check the account before connecting; an error or
wrong account must stop the command:

```bash
# access-preflight
account="$(aws --profile leadpoet-lab --region us-east-1 sts get-caller-identity --query Account --output text)" &&
  test "$account" = 493765492819 &&
  ssh -o BatchMode=yes leadpoet-gateway-ssm true
```

Repeat the SSH check for `leadpoet-validator-ssm` and `leadpoet-dashboard-ssm`. The explicit proxy profile and
region take precedence over ambient AWS defaults. A missing or expired profile
must fail; there is no direct-SSH fallback. Refresh SSO, then start a fresh SSH
session. An open session is not proof that a new one will work.

Check the effective Session Manager maximum duration and idle settings for SSH
sessions, plus the SSO/role credential lifetime. They must support the canonical
200-minute drain **plus restart and recovery margin**; plan at least 270 minutes
for the deployment window. Do not start a release near credential/session expiry.
A keepalive does not extend the maximum session duration. Verify the actual
SSH-over-SSM duration policy; do not infer it from an ordinary shell session.

Prove stdin and file transfer on each alias before changing ingress:

```bash
printf 'ssm-stdin-check\n' | ssh -o BatchMode=yes leadpoet-gateway-ssm 'cat'
probe="$(mktemp)" && printf 'ssm-scp-check\n' > "$probe"
remote_probe="$(ssh leadpoet-gateway-ssm 'umask 077; mktemp /tmp/leadpoet-ssm-access.XXXXXXXX')" &&
  scp "$probe" "leadpoet-gateway-ssm:$remote_probe" &&
  ssh leadpoet-gateway-ssm "cat '$remote_probe' && rm -- '$remote_probe'"
rm -- "$probe"
```

The remote `mktemp` reserves a private, unique probe. Confirm exact returned
bytes and remove only the probe you created; inspect a failed transfer before cleanup. Repeat for the validator and dashboard. SSM SSH
tunnels do not provide Session Manager shell command-content logging; retain the
canonical private deployment logs and AWS session audit records.

## Keep the canonical release path

The existing controller uses ordinary SSH and accepts these aliases. From the
clean checkout of exact current `origin/main`, use the same deployment lock and
private log directory as the deployment owner:

```bash
python3 -B -m scripts.restart_arena_pair \
  --commit <full-current-main-sha> \
  --gateway ec2-user@leadpoet-gateway-ssm \
  --validator ec2-user@leadpoet-validator-ssm \
  --identity "$HOME/.ssh/leadpoet-2026-07-28.pem" \
  --logs <new-private-log-directory> \
  --deployment-lock <existing-shared-deployment-lock>
```

All controller source, drain, epoch, ownership, readiness and rollback gates still
apply. Follow [paired restart recovery](arena_paired_restart.md) after a failure;
do not clear a retained guard or run concurrent deployments. Verify actual Arena
and validator results after restart. Test fresh sessions after ingress removal.

## Emergency recovery if SSM fails

Before cutover, prove independent AWS console/control-plane access with the
approved role and MFA. Keep the exact security-group backup and trusted host keys
available outside the affected hosts. Test the recovery procedure before relying
on it; a current SSM connection cannot prove independent recovery.

If new SSM sessions fail, inspect IAM, agent, network, session and credential
errors first. Use the independent AWS control plane to restore **only TCP 22 from
the operator's current verified public IPv4 `/32`** to the exact affected group.
Record the new rule ID. Use the saved rule data to confirm the group and previous
SSH intent; do not restore a broad old source or unrelated port 3000 rule. Never
open `0.0.0.0/0`. If the operator uses IPv6, use a verified single-address `/128`
only after confirming the host's IPv6 route and attachment.

Bypass the SSM include explicitly, while keeping the existing pinned host key:

```bash
ssh -F /dev/null -o StrictHostKeyChecking=yes -o HostKeyAlias=52.91.135.79 \
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts" -o IdentitiesOnly=yes \
  -o ForwardAgent=no -o ConnectTimeout=15 \
  -i "$HOME/.ssh/leadpoet-2026-07-28.pem" ec2-user@52.91.135.79
```

For the validator, use `100.59.201.156` for both host and `HostKeyAlias`; for the
dashboard, use `13.218.210.185`. Confirm
host identity and repair SSM through the approved path. Prove fresh SSM sessions
and preserved application behavior, then remove only the temporary rule by its
recorded ID. Compare the final rules with the exact backup and intended cutover.
Do not change unknown attachments or assume security groups act independently:
the effective ingress is the union of every group attached to an interface.
