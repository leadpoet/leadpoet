# Arena SSH server restrictions

The reviewed drop-in changes only `PermitRootLogin`, `AllowAgentForwarding`,
and `X11Forwarding` to `no`. Normal access stays `ec2-user` over SSM with sudo.
TCP forwarding remains available for the private TLS database tunnel. Public
SSH ingress stays closed. This does not change the SSM agent, authorized keys,
PAM, passwords, user lists, ports, or application processes.

Before application, independently verify the AWS account and exact instance,
keep the approved SSM recovery path available, and prove a fresh SSM SSH login,
sudo, binary stdin/SCP, and the existing local TCP database tunnel. Check the
actual effective connection contexts and any `Match` rules. Root authorized
keys do not substitute for this check. The helper checks loopback SSH contexts
for root and ec2-user on both IPv4 and IPv6 loopback; other address-specific
rules need separate operator inspection.

Run the exact reviewed committed helper as root on the verified host:

```bash
sudo python3 scripts/configure_arena_sshd.py --check
sudo python3 scripts/configure_arena_sshd.py --apply
```

Default operation is check-only. It requires the existing global
`Include /etc/ssh/sshd_config.d/*.conf`, root-owned regular configuration files
and protected directories. It simulates the actual lexical include position,
uses native `sshd -t` and `sshd -T`, and refuses first-value conflicts or any
unrelated effective setting change. Existing unknown target bytes are refused.

Application installs only `/etc/ssh/sshd_config.d/00-arena-hardening.conf`
with an atomic no-clobber link, revalidates, and reloads sshd. It does not restart
the host or terminate existing SSH sessions. Failure removes only its own
unchanged candidate, validates the restored file set and attempts a reload.
A concurrent target change prevents rollback deletion and requires inspection.
No other configuration file is restored or overwritten. Repeated application
of an already-effective exact drop-in makes no change and does not reload.

Afterward, prove new SSM SSH connections and repeat sudo, binary stdin/SCP and
the existing TCP database tunnel. An existing open SSH session is not proof.
Keep the prior private inventory and native command results for review. Inspect
any reload or rollback failure before another application; do not clear public
SSH restrictions as a routine recovery step.

## Sudo security maintenance

The November 11 audit found `sudo-1.9.15-1.p5.amzn2023.0.2` on the three
managed hosts. AWS advisory ALAS2023-2026-3156 fixes a time-based sudoers
authorization-window error in `sudo-1.9.15-1.p5.amzn2023.0.4`. This is not a
general password bypass. Keep the root maintenance session open while applying
the signed vendor update. Review the exact transaction first and require that
only the `sudo` package changes; do not run a broad OS upgrade during a round.

```bash
sudo dnf --releasever=2023.12.20261012 --disablerepo='*' \
  --enablerepo=amazonlinux --setopt=install_weak_deps=False \
  --setopt=gpgcheck=True upgrade --assumeno \
  sudo-1.9.15-1.p5.amzn2023.0.4.x86_64
```

Use `-y` instead of `--assumeno` only after the single-package plan is verified.
Canary the dashboard first. Require unchanged sudo configuration hashes,
`visudo -c`, clean `rpm -V sudo`, an otherwise unchanged RPM inventory, and a
fresh ec2-user SSM connection that passes `sudo -n id -u`. Preserve application
processes; this package update needs no service or host restart. Upgrade to a
later reviewed vendor fix when required; do not downgrade a newer installation.

This targeted patch does not establish that the OS is fully patched. Kernel,
container runtime and other resident-service updates need separate drained
maintenance and runtime verification.
