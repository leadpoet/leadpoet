---
name: loop-monitoring
description: Audit Leadpoet Research Lab gateway logs and live read-only state, then create a markdown report of bugs, suspected issues, evidence, and fixes. Use when the user says "run loop_monitoring.skill", "audit gateway logs", "create a research lab loop monitoring report", or asks why autoresearch loops, candidate scoring, stale-parent rebase, OpenRouter credit blocking, Exa rate limits, Docker/ECR builds, or public loop statuses look wrong.
---

# Loop Monitoring

## Workflow

1. Read `references/research_lab_flow.md` before interpreting logs.
2. Run `scripts/collect_gateway_loop_report.py` instead of ad hoc greps.
3. Treat log-only findings as hypotheses. Treat DB rows, event hashes, score bundle rows, candidate rows, and public card rows as stronger evidence.
4. Do not run production writes. Do not rsync, restart, mutate Supabase, kill processes, or repair rows from this skill.
5. Write reports under `/Users/pranav/Downloads/`.

## Default Command

```bash
python3 loop-monitoring/scripts/collect_gateway_loop_report.py \
  --ssh-target ec2-user@52.91.135.79 \
  --ssh-key ~/Downloads/leadpoet-gateway-tee-main.pem \
  --gateway-root /home/ec2-user/gateway
```

The script reads `/home/ec2-user/gateway/gateway.log`, `/home/ec2-user/gateway/nohup.out`, and accessible read-only journal output. If `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are present, it also reads current Research Lab views through PostgREST.

## Output Standard

The generated report must include:

- Executive Summary
- Production Flow Health
- Confirmed Bugs
- Potential Issues
- Evidence Table
- Recommended Fixes
- Do Not Resume Blindly
- Operator Commands To Run Next, read-only first

Every conclusion must show evidence. Redact secrets before quoting logs or rows.
