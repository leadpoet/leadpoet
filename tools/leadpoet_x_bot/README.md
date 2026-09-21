# Leadpoet X reply bot

One Python process, one SQLite file, no third-party packages. Python 3.10+ on Linux or macOS. It uses the official X API v2 and OAuth 1.0a user authentication. Nothing is connected to the subnet gateway or validator.

For a direct comment under a configured Leadpoet root post:

1. Match `Poet`, any case or punctuation, plus a small configurable typo list.
2. Reply **Sending!** directly to that comment.
3. DM its author:

> Book a quick demo, we’ll define your ICP, and get you 100 free lead credits:
>
> https://cal.com/team/leadpoet/chat

4. If X explicitly reports that this recipient cannot receive the DM, reply to the original comment:

> I couldn’t send you a DM. Please open your DMs or DM us first.

Every action is recorded separately in SQLite before its API call. A restart resumes the next safe step. It does not repeat a confirmed or uncertain action. Live replies and DMs are both disabled by `DRY_RUN=true`.

Matching uses whole words. `Poet`, `poet`, `POET`, `Poet!`, and `Poet, please` match. Default typo aliases are `peot`, `poett`, `poeet`, `poat`, `poer`, and `poey`. `post`, `pet`, `port`, and `poetry` do not match. Set `TRIGGER_ALIASES` to replace the typo list, or to an empty string for exact-word matching only. A different trigger has no default typo aliases. Replies to other posts, nested replies, and our own account's comments are ignored.

## X account and credentials

1. Sign in as Leadpoet at the [X Developer Console](https://console.x.com/). Register a developer account and create an app. Describe this as a fixed-text, opt-in campaign reply and DM bot.
2. In the app's user authentication settings, enable OAuth 1.0a with **Read, write and Direct Messages** permission. Save the settings before generating user tokens. If asked for website/callback URLs, use your valid company website and a callback URL you control; this single-account bot does not run a callback server.
3. In **Keys and tokens**, obtain the app's API Key and API Key Secret. Generate the Access Token and Access Token Secret for the Leadpoet account. Regenerate user tokens if they were created before granting write/DM permission. If another account owns the app, authorize Leadpoet through X's [user authorization flow](https://docs.x.com/fundamentals/authentication/oauth-1-0a/obtaining-user-access-tokens).
4. Enable access/billing for recent search, post lookup, user lookup, post creation, and direct messages. Current X access is usage-based; confirm availability and spending limits in your console. A dry run that reads X still uses billable API reads. This bot does not buy credits.
5. Keep all four values in the environment file below. An app-only Bearer Token cannot replace these user credentials. Startup calls `/2/users/me` and verifies that the configured posts belong to that account.

**Current live-reply restriction:** X's [Manage Posts docs](https://docs.x.com/x-api/posts/manage-tweets/introduction) state that self-serve replies require the recipient to have explicitly summoned the replying account with an @mention or by quoting one of its posts. A bare `Poet` comment is detected, but the public reply can be rejected by X. Ask users to reply `@YOUR_ACTUAL_HANDLE Poet`; confirm posting eligibility on your account before launch. There is no code bypass for an API permission restriction.

**Public reply count:** [X automation rules](https://help.x.com/en/rules-and-policies/x-automation) limit automated public replies to one per interaction. The requested acknowledgement plus a fallback is two public replies when a DM is unavailable. Test this flow in dry run; resolve this policy constraint before enabling live use. The simpler compliant ordering is to try the DM first and send only one public success/failure reply.

Make the campaign post clear about the automated reply and DM, including a way to opt out. For example: “Reply @YOUR_ACTUAL_HANDLE Poet to request our automated demo link by DM and 100 free lead credits. To opt out, DM us.” Monitor that inbox and add opt-out user IDs to `DENIED_USER_IDS`, then restart the bot. This is a fixed-text bot, not an AI-generated reply bot. It does not automate the DM inbox or retry delivery after a recipient opens their DMs; staff can handle their inbound DM.

## Offline test first

From the repository root:

```sh
python3 -m unittest discover -s tools/leadpoet_x_bot -p 'test_bot.py' -v
cd tools/leadpoet_x_bot
MONITORED_POST_URLS='https://x.com/leadpoet/status/100,https://x.com/leadpoet/status/200' DRY_RUN=true STATE_DB=/tmp/leadpoet-x-bot-demo.sqlite3 \
  python3 bot.py --fixture fixture.json
# Repeat exactly: no new actions because the same comment IDs are already recorded.
MONITORED_POST_URLS='https://x.com/leadpoet/status/100,https://x.com/leadpoet/status/200' DRY_RUN=true STATE_DB=/tmp/leadpoet-x-bot-demo.sqlite3 \
  python3 bot.py --fixture fixture.json
```

Use a new demo database path for a fresh test. The fixture covers both configured posts, case/punctuation variants, a typo, and a recipient with unavailable DMs. Unrelated text, near-match common words, another post, our own comment, and a nested comment are excluded. Fixture mode makes no network calls and requires dry run. Its `dm_unavailable_user_ids` field simulates recipient DM failure; an online dry run cannot establish actual DM deliverability without sending.

## Configure and run locally

```sh
cd tools/leadpoet_x_bot
umask 077
cp .env.example .env
# Edit .env in your editor. Add credentials and real monitored post links.
# Set STATE_DB to an absolute, persistent local path you own.
# Keep DRY_RUN=true.
set -a
. ./.env
set +a
python3 bot.py --once
python3 bot.py
```

The program reads environment variables; it does not load `.env` automatically. Stop with Ctrl-C. Never commit `.env` or the database. Logs contain IDs/statuses, not credentials or reply bodies.

| Variable | Default / meaning |
| --- | --- |
| `X_API_KEY`, `X_API_SECRET` | Required app credentials for online runs |
| `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` | Required Leadpoet user credentials for online runs |
| `MONITORED_POST_URLS` | Comma-separated full `https://x.com/HANDLE/status/ID` links (`twitter.com` links also accepted); owned root posts only |
| `MONITORED_POST_IDS` | Optional comma-separated IDs; combined with URLs if both are set |
| `TRIGGER_WORD` | `Poet`; case-insensitive whole-word matching |
| `TRIGGER_ALIASES` | Default Poet typo list above; comma-separated whole-word alternatives |
| `REPLY_TEXT` | `Sending!` |
| `DM_MESSAGE` | Exact CTA shown above, including the blank line |
| `DM_UNAVAILABLE_REPLY` | Open-DMs request shown above |
| `POLL_INTERVAL_SECONDS` | `300` seconds |
| `DRY_RUN` | `true`; set explicitly to `false` to post |
| `STATE_DB` | `state.sqlite3`; use a stable absolute path in production |
| `DENIED_USER_IDS` | Empty; comma-separated opt-out user IDs |

To customize a multiline DM locally, use a quoted multiline environment value. In systemd's EnvironmentFile, use a quoted multiline value as well. Do not put shell substitutions or an `export` prefix in that file.

## Deploy on a small Linux host

Use an existing Linux VM with Python 3.10+ and systemd. No web server, inbound port, container, Redis, or managed database is needed. Allow outbound HTTPS to `api.x.com`. From a clean checkout of the tested commit:

```sh
sudo useradd --system --home /var/lib/leadpoet-x-bot --shell /usr/sbin/nologin leadpoet-x-bot
sudo install -d -m 755 /opt/leadpoet-x-bot
sudo install -m 644 tools/leadpoet_x_bot/bot.py /opt/leadpoet-x-bot/bot.py
sudo install -m 600 tools/leadpoet_x_bot/.env.example /etc/leadpoet-x-bot.env
sudoedit /etc/leadpoet-x-bot.env
# Fill credentials, post links; leave DRY_RUN=true and the default /var/lib/... STATE_DB.
sudo install -m 644 tools/leadpoet_x_bot/leadpoet-x-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now leadpoet-x-bot
sudo systemctl status leadpoet-x-bot
sudo journalctl -u leadpoet-x-bot -n 100 --no-pager
```

Submit a test reply from another account to a configured post. Confirm one dry-run acknowledgement and DM sequence. Restart the service and confirm that none of those actions repeat. Check the service logs after each change. Never run a second host with a separate database for this campaign.

After resolving X account access and the public-reply policy above, change **only `DRY_RUN=false`** in `/etc/leadpoet-x-bot.env` and run `sudo systemctl restart leadpoet-x-bot`. Dry-run and live claims/checkpoints are separate: enabling live will process matching replies in the recent-search window, including those previously tested. To stop replies and DMs, set it back to `true` and restart, or stop the service.

For updates, stop the service, install `bot.py` from the new tested commit, and start it. Retain `/var/lib/leadpoet-x-bot/` across upgrades. Back up the SQLite file while the service is stopped. Never restore an older database after posting: it may forget replies that were sent after the backup. Run only one instance on one local disk; the file lock prevents a second process using the same database.

## Reliability and operations

The bot persists each detected comment and commits a unique action claim before every public reply or DM. A confirmed action records the created post/event ID. It resumes pending steps from SQLite even if search no longer returns the comment. Removing a post from configuration or adding a recipient to the denylist stops their pending work.

A timeout, server error, malformed success, or crash during sending leaves an uncertain claim that is never automatically retried. An uncertain acknowledgement stops the sequence; an uncertain DM does not trigger the open-DMs fallback. This gives **at-most-once actions**, not guaranteed delivery. X's documented write endpoints provide no idempotency key that could guarantee both. Do not clear an uncertain record merely because you cannot find the message immediately. Deduplication is per comment; a new qualifying comment is a new interaction.

Explicit rate-limit rejections are retried after X's reset time/Retry-After. Read failures use backoff. Recipient DM error codes `150` and `349`, or an explicit recipient-unavailable error, enable the fallback. Generic `403`, missing app permissions, invalid credentials, rate limits, and server errors do **not** prove that the recipient closed their DMs. They must not produce that claim. See [X error definitions](https://developer.x.com/en/support/twitter-api/error-troubleshooting). Review failed/unknown/sending action records and logs; fix credentials or campaign permissions as needed. Do not delete processed IDs to force a retry.

Recent search only returns replies from the last seven days. Startup scans that window; later scans use a saved checkpoint with overlap to catch delayed indexing, and paginate all results. Checkpoints advance only after a complete scan. A downtime longer than seven days can miss replies, and search visibility/indexing can omit posts. New replies to old root posts can still be found. This small bot is intended for modest campaign traffic, not a firehose.

A 300-second interval means approximately 288 search requests per configured post per day before pagination/retries. X read/write charges are separate from VM cost. Increase the interval for lower request volume; watch the X console's usage and spending limit. Configure journal retention on the host as needed.

Official docs checked during implementation:

- [Recent search parameters and pagination](https://docs.x.com/x-api/posts/search-recent-posts)
- [Send direct messages](https://docs.x.com/x-api/direct-messages/manage/integrate)
- [DM endpoint](https://docs.x.com/x-api/direct-messages/create-dm-message-by-participant-id)
- [Create posts and replies](https://docs.x.com/x-api/posts/create-post)
- [OAuth 1.0a keys](https://docs.x.com/fundamentals/authentication/oauth-1-0a/api-key-and-secret)
- [Rate limits and response headers](https://docs.x.com/x-api/fundamentals/rate-limits)
