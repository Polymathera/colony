# GitHub App setup

Colony talks to GitHub through a **GitHub App** rather than personal access tokens. Setup spans three roles; each does its piece once:

| Role | Who | What they do | When |
|---|---|---|---|
| **Service provider** | The company / team running this Colony deployment (e.g. a SaaS vendor selling Colony, or an internal platform team running it for the enterprise). | Registers a single Colony GitHub App + sets four deploy-wide env vars. | Once per Colony deployment. |
| **Tenant admin** | An enterprise customer of the service provider; specifically the person who can install GitHub Apps into that enterprise's GitHub organisation. | Installs the service provider's Colony App into the tenant's GitHub org + pastes the resulting installation id into the dashboard. | Once per tenant. |
| **User** | An individual employee of the tenant. | Clicks "Connect GitHub" on their Colony profile to OAuth-verify their personal GitHub identity. | Once per user (re-verify on email/login change). |

This page covers the **service provider** + **tenant admin** steps. The user step lives in [`Connect GitHub`](connect-github.md).

> Required before you point Colony at any private GitHub repository — design monorepo, project-planning targets, issue automation, anything. Without it: clones / pushes / REST calls all fail with "Authentication failed."

## 0. Why an App, not a PAT

- **Per-tenant scoping.** Each tenant installs the App into their org and grants access to specific repos. Tenant A cannot read Tenant B's repos even when both run on the same Colony deployment.
- **Short-lived tokens.** The agent mints installation tokens that expire in ~60 minutes and refreshes them in the background. A leaked token's blast radius is one hour, not forever.
- **Identifiable bot.** Comments, commits, and assignments by Colony show up on GitHub as `<your-app-slug>[bot]` — distinct from any human user.
- **Per-user OAuth on the same App.** The same App registration handles the "Connect GitHub" flow for human users (next doc), so you set up one App and get both pieces.

---

## 1. Register the Colony GitHub App  *(service provider · once per Colony deployment)*

You're registering **one** App on GitHub. That single App will be shared by every tenant of this Colony deployment (each tenant *installs* the same App into their own org in §3 — they don't register their own). Decide which GitHub account owns the App registration: a service-provider org account is the right answer for a SaaS Colony; for an internal-only deployment, the platform team's GitHub org works.

1. Go to **Settings >> Developer Settings** or <https://github.com/settings/apps/new> (or the org form:
   `https://github.com/organizations/<org>/settings/apps/new`).
2. Fill in:
   - **GitHub App name**: e.g. `Polymathera Colony` — this becomes the bot's GitHub login as `polymathera-colony[bot]`, the identity that posts comments, assigns issues, and commits on behalf of your Colony deployment.
   - **Homepage URL**: your Colony dashboard's public URL.
   - **Callback URL**: `<your-colony-public-url>/api/v1/auth/github/callback` — required for the per-user OAuth flow (the "Connect GitHub" button on the user profile). The URL Colony sends to GitHub is constructed as `<request scheme>://<request host>/api/v1/auth/github/callback`, so it must match what's registered here byte-for-byte (GitHub refuses mismatches with a *"The redirect_uri is not associated with this application"* page). Two common shapes:
     - **Local development**: `http://localhost:8080/api/v1/auth/github/callback` (port matches `COLONY_DASHBOARD_UI_PORT` in `docker-compose.yml`, default 8080). GitHub Apps accept `http://localhost` — the only `http://` URL they accept — so no HTTPS / tunnel needed for local work.
     - **Hosted deployment**: `https://<your-colony-dashboard-host>/api/v1/auth/github/callback`.
     If you run both local and hosted environments against the same App, click **Add Callback URL** on the App settings page to add the second URL alongside the first — GitHub allows multiple, and the connect router sends whichever one matches the live request's host. Can be left blank if no users in any tenant will ever connect GitHub; you can add it later by editing the App.
   - **Setup URL**: leave blank.
   - **Webhook** (a section on the App registration form, further down the same page): leave its **Active** checkbox ticked if you want inbound events (issue opened, PR comment, …) to flow into Colony's blackboard for downstream reactions (P10 mention routing, the `InteractionLog` cross-channel log). When ticked, fill the three sub-fields in the **Webhook** section as follows:
     - **Webhook URL** (text input directly under the Active checkbox): the single public URL GitHub will POST every webhook delivery to. Type the value below into that input box on the GitHub form — GitHub stores it on the App registration and uses it as the delivery destination for every install of this App, across every tenant:
       - **Hosted / production** (single Colony dashboard reachable from the public internet): `https://<your-colony-dashboard-host>/api/v1/github/webhook`. The same URL handles every tenant — see §1.5 below for how the receiver demuxes deliveries back to the right tenant + colonies.
       - **Local development** (`localhost` — GitHub can't POST to it directly): you need a public tunnel (smee\.io is the supported default; ngrok / cloudflared work too if you wire them yourself) that forwards `https://<tunnel-host>/...` to `http://localhost:8080/api/v1/github/webhook`. `colony-env up` ships a built-in smee.io forwarder *sidecar* — it's a one-line opt-in. End-to-end:
         1. Visit <https://smee.io/new> in a browser. The page issues a fresh channel and redirects to `https://smee.io/<channel-id>` — copy that URL.
         2. Paste the channel URL into your Colony deployment's `.env` as `POLYMATHERA_SMEE_FORWARDING_URL=https://smee.io/<channel-id>` (see [`.env.template`](../../src/polymathera/colony/cli/deploy/.env.template)). This is local-dev-only; leave the var empty in any production deployment.
         3. Run `colony-env up` (or `colony-env down && colony-env up ...` if the cluster is already up). The compose provider sees `POLYMATHERA_SMEE_FORWARDING_URL` set and activates the `local-webhook` profile, which starts a `smee-forwarder` sidecar container that runs `npx smee-client@latest --url <your-channel> --target http://dashboard:8080/api/v1/github/webhook` (resolved over the docker-compose network — no `localhost` indirection). It streams deliveries from smee.io to your local dashboard over a persistent SSE connection. `colony-env down` tears the sidecar down with the rest of the stack.
         4. Paste the same `https://smee.io/<channel-id>` URL into GitHub's **Webhook URL** input box on the App registration form.
         > **Why a sidecar, not a host process?** `smee-client` needs to run for the *lifetime of the dev session*, which means owning a long-lived shell, surviving laptop reboots, and being remembered for teardown. Wiring it into `colony-env`'s docker-compose orchestration (gated by a single env var) makes setup and teardown automatic, ties the forwarder's lifecycle to the cluster it serves, and keeps Node off the host. If smee.io itself is down the sidecar logs reconnect attempts but doesn't crash anything else — the dashboard and `mode: poll` fallback keep working.
     - **Webhook secret** (text input immediately below **Webhook URL** on the same GitHub form): a deploy-wide HMAC key the receiver uses to verify every delivery actually came from GitHub. Generate one long random value (e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`), then paste the **same value** into **two** places:
       1. The **Webhook secret** input on this GitHub App registration form (so GitHub signs every outgoing delivery's body with it).
       2. The `GITHUB_WEBHOOK_SECRET=` line in your Colony deployment's `.env` file (see §2) — so the receiver computes the same HMAC and matches the `X-Hub-Signature-256` header GitHub sends. A mismatch yields 401 with no further work.
     - **SSL verification**: leave **Enable SSL verification** ticked (default) for hosted deployments — Colony serves the receiver over HTTPS. For smee\.io / ngrok HTTPS tunnels the default also works.
   - **Subscribe to events** (in the "Permissions & events" tab on the GitHub App registration form — this is the per-event-type opt-in that tells GitHub which deliveries to send to your Webhook URL): tick the boxes for `Issues`, `Issue comments`, and `Pull requests`. Leave the rest (Discussions / Wiki / Release / etc.) unticked — the receiver returns `{"status": "ignored"}` (HTTP 200) for any event type Colony's normalizer doesn't handle in v1, so subscribing to extra events just costs you bandwidth.
   - **Polling fallback** (not a GitHub-side setting — a *per-colony* Colony-side switch): each colony's `.colony/github_inbound.yaml` chooses between `mode: poll` (the agent-side ticker hits the GitHub GraphQL API on a cadence — works without a public webhook URL, ideal for local dev) and `mode: webhook` (the dashboard receiver is the active surface — strictly preferable for prod once the App webhook is wired). Both modes emit the same downstream `GitHubEventProtocol` blackboard events; subscribers don't care which surface fired.
   - If no colony in any tenant will ever use `mode: webhook`, **un-tick** the **Webhook → Active** checkbox above. The receiver short-circuits to 503 in that case, and the poll-only path still works.
3. **Repository permissions** — set each to *Read & Write* unless
   noted:
   - **Contents** — clone + push to the design monorepo.
   - **Issues** — `create_issue`, `assign_issue`, `comment_on_issue`.
   - **Pull requests** — `create_pull_request`, `comment_on_pr`, `review_pr`.
   - **Metadata** (auto-included, *Read*) — required.
4. **Organization permissions** — only if any tenant will use Colony's Projects v2 attachment actions (`add_issue_to_project`, `list_project_items`, the `auto_attach_to_default_project` flag on `create_issue`):
   - **Projects** — *Read & Write*.

   > **What's Projects v2?** GitHub's modern project-board feature (introduced 2022) — boards, tables, and roadmap views built on top of issues and pull requests. Distinct from the legacy "Projects (classic)" which GitHub is deprecating. Colony's `ProjectPlanningMission` optionally attaches every issue it creates to a Projects v2 board the tenant has set as their default; if you don't plan to use that, skip this permission.
5. **Account permissions** — only if you want users to connect GitHub via the "Connect GitHub" button on their profile:
   - **Email addresses** — *Read* (the OAuth callback fetches the user's verified primary email).
6. **Where can this GitHub App be installed?** Pick *Any account* for a multi-tenant SaaS deployment (tenants from any GitHub org should be able to install). Pick *Only on this account* if the deployment serves only one organisation.
7. Save the App. GitHub shows the **App ID** at the top of the settings page — note it down.
8. Scroll to **Private keys** and click **Generate a private key**. GitHub downloads a `.pem` file — keep it safe; you'll paste its contents into `.env` in §2.
9. *Only if you enabled step 5* (per-user OAuth): scroll to **Client secrets** (further down the App's settings page) and click **Generate a new client secret**. The page now shows two values you'll need in §2: the **Client ID** (visible at the top of the settings page near the App ID) and the **Client secret** (only shown once — copy it now).

   > **What are the Client ID and Client secret?** They're the same Colony GitHub App's *OAuth client credentials* — distinct from the App ID + private key (which authenticate the App server-to-server). The Client ID + secret authenticate the **user-to-server OAuth web flow** that Colony uses to verify each Colony user's personal GitHub identity. Both credential pairs live on the same App; the service provider sets both, and the OAuth credentials are reused for every "Connect GitHub" click by every user across every tenant. Colony stores them as deploy-wide env vars in §2 — they are not per-tenant or per-user.

## 1.5. How one webhook URL serves every tenant  *(reference — no action required)*

You configured **one** Webhook URL on **one** GitHub App registration, and the App is shared by every tenant. Yet each tenant installs the App into their own GitHub org and grants access to *their* repos. So how does a delivery for `tenant-acme/their-repo` find its way to the right colonies inside Colony, without leaking to `tenant-globex`?

The receiver demuxes purely on the `installation.id` field that GitHub stamps onto every webhook payload. The flow is:

1. **Tenant admin installs the App** (§3 below). GitHub creates a new *installation* — an `(App, GitHub org)` pair — and assigns it a stable numeric `installation_id`. The tenant admin pastes that id into the Colony dashboard's **Tenant GitHub Installation** panel, which writes it to the row of `tenants.github_installation_id` for *their* tenant in Colony's Postgres.
2. **Something happens on a repo** the tenant granted Colony access to (a user opens an issue, comments on a PR, …). GitHub picks every App installed on that repo, looks up each App's **Webhook URL**, and POSTs the event there. For Colony's App, that single URL is your dashboard's `/api/v1/github/webhook`.
3. **The receiver** ([`web_ui/backend/routers/github_webhook.py`](../../src/polymathera/colony/web_ui/backend/routers/github_webhook.py)):
   - Verifies HMAC against the **single deploy-wide** `GITHUB_WEBHOOK_SECRET` (the same secret you set on the App registration in §1 — every delivery from every install is signed with it, because the secret lives on the App, not on the install).
   - Reads `payload["installation"]["id"]` (GitHub stamps this on every App-delivered event).
   - Runs `SELECT tenant_id FROM tenants WHERE github_installation_id = $1` to find which Colony tenant owns this installation.
   - Fans out the normalized `GitHubEventProtocol` write to **every colony in that tenant** (`SELECT colony_id FROM colonies WHERE tenant_id = $1`), writing each to the colony-scoped `EnhancedBlackboard`. Other tenants' colonies never see the event because the lookup never names them.

What this means in practice for a multi-tenant SaaS deployment:

- **One dashboard URL, one webhook secret, one App registration.** All shared by every tenant. No per-tenant DNS or per-tenant secret to manage.
- **Per-tenant isolation is enforced by the `tenants.github_installation_id` lookup.** A delivery whose `installation_id` doesn't appear in any tenant row returns `{"status": "no_tenant_for_installation"}` (200 — GitHub doesn't retry) and writes nothing.
- **Intra-tenant fan-out is currently broadcast, NOT filtered.** Every colony inside the same tenant receives every webhook the receiver accepts for that tenant — even if the colony's `.colony/github_inbound.yaml` `poll_repos` list wouldn't subscribe to that repo. This is a v1 simplification documented in the publisher's own module docstring ([`publisher.py`](../../src/polymathera/colony/web_ui/backend/github_webhook/publisher.py)). The practical cost is wasted blackboard writes + duplicate `interaction_log` rows in colonies that don't care about the repo; it's not cross-colony data *leakage* (every colony inside a tenant is by-design allowed to see that tenant's events) but it IS wasted work. A follow-up will add the `poll_repos` filter when the cost becomes real. Until then, operators running tenants with many colonies + a noisy webhook stream should prefer `mode: poll` per-colony — the poller honors `poll_repos` natively.
- **Per-user identity is NOT a routing input.** GitHub webhooks identify the human actor by their GitHub login (`payload["sender"]["login"]`, normalized into `value["author_login"]`); Colony stores it as data on the blackboard event so downstream consumers (mention routing, `InteractionLog`) can see *who* did the thing — but the receiver never looks up "which Colony user is this?" because GitHub events fan out to *colonies*, not to *user sessions*. (User sessions consume colony-scoped events via the system session's `SessionAgent`; see [`design_top_level_design_process.md`](../../../colony_docs/markdown/plans/design_top_level_design_process.md) §10–§15.)
- **Local dev** uses the same routing logic — the only difference is that the **Webhook URL** points at a smee\.io / ngrok tunnel that forwards to your local dashboard. The HMAC / lookup / fan-out steps run identically.

## 2. Configure deploy-wide env vars  *(service provider · once per Colony deployment)*

Same role as §1 — these env vars belong to the Colony deployment, not to individual tenants or users. Every tenant + every user shares them implicitly through the running deployment.

Edit `colony/src/polymathera/colony/cli/deploy/.env` (or your deployment's equivalent — `.env.template` documents the shape):

```bash
# Required — minted into installation tokens.
GITHUB_APP_ID=123456
# Single line, ``\n`` escapes between header / body / footer. See the
# PEM-formatting note below the snippet — this is NOT optional.
GITHUB_PRIVATE_KEY_PEM="-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n"

# Optional — only when you want users to be able to click
# "Connect GitHub" on their profile.
GITHUB_APP_CLIENT_ID=Iv1.abc...
GITHUB_APP_CLIENT_SECRET=...

# Optional — only when at least one colony uses ``mode: webhook`` in
# its ``.colony/github_inbound.yaml``. Must match the "Webhook secret"
# value set on the GitHub App in §1. Leave empty for poll-only
# deployments (the receiver short-circuits to 503; the agent-side
# poller still ticks).
GITHUB_WEBHOOK_SECRET=...
```

> **PEM must be single-line with `\n` escapes.** `docker-compose`'s `environment:` list truncates env-var values at the first real newline (everything after the `BEGIN RSA PRIVATE KEY` header gets dropped on its way to the container — pyjwt then fails with `InvalidKeyError: Could not parse the provided public key`). The `GitHubAuthConfig._normalize_pem` validator converts the `\n` escapes back to real newlines before signing, so the key reaches pyjwt in valid PEM form. Convert a downloaded `.pem` with:
>
> ```bash
> awk 'BEGIN{ORS="\\n"} {print}' /path/to/colony-app.pem
> ```
>
> Then wrap the output in double quotes when pasting into `.env`.

Then restart the cluster so Ray services pick up the new env:

```bash
colony-env down && colony-env up --workers 3
```

The deploy-wide values let Colony mint a JWT (App ID + private key) and run the OAuth web flow (client id + secret). They are necessary but not sufficient — the per-tenant installation (below) is what actually grants access to specific repos.

## 3. Install the App on the tenant's GitHub org  *(tenant admin · once per tenant)*

This is the only step that involves the **tenant admin** (the person inside the customer enterprise who has GitHub-App-install rights on the tenant's GitHub organisation — typically the org owner or a GitHub-org admin). The service provider does not have access to do this; each tenant runs it once when they sign up to Colony.

1. From the App's public page (the one named in step 1.2's Homepage URL, or the GitHub-hosted page at `https://github.com/apps/<your-app-slug>` such as `https://github.com/apps/polymathera-colony`), click **Install**.
2. Pick the GitHub organisation (or personal account) the tenant wants Colony to operate on.
3. Choose **Only select repositories** and pick the ones Colony should have access to — Colony will refuse anything not in this list. *All repositories* is fine if you want Colony to work on every repo, but tighter is safer.
4. After confirming, GitHub redirects to a URL containing `installation_id=<number>`. **Copy that number** — it's the tenant's per-tenant installation id.
5. In the Colony dashboard, open the **Tenant GitHub Installation** panel on the landing page (see [`connect-github.md`](connect-github.md) for screenshots of the landing-page layout). Paste the number into the **Installation id** input and click **Save**.

That's it. The next session this tenant starts will see the installation id in agent metadata, mint a per-tenant installation token, and use it for both REST API calls and git push/pull.

## 4. Verify  *(tenant admin or end user · after §3 lands)*

Start a session as a user in the tenant whose installation you just configured. In the chat, ask something like:

> *List the open issues in `acme-org/their-repo`*

If the App + installation are wired right, the agent calls `list_issues(repo="acme-org/their-repo")` and returns them. If something's off, the action returns an error dict explaining what's missing — typically one of:

| Error message | Fix |
|---|---|
| `GITHUB_APP_ID and GITHUB_PRIVATE_KEY_PEM env vars are required` | Step 2 wasn't done; restart after editing `.env`. |
| `per-tenant GitHub App installation id is missing` | Step 3 wasn't done; tenant admin pastes the installation id into the dashboard. |
| `GitHub App slug not configured`-style messages from the OAuth path | Step 1.8 (client id/secret) wasn't done; per-user "Connect GitHub" doesn't work yet, but everything else does. |

For git push / clone failures, the [`_classify_git_clone_error`](../../src/polymathera/colony/distributed/stores/git.py) classifier shapes generic `git` auth errors into a typed `GitAuthError` whose message names the same three fix paths.

## 5. Rotation

**Service provider** (deploy-wide credential rotation):

- **App private key**: generate a new key from the App settings → Private keys page; copy into `.env`; `colony-env down && up`. GitHub leaves the old key valid for 24h, so you can stage the rollout with no downtime.
- **OAuth client secret**: generate a new one from the Client secrets section; copy into `.env`; restart. Existing user OAuth identities on `users.github_*` are not affected (Colony discards the user-to-server token after the verification round-trip — only the deploy-wide secret matters for future "Connect GitHub" clicks).

**Tenant admin** (per-tenant access revocation):

- **Removing a tenant's access**: the tenant admin uninstalls the App from the tenant's GitHub org. The next session-create in that tenant fails to mint an installation token; agents that depend on git push / REST surface a clean `GitAuthError`. Colony does **not** automatically remove the stored `tenants.github_installation_id` from Postgres — the tenant admin clears it via the **Tenant GitHub Installation** panel on the dashboard (paste an empty value + Save).

## How the deploy-time pieces flow to the runtime

```
.env file (deploy-wide secrets)
    │
    ├── GITHUB_APP_ID  ───────────►  GitHubAuthConfig.app_id ───────────┐
    ├── GITHUB_PRIVATE_KEY_PEM ───►  GitHubAuthConfig.private_key_pem ──┤
    │                                                                   │
    └── GITHUB_APP_CLIENT_ID / _SECRET ──► routers/github_oauth.py      │
                                            │                           │
                                            ▼                           │
                                      GET /auth/github/connect          │
                                      GET /auth/github/callback         │
                                            │                           │
                                            ▼                           │
                                      OAuth callback writes verified    │
                                      identity to users.github_*        │
                                                                        │
Postgres                                                                │
    │                                                                   │
    ├── tenants.github_installation_id (set by tenant admin via UI)  ──►│
    ├── users.github_login / github_email / git_user_name ─────────────►│
    └── colonies.commit_principal / commit_co_author ──────────────────►│
                                                                        │
                            session-create handler (sessions.py)        │
                                threads all three into agent metadata   │
                                                                        │
                                                                        ▼
                            Agent process                               │
                                ├── GitHubCapability ── mints REST tokens
                                ├── DesignMonorepoCapabilityBase ── starts
                                │   git_credentials.py refresh task that
                                │   writes /tmp/colony-git-credentials
                                └── _resolve_attribution ── per-commit
                                    principal + Co-Authored-By trailer
```

## Related

- [`connect-github.md`](connect-github.md) — what users do once the App is installed.
- [`git-attribution.md`](git-attribution.md) — how per-colony preferences combine with per-user OAuth identity to drive commit attribution.
- [`capability-setup.md`](capability-setup.md) — the broader capability-env-vars reference.
- [`architecture/github-capability.md`](../architecture/github-capability.md) — internal token mint + cache flow.
