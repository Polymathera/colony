# Capability operator setup

This guide tells operators what to configure on the **host** (env vars,
external accounts, Docker socket, optional volume mounts) so that the
five new agent capabilities work end-to-end.

The capabilities themselves are bound to the session agent
unconditionally — none of them crash the agent when their
configuration is missing. Each one degrades to a clean *"not
configured"* error dict that the LLM can surface to the user. So **no
setup is mandatory**; configure only the capabilities you actually
plan to use.

| Capability | Required env vars | External setup |
|------------|-------------------|----------------|
| [`VCMCapability`](../architecture/vcm-capability.md) | none | none |
| [`WebSearchCapability`](../architecture/web-search-capability.md) | `TAVILY_API_KEY` | Tavily account |
| [`ColonyDocsCapability`](../architecture/web-search-capability.md) | `TAVILY_API_KEY` (same) | Tavily account |
| [`SandboxedShellCapability`](../architecture/sandboxed-shell-capability.md) | none | Docker daemon (already mounted in dev) |
| [`UserPluginCapability`](../architecture/user-plugin-capability.md) | none | optional: host mount for custom skills |
| [`GitHubCapability`](../architecture/github-capability.md) | `GITHUB_APP_ID`, `GITHUB_INSTALLATION_ID`, `GITHUB_PRIVATE_KEY_PEM` | GitHub App registration + installation |

The full list of compose env-var passthroughs is in
[`colony/cli/deploy/docker/docker-compose.yml`](https://github.com/polymathera/colony/blob/main/colony/src/polymathera/colony/cli/deploy/docker/docker-compose.yml).
Every entry uses `${VAR:-}` so an empty value is acceptable; the
capability simply stays disabled.

## How env vars reach the cluster

`colony-env up` shells out to `docker compose up`. Docker Compose
substitutes `${VAR:-default}` against the **operator's shell
environment**, so any variable you `export` in the shell that runs
`colony-env up` flows through to ray-head and ray-worker.

The cleanest pattern is a `.env` file at the directory you launch
from (Compose auto-loads it):

```bash
# .env  (gitignored — never check in)
TAVILY_API_KEY=tvly-...
GITHUB_APP_ID=123456
GITHUB_INSTALLATION_ID=78901234
GITHUB_PRIVATE_KEY_PEM="-----BEGIN RSA PRIVATE KEY-----
MIIEow...
-----END RSA PRIVATE KEY-----
"
```

Then:

```bash
colony-env down && colony-env up --workers 3
```

To rotate a key: edit `.env`, then `colony-env down && colony-env up`.
Restarting just the cluster process (without `down`) doesn't pick up
new env vars.

## `WebSearchCapability` / `ColonyDocsCapability`

Both capabilities use the same `TavilyBackend` and therefore the same
`TAVILY_API_KEY`.

1. Create a Tavily account at <https://tavily.com>.
2. Generate an API key in the dashboard.
3. `export TAVILY_API_KEY=tvly-...` (or add to `.env`).
4. `colony-env down && colony-env up`.

The first `search_web` or `search_docs` call after restart will
exercise the backend; if the key is wrong, the action returns
`{ok: false, message: "Tavily API ..."}` instead of crashing the
agent.

To swap to a different backend (SerpAPI, Bing, Brave) without
changing capability code, subclass `SearchBackend` and pass it via
the blueprint — see the
[capability doc](../architecture/web-search-capability.md#backend-abstraction).

## `SandboxedShellCapability`

Already wired in dev. Docker socket is bind-mounted into both
ray-head and ray-worker; the curated image registry is mounted at
`/etc/colony/sandbox-images.yaml:ro`.

**Production hardening.** The dev mount of `/var/run/docker.sock`
gives anything inside ray-head root-equivalent access to the host
through the daemon. For multi-tenant deployments:

1. Run a separate hardened Docker daemon and expose it over TLS.
2. Set `DOCKER_HOST=tcp://hardened-daemon:2376` in ray-head /
   ray-worker (and remove the socket mount).
3. Mount the TLS client certs (`DOCKER_CERT_PATH`, `DOCKER_TLS_VERIFY`).

See `design_SandboxedShellCapability.md` §5.3 for the full plan.

**Image registry.** The default
[`sandbox-images.yaml`](https://github.com/polymathera/colony/blob/main/colony/src/polymathera/colony/cli/deploy/docker/sandbox-images.yaml)
ships two roles (`default`, `code_analysis`) both pointing at
`python:3.11-slim` — an image that exists on Docker Hub so the
capability works out of the box. To add a role with a real toolchain:

1. Edit `sandbox-images.yaml` (the file is mounted read-only from
   the repo, so edit it on the host and `colony-env down/up` to pick
   up the change).
2. Pin by digest in production: `image: ghcr.io/.../analyzer@sha256:abc…`.
3. Optionally declare named scripts so `execute_script(name=…)` is
   available — these are vetted command lines per role.

## `UserPluginCapability`

The capability ships a bundled
[`colony-samples` plugin](../architecture/user-plugin-capability.md#the-bundled-colony-samples-plugin)
(three skills) that auto-discovers without any setup — they live
inside the wheel and the session agent's blueprint passes
`extra_plugin_roots` to expose them.

To add **custom skills** that live on your host, mount their
directory into the container at one of the discovery roots:

| Discovery root inside the container | What it's for |
|-------------------------------------|----------------|
| `/etc/colony/skills` and `/etc/colony/plugins` | operator-managed shared skills (lowest priority) |
| `~/.colony/skills` and `~/.colony/plugins` | per-user skills — but `~` inside the ray container is the `ray` user's home, not the operator's |
| `/workspace/.colony/skills` | session-scoped (mounted per session by the workspace mount) |

The simplest pattern for a developer machine is to drop a
`docker-compose.override.yml` next to the main one:

```yaml
services:
  ray-head:
    volumes:
      - ${HOME}/.colony:/etc/colony:ro
  ray-worker:
    volumes:
      - ${HOME}/.colony:/etc/colony:ro
```

Skills you put in `~/.colony/skills/<name>/SKILL.md` on the host
appear at `/etc/colony/skills/<name>/SKILL.md` inside the container
and the capability picks them up at SYSTEM priority. (See the
[layout reference](../architecture/user-plugin-capability.md#layout)
for `SKILL.md` schema.)

A future Settings UI tab will surface discovered skills with
enable/disable toggles; until then, edit on the host and call
`reload_skills` from the agent (or restart the cluster).

## `GitHubCapability`

Uses GitHub App auth — not personal access tokens. The setup is
slightly more involved than the others because you have to register
an App with GitHub first.

### 1. Register a GitHub App

1. Open <https://github.com/settings/apps> and click **New GitHub
   App**.
2. **Name**: anything (e.g., `acme-colony`).
3. **Homepage URL**: a placeholder is fine for now.
4. **Webhook**: leave **Active** *unchecked* unless you intend to
   wire up the webhook endpoint (the capability's webhook receiver
   is a documented follow-up; the current code only emits
   blackboard events from its own action calls). If active, set
   the URL to something like
   `https://your-host/api/v1/github/webhook` and a strong secret.
5. **Repository permissions** the capability needs (set to
   *Read & Write* for the actions you plan to use):
   - **Contents** — for `get_file_contents`, `search_code`,
     `create_pull_request`.
   - **Issues** — for every issue/comment/label/claim action.
   - **Pull requests** — for PR list/get/create/comment/review.
   - **Metadata** (auto-included).
   - **Checks** (read) — for `get_pr_checks`.
6. **Organization permissions** (only if you'll use Projects v2):
   - **Projects** — *Read & Write*.
7. Save the App. GitHub shows the **App ID** at the top.
8. Scroll to **Private keys** and click **Generate a private key**.
   GitHub downloads a `.pem` file — keep it safe.

### 2. Install the App on your org / repos

1. From the App settings, click **Install App** in the left nav.
2. Choose the org / user and the specific repos to install on.
3. After install, GitHub redirects to a URL containing
   `installation_id=…`. Copy that number — that's your
   `GITHUB_INSTALLATION_ID`.

### 3. Set the env vars

```bash
export GITHUB_APP_ID="123456"
export GITHUB_INSTALLATION_ID="78901234"
# Either inline:
export GITHUB_PRIVATE_KEY_PEM="$(cat ~/.ssh/acme-colony.private-key.pem)"
# Or — for `.env` files that don't handle multi-line strings well —
# bind-mount the PEM into the container and pass `private_key_path`
# as a kwarg to GitHubCapability.bind() in your custom session-agent
# blueprint.

colony-env down && colony-env up --workers 3
```

### 4. Verify

In a new session, ask the agent something like:

> *"List the open issues in `acme/myrepo`."*

The agent should call `list_issues(repo="acme/myrepo")` and return
real data. If you see *"app_id, installation_id, and a private key
are all required"* in the response, the env vars didn't propagate —
check `docker compose exec ray-head env | grep GITHUB_`.

### 5. Audit + rate limits

- Every mutation writes a blackboard record at
  `audit:github:{ts}:{uuid}` — visible from the dashboard's
  Blackboard tab.
- The App's installation is rate-limited to 5 000 req/h. The
  capability surfaces primary-rate-limit errors as
  `{ok: false, status_code: 403}` and backs off automatically on
  secondary (abuse) limits, honouring `Retry-After`.

## `VCMCapability`

No setup. The VCM is part of the cluster's standard deployment, and
the capability is a thin facade over its existing endpoints. The
filesystem watcher uses [`watchfiles`](https://watchfiles.helpmanual.io/),
which is already in the dependency closure.

The watcher operates on paths visible to the ray-head /
ray-worker process — typically anything under `/mnt/shared/filesystem`
where Colony clones repos via `mmap_repo`. If you mount additional
host directories and want them watched, configure
`watch_root="/your/path"` on the capability blueprint.

## Where the env vars live in code

For traceability, here's where each variable is read:

| Env var | Reader | Action surface |
|---------|--------|----------------|
| `TAVILY_API_KEY` | `_github/auth.py`-style fallback in `TavilyBackend.__init__` | `search_web`, `fetch_page`, `search_docs`, `fetch_doc` |
| `GITHUB_APP_ID` | `GitHubCapability._build_live_client` | every `GitHubCapability` action |
| `GITHUB_INSTALLATION_ID` | same | same |
| `GITHUB_PRIVATE_KEY_PEM` | same (also accepts `private_key_path` kwarg) | same |

A capability whose env var is missing logs a one-line warning at
agent startup and returns clean error dicts when invoked.
