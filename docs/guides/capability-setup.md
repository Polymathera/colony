# Capability operator setup

This guide tells the **service provider** running a Colony deployment
what to configure on the **deployment host** (env vars, external
accounts, Docker socket, optional volume mounts) so that the five
new agent capabilities work end-to-end.

> Some capabilities (`GitHubCapability` in particular) also require
> per-tenant work by a **tenant admin** and per-user work by an
> **end user** — see [`github-app-setup.md`](github-app-setup.md)
> and [`connect-github.md`](connect-github.md) for the three-role
> split. This page covers the service-provider piece only.

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
| [`GitHubCapability`](../architecture/github-capability.md) | `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY_PEM`, optionally `GITHUB_APP_CLIENT_ID` + `GITHUB_APP_CLIENT_SECRET` for the per-user OAuth flow | GitHub App registration; per-tenant App installation set via the dashboard (not env) — see [`github-app-setup.md`](github-app-setup.md) |

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
GITHUB_PRIVATE_KEY_PEM="-----BEGIN RSA PRIVATE KEY-----
MIIEow...
-----END RSA PRIVATE KEY-----
"
# Optional — only needed for the "Connect GitHub" user OAuth flow.
GITHUB_APP_CLIENT_ID=Iv1.abc...
GITHUB_APP_CLIENT_SECRET=...
```

The per-tenant `installation_id` is **not** set in env — it's stored per-tenant in Postgres (set by the tenant admin via the dashboard's Tenant GitHub Installation panel). See [`github-app-setup.md`](github-app-setup.md).

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

Uses GitHub App auth — not personal access tokens. The full setup
(register the App, set the env vars, install the App per tenant,
wire the per-user OAuth flow) is documented in
[`github-app-setup.md`](github-app-setup.md). The summary for this
page:

- **Service provider** (you, on this host): set `GITHUB_APP_ID`,
  `GITHUB_PRIVATE_KEY_PEM`, and (for the per-user "Connect GitHub"
  flow) `GITHUB_APP_CLIENT_ID` + `GITHUB_APP_CLIENT_SECRET`. The
  shape is shown in the `.env` block earlier on this page;
  [`github-app-setup.md`](github-app-setup.md) §2 has the rotation
  + private-key handling.
- **Tenant admin**: installs the App into the tenant's GitHub org
  and pastes the resulting installation id into the dashboard's
  **Tenant GitHub Installation** panel —
  [`github-app-setup.md`](github-app-setup.md) §3.
- **End user**: clicks **Connect GitHub** on their profile —
  [`connect-github.md`](connect-github.md).

### Audit + rate limits

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
| `GITHUB_PRIVATE_KEY_PEM` | same (also accepts `private_key_path` kwarg) | same |
| `GITHUB_APP_CLIENT_ID` | `routers/github_oauth.py::github_connect` | `GET /auth/github/connect`, `GET /auth/github/callback` |
| `GITHUB_APP_CLIENT_SECRET` | `routers/github_oauth.py::github_callback` | `GET /auth/github/callback` |

The per-tenant `installation_id` is **not** an env var — it's
populated per-tenant in Postgres (`tenants.github_installation_id`),
set via the dashboard. `GitHubCapability._build_live_client` reads
it from `agent.metadata.parameters["github_identity"]["tenant_installation_id"]`
(threaded by the session-create handler).

A capability whose env var is missing logs a one-line warning at
agent startup and returns clean error dicts when invoked.
