# `SandboxedShellCapability`

Lets agents launch curated Docker containers, execute shell commands and named scripts inside them, copy files in and out, and share a container with peer agents — all as `@action_executor` methods.

Code: `polymathera.colony.agents.patterns.capabilities.SandboxedShellCapability`. Subpackage: `_sandbox/{backend,registry}.py`. Image registry: `colony/cli/deploy/docker/sandbox-images.yaml`.

## Why it exists

[`REPLCapability`](action-policies.md) gives agents an in-process Python sandbox but deliberately blocks `subprocess`, `open`, `eval`, `exec`. There was no agent-driven path to run a real shell command — `git clone`, `pyright`, `pytest`, `ruff`, anything domain-specific.

`SandboxedShellCapability` is that path. It is also the load-bearing dependency of [`UserPluginCapability`](user-plugin-capability.md), which delegates every skill execution to it.

## Action surface

| Group | Actions |
|-------|---------|
| Lifecycle | `launch_container`, `stop_container`, `restart_container`, `list_containers`, `get_container_status` |
| Sharing | `attach_container`, `detach_container` |
| Execution | `execute_command`, `execute_script` |
| Registry inspection | `list_images`, `list_scripts` |
| File transfer | `copy_file_in`, `copy_file_out`, `read_file`, `write_file` |

15 actions total. Every method returns a uniform `{container_id, ..., message}` shape. Failures degrade to error dicts (the LLM observes them as data).

### Launching

`launch_container(image_role=…, cpu_limit=…, memory_limit_mb=…, max_wall_time_seconds=…, network_mode=…, shared=…, extra_volumes=…)`

The caller picks an image by **role label** from the registry, never by raw image name. A misconfigured agent cannot run an untrusted image — operators control the image set. The per-session host workspace is bind-mounted at `workdir` (default `/workspace`) so files persist across multiple `execute_command` calls.

### Executing

```python
execute_command(container_id, ["ruff", "check", "src/"], timeout_seconds=60)
execute_command(container_id, "git status", stream_to_blackboard=True)
execute_script(container_id, "lint_python", args={"path": "src/app.py"})
```

A list `command` is passed verbatim. A string is wrapped in `bash -lc` so shell features (pipes, redirection) work. `execute_script` resolves a name from the registry, validates `args` against the script's declared params, and substitutes `{name}` placeholders before exec.

`stream_to_blackboard=True` publishes stdout/stderr chunks under `shell:stream:{container_id}:{exec_id}:{seq}` and a final `shell:exec:{exec_id}:complete`. Other agents (or the dashboard) tail those keys for live output.

### Sharing

`launch_container(shared=True)` writes a shareable container. Peer agents in the same session call `attach_container(container_id)` — afterwards they can `execute_command` against it but cannot stop or restart it (only the owner can). `detach_container` drops the attachment.

## Image registry

YAML at `colony/cli/deploy/docker/sandbox-images.yaml`, mounted into ray-head/ray-worker at `/etc/colony/sandbox-images.yaml:ro`:

```yaml
images:
  - role: default
    image: python:3.11-slim
    description: Minimal Python 3.11 environment.
    scripts:
      - name: python_version
        description: Report the Python version.
        params: {}
        cmd: ["python", "-V"]
      - name: run_python
        description: Execute a Python one-liner.
        params:
          code: { type: string, required: true }
        cmd: ["python", "-c", "{code}"]
```

Operators add roles by editing the file and reloading the agent (or — once the Settings UI lands — through the dashboard). Production deployments pin images by digest:

```yaml
image: polymathera/sandbox-code-analysis@sha256:abc123…
```

## Backend abstraction

```python
class ContainerBackend(ABC):
    async def launch(self, spec: ContainerSpec) -> ContainerHandle: ...
    async def stop(self, handle, *, timeout_s=10): ...
    async def restart(self, handle): ...
    async def is_running(self, handle) -> bool: ...
    async def inspect(self, handle) -> dict: ...
    async def exec(self, handle, cmd, *, timeout_seconds, env=None, workdir=None, stdin=None) -> ExecResult: ...
    def exec_stream(self, handle, cmd, *, timeout_seconds, env=None, workdir=None, stdin=None): ...  # async iterator
    async def copy_in(self, handle, *, src_host_path, dst_container_path): ...
    async def copy_out(self, handle, *, src_container_path, dst_host_path): ...
    async def list_by_label(self, labels) -> list[dict]: ...
```

Default: `DockerCLIBackend` — shells out to `docker` via `asyncio.create_subprocess_exec`. No `aiodocker` dependency on the critical path. Future: `AiodockerBackend` (streaming, real exit codes, async cleanup), `KubernetesBackend` (pods + ephemeral containers).

## Security posture

- **Image trust**: registry-only. Unknown roles rejected before any Docker call.
- **Container hardening**: every launch passes `--cap-drop=ALL --security-opt no-new-privileges`.
- **Network**: defaults to `bridge` — outbound internet OK, internal services blocked. `none` for offline work; `host` is opt-in only.
- **Per-agent caps**: `max_concurrent_containers`, `max_total_cpu_cores`, `max_total_memory_mb` enforced before launch.
- **Wall-time killer**: each container's `max_wall_time_seconds` is enforced by an asyncio task that calls `stop_container` on expiry.
- **Audit log**: every `execute_command` writes `audit:shell:{ts}:{uuid}` with tenant_id, session_id, agent_id, container_id, image, command, exit_code, wall_time_ms, stdout/stderr sizes (not content).

## Docker daemon access

The capability talks to Docker via the local socket, mounted into ray-head and ray-worker in `docker-compose.yml`:

```yaml
ray-head:
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
    - ./sandbox-images.yaml:/etc/colony/sandbox-images.yaml:ro
```

This is the **dev default**. Anything inside ray-head can effectively root the host through the socket — production should configure `DOCKER_HOST=tcp://host:2376` with TLS certs to talk to a hardened remote daemon.

## Configuration

```python
SandboxedShellCapability.bind(
    scope=BlackboardScope.SESSION,
    backend=DockerCLIBackend(),                # or your own
    registry_path="/etc/colony/sandbox-images.yaml",
    host_workspace_root="/mnt/shared/workspaces",
    default_network_mode="bridge",
    max_concurrent_containers=4,
    max_total_cpu_cores=4.0,
    max_total_memory_mb=8192,
    audit_enabled=True,
)
```

Wired into the session agent. Coordinator agents that need to run domain tools add it via YAML.

## Test surface

`tests/test_sandboxed_shell_capability.py` (27 tests). All Docker calls go through a fake `ContainerBackend` so no daemon is needed. Covers: registry parsing + malformed-entry tolerance, role rejection, label wiring + workspace mount, per-agent caps (concurrent/CPU), invalid network mode, ownership enforcement, `NoSuchContainer` swallowing, list filtering, status reporting, command-string-vs-list wrapping, output truncation, audit writes, streaming chunks + complete record, script param validation + template substitution, attach gating, file transfer delegation.

## Open follow-ups

- **Settings UI** for image registry editing.
- **Per-tenant resource quotas** wired into the existing `TenantQuota` system.
- **`AiodockerBackend`** for accurate streaming exit codes and lower-overhead exec.
- **`KubernetesBackend`** for production deployments without daemon-socket access.
- **`command_validator` hook** ecosystem for plugging in custom guardrails.
