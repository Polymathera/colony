# Image extensions (L1-G)

How operators add Python packages and arbitrary setup steps to a Colony runtime — without forking the Colony codebase or re-publishing the image.

## TL;DR

The cluster YAML carries three optional fields that compose at container start (mirroring [Ray's cluster YAML][ray-yaml]):

```yaml
cluster:
  docker:
    image: polymathera/colony:0.3.0          # optional override; default = colony:local

  extensions:
    packages:
      - { name: polymathera-cps, version: "0.1.0", extras: [quantum] }
      - { name: polymathera-cps, source: path, path: ../cps }   # dev-mode path source

  setup_commands: []           # general (every node)
  head_setup_commands: []      # ray-head only
  worker_setup_commands: []    # ray-worker only
```

`colony-env up --config <yaml>` reads these fields, builds the runtime image if necessary, writes a JSON snapshot the container-start hook reads, and runs `docker compose up`. Inside each colony service, the hook `pip install`s the resolved packages into a persistent overlay volume (cache-keyed by a hash of the resolved spec) and runs the `setup_commands`. Re-`up`-ping with the same YAML hits the overlay cache.

[ray-yaml]: https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html

## Why this exists

Colony's three entry-point groups (`polymathera.mission_types`, `polymathera.cli_extensions`, `polymathera.config_components`) discover extensions via `importlib.metadata.entry_points` — environment-wide. Without `cluster.extensions.packages` in the YAML, an extension package's entries are invisible at runtime because the package is not installed in the container's Python environment. L1-G fixes that gap.

## Surface anatomy

### `cluster.docker.image` — runtime image override

Optional. Defaults to `colony:local` (the locally-built runtime). Override when you need:

- A version-pinned production image (`polymathera/colony:0.3.0`).
- A custom image you built `FROM polymathera/colony-base:<version>` to bundle system-level deps (CUDA toolkit, vendor SDKs, …).
- An air-gapped or signed image mirrored to your registry.

The compose file substitutes `${COLONY_IMAGE}` at parse time; the `colony-env up` CLI sets it from the YAML or `--bake` snapshot.

### `cluster.extensions.packages` — Python extensions

A list of `PackageSpec` entries. Two source variants:

- `source: version` (default): pip-installed as `name[extras]<op><version>`. Pip-compatible operators (`==`, `>=`, `~=`, `!=`, …) pass through; a bare `version` becomes `==`. Poetry-style `^` / `~`-with-caret operators are NOT translated and pip will reject them.
- `source: path`: pip-installed from a local directory. The path is resolved relative to the YAML file. Mirrors how Colony itself is installed in the base image (`poetry install` from source).

The container-start hook installs the resolved list into `/opt/colony-overlay` (a persistent Docker volume). The hash of the resolved spec is the cache key — re-running with the same YAML hits the cache; changing any field invalidates and reinstalls.

### `cluster.setup_commands` (and `head_`/`worker_` variants) — escape hatch

Arbitrary shell run on every container start, AFTER package install and BEFORE the service command (`ray start` / `dashboard` main). Use *only* when `extensions.packages` cannot — system-level installs, S3 downloads, EFS mounts, vendor SDK installers, custom Ray resource registration.

The general `setup_commands` runs on every node; the role-specific variants fire only on the matching role.

`colony-env up` flags any `pip install` inside `setup_commands` as a warning. Pip-installable packages belong in `extensions.packages` where they pick up overlay caching, hash-keyed reinstall, and the distinct-from-`SandboxedShellCapability` validation pipeline.

## Composition order

```
docker pull (or use local) cluster.docker.image
  └─ container starts
      └─ container-start hook (cluster-runtime-hook.sh):
          1. read /etc/colony/cluster-runtime.json
          2. if hash matches /opt/colony-overlay/.installed-hash → skip
             else: pip install --target=/opt/colony-overlay <resolved pip_args>
          3. run setup_commands
          4. run {head,worker}_setup_commands (role-dependent)
          5. exec the service command (ray start / dashboard main)
```

`PYTHONPATH` gets `/opt/colony-overlay` prepended so installed packages override any base-image-bundled copies.

## Default fast path vs `--bake`

- **Fast path (default):** packages overlay-installed at container start. No image rebuild on extension changes. Recommended for development.

- **`colony-env up --bake`:** snapshots the resolved YAML into a pinned `colony-local:<hash>` image (packages installed at build time, hash file pre-populated so the runtime hook short-circuits). Slower up, faster steady-state, fully reproducible. Recommended for production / multi-node clusters where the `setup_commands` cost multiplies.

`setup_commands` ALWAYS run at container start, regardless of `--bake` — they're the operator's "every-boot" hook.

## Two-stage Docker build

Refactored from a monolithic `Dockerfile.local`:

- **`polymathera/colony-base:<version>`** (Dockerfile.base) — heavy deps pinned to a Colony release: `rayproject/ray` base + system packages + Node + Linguist + Python dependency tree from pyproject.toml/lock. Excludes Colony's own source. Publishable.

- **`polymathera/colony:<version>`** / `colony:local` (Dockerfile.local) — `FROM polymathera/colony-base:<version>` + Colony source + dashboard frontend build + permission fixes. Builds fast on top of cached base.

Users wanting maximum control build their own:

```dockerfile
FROM polymathera/colony-base:0.3.0
RUN pip install polymathera-colony==0.3.0 polymathera-cps[quantum]==0.1.0 my-private-pkg
```

…then point colony-env at that image:

```yaml
cluster:
  docker:
    image: my-org/colony:custom-build
```

## Path-source bind mounts

When `source: path` packages are configured, `colony-env up` writes a sidecar compose override (`docker-compose.path-extensions.yml` under `.runtime/`) bind-mounting each host path to `/mnt/path-extensions/<package-name>` of every colony service. The container-start hook then `pip install`s from those in-container paths.

This is what makes `{name: polymathera-cps, source: path, path: ../cps}` "just work" for solo-dev iteration on CPS itself — edits to `../cps/` show up at next `colony-env up` without a version bump.

`--bake` does NOT yet support path sources (it would require copying host paths into the bake build context). Use the default fast path for path-source workflows.

## GPU overlay

Optional sidecar `docker-compose.gpu.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Reserves all visible NVIDIA GPUs for ray-head and ray-worker. Requires the NVIDIA Container Toolkit on the host.

## Subcommands

- **`colony-env image-info`** — list polymathera-* packages installed in the running cluster, distinguishing baked (in the runtime image) vs overlay-installed (from `extensions.packages`). Useful for debugging "why isn't my CPS entry showing up?".
- **`colony-env image-build [--config X] [--bake]`** — build the base + runtime images (and optionally a bake image) WITHOUT bringing the cluster up. Useful for CI / pre-staging.

## File map

```
colony/src/polymathera/colony/cli/deploy/
├── extensions.py                   # Pydantic schema + resolver + lint
├── runtime_writer.py               # writes cluster-runtime.json + path overlay
├── cli.py                          # adds --bake, image-info, image-build
├── providers/
│   ├── base.py                     # adds bake/image_info/image_build to ABC
│   └── compose.py                  # _build_base_image, _build_bake_image, up()
└── docker/
    ├── Dockerfile.base             # publishable base image
    ├── Dockerfile.local            # FROM base + Colony source
    ├── docker-compose.yml          # parameterized (COLONY_IMAGE, COLONY_SHM_SIZE)
    ├── docker-compose.gpu.yml      # optional GPU overlay
    ├── cluster-runtime-hook.sh     # container-start hook
    └── .runtime/                   # generated per-up: cluster-runtime.json + path overrides
```

## Approval-gate decisions

The L1-G design landed three sign-off questions (alignment plan §6 gates 9–11). The current implementation realizes:

- **Gate 9** (YAML-only — drop manifest `imports_packages` and `--install` flag): YES, implemented as designed.
- **Gate 10** (base image registry — Docker Hub / GHCR / ECR Public): publishing not yet wired; locally-built `polymathera/colony-base:local` for development. Choose the registry when the open-source release lands.
- **Gate 11** (fast-path vs `--bake` default): fast path is the default; `--bake` is opt-in for production/reproducibility.
