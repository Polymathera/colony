# Cloud-Agnostic Storage Refactoring Plan

## Context

Colony's storage layer (`colony/python/colony/distributed/stores/`) is deeply coupled to AWS services: S3, DynamoDB, EFS, RDS, Secrets Manager, EC2, CloudWatch. Every store module has `import boto3` at the top level, meaning the package crashes on import if `boto3` is not installed. This blocks the `colony-env` local deployment tool and prevents anyone from running Colony without an AWS account.

**Goal**: Make each storage subsystem pluggable with a cloud-agnostic interface, driven by YAML config, following the existing `StateStorageBackend` + factory pattern already used for state management. Local mode should work with zero AWS dependencies.

**User decisions**:
- Scope: Essential backends only — local filesystem, local JSON (file-based), PostgreSQL (no Secrets Manager), skip cold storage & object storage locally
- Auth module: Ignore (not in this repo) — remove dead import
- Relational: Add PostgreSQL to docker-compose, password from env var directly
- boto3: Make optional via `aws` extras group in pyproject.toml

---

## AWS Dependency Inventory

| Module | AWS Services | boto3 import | Local Alternative |
|--------|-------------|--------------|-------------------|
| `objects.py` | S3 | Top-level (line 5) | Skip in local mode (disable via config) |
| `json.py` | DynamoDB | Top-level (line 7) | Filesystem JSON files |
| `git.py` | S3 + DynamoDB (GitColdStorage) | Top-level (line 16) | Skip cold storage in local mode |
| `databases.py` | RDS + Secrets Manager | Top-level (line 20) | Same PostgreSQL, password from env var |
| `files.py` | EFS + EC2 + CloudWatch (aiobotocore) | Top-level (line 12) | Local filesystem (already has fallback) |
| `storage.py` | None directly, but imports non-existent `auth.py` | Dead import (line 14) | Remove import |

**Additional bug**: `databases.py:12` has `from polymathera.schema.vmr import ...` — wrong package name, should be relative import.

---

## Architecture

Follow the existing pattern in `state_base.py` / `state_redis.py` / `state_etcd.py` / `configs.py:232-280`:

```
Interface (ABC)  →  Implementation (AWS / Local)  →  Factory  →  Config enum selects backend
```

### New Enums (in `configs.py`)

```python
class FileSystemBackendType(str, Enum):
    LOCAL = "local"      # Local filesystem (no mount required)
    EFS = "efs"          # AWS EFS mount

class JsonStorageBackendType(str, Enum):
    LOCAL = "local"      # Filesystem JSON files
    DYNAMODB = "dynamodb" # AWS DynamoDB

class ObjectStorageBackendType(str, Enum):
    DISABLED = "disabled" # No object storage (local mode)
    S3 = "s3"            # AWS S3

class GitColdStorageBackendType(str, Enum):
    DISABLED = "disabled" # No cold storage (local mode)
    S3 = "s3"            # AWS S3 + DynamoDB metadata
```

### Subsystem-by-Subsystem Design

#### 1. `FileSystem` (`files.py`)

**Already abstract**: `FileSystemInterface` ABC exists (lines 27-108) with 16 abstract methods.

**Problem**: `import aiobotocore.session` at module top (line 12) crashes without aiobotocore. Only `ScalableDistributedFileSystem1` (line 559+) uses it.

**Fix**:
- Move `import aiobotocore.session` inside `ScalableDistributedFileSystem1.__init__()`
- Extract `LocalFileSystem(FileSystemInterface)` from the existing fallback logic in `ScalableDistributedFileSystem.initialize()` (lines 282-292) — the fallback already uses plain Path operations on `/tmp/polymathera_efs_stub`
- Add `backend: FileSystemBackendType` to `DistributedFileSystemConfig` (default `"efs"` for backward compat)
- `Storage.initialize()` checks config to instantiate `LocalFileSystem` or `ScalableDistributedFileSystem`

**`LocalFileSystem` implementation**: Thin class using standard `pathlib.Path` operations. Root path from config field `local_root_path` (default `/tmp/colony_storage`). Reuses the same hash-based namespace directory structure from `ScalableDistributedFileSystem._get_path_for_namespace()`. All methods are trivial — `Path.exists()`, `Path.read_bytes()`, `Path.write_bytes()`, `shutil.rmtree()`, etc.

#### 2. `JsonStorage` (`json.py`)

**Current**: DynamoDB + Redis cache. `import boto3` at top.

**Fix**:
- Create `JsonStorageInterface` ABC with methods: `save(data, metadata)`, `load(metadata)`, `contains(metadata)`, `initialize()`, `cleanup()`
- Rename current `JsonStorage` → `DynamoDBJsonStorage(JsonStorageInterface)` (keep in same file, guarded by lazy boto3 import)
- Create `LocalJsonStorage(JsonStorageInterface)` — stores JSON files on disk at `{root}/{md5_hash}.json`, uses Redis cache if available (same pattern as DynamoDB version)
- Add `backend: JsonStorageBackendType` to `JsonStorageConfig`
- Guard `import boto3` with try/except, only fail if backend is `DYNAMODB` and boto3 is missing

#### 3. `ObjectStorage` (`objects.py`)

**Current**: S3 only. `import boto3` at top.

**Fix**:
- Add `backend: ObjectStorageBackendType` to `ObjectStorageConfig` (default `"disabled"` when running locally)
- Guard `import boto3` with try/except
- When `backend == "disabled"`: `ObjectStorage.initialize()` becomes a no-op, all methods raise `RuntimeError("Object storage is disabled")`
- When `backend == "s3"`: Current behavior
- No need for a local filesystem object storage — it's not used in the local test flow

#### 4. `GitColdStorage` (in `git.py`)

**Current**: S3 + DynamoDB for cold storage of git repos. `import boto3` at top.

**Fix**:
- Add `enable_cold_storage: bool = True` to `GitColdStorageConfig` (replaces the need for a full ABC — cold storage is either on or off)
- Guard `import boto3` with try/except in `git.py`
- Create `NoOpGitColdStorage` class with the same interface as `GitColdStorage`:
  - `repository_exists()` → always `False`
  - `store_repository()` → no-op
  - `retrieve_repository()` → raise `FileNotFoundError`
  - `delete_repository()` → no-op
  - `initialize()` / `cleanup()` → no-op
  - `get_all_repo_metadata()` → `[]`
  - `repair_metadata_once()` → no-op
- `GitFileStorage.initialize()` instantiates `NoOpGitColdStorage` when `enable_cold_storage=False`, otherwise `GitColdStorage` (current code)
- This avoids the massive refactoring of extracting a full cold storage ABC — pragmatic approach

#### 5. `RelationalStorage` (`databases.py`)

**Current**: PostgreSQL (asyncpg) + AWS Secrets Manager for password. `import boto3` at top. Also has wrong import: `from polymathera.schema.vmr import ...`

**Fix**:
- Fix import: `from polymathera.schema.vmr import ...` → `from ...schema.vmr import ...`
- Guard `import boto3` with try/except
- Add `db_password: str | None = Field(default=None, json_schema_extra={"env": "RDS_PASSWORD"})` to `RelationalStorageConfig`
- Modify `_resolve_db_password()`:
  1. If `self.config.db_password` is set → use it directly (local mode)
  2. Elif `self.config.db_password_secret_arn` is set and boto3 available → fetch from Secrets Manager (cloud mode)
  3. Else → return `None` (will fail at connection time with clear error)
- Make `db_password_secret_arn` optional: change `Field(...)` to `Field(default=None, ...)`
- Add PostgreSQL service to docker-compose.yml

#### 6. `Storage` orchestrator (`storage.py`)

**Fix**:
- Remove dead import: `from .auth import AuthToken, DistributedAuthManager` (line 14)
- Remove `AuthToken` from method signatures — replace with `str` (actor_id) or remove auth methods entirely since auth module doesn't exist
- Actually, since auth is gated by `enable_auth=False` (default), and the module doesn't exist, the simplest fix is to make the import conditional and remove `DistributedAuthManager` from `__init__`
- Use config to instantiate correct backends:
  ```python
  # Filesystem
  if self.config.distributed_file_system.backend == FileSystemBackendType.LOCAL:
      self.distributed_file_system = LocalFileSystem(self.config.distributed_file_system)
  else:
      self.distributed_file_system = ScalableDistributedFileSystem(self.config.distributed_file_system)
  ```
  Similar pattern for JSON, object, git storage.

---

## Files to Modify

| # | File | Changes |
|---|------|---------|
| 1 | `distributed/configs.py` | Add 4 backend type enums, add `backend` fields to configs, add `db_password` field, make `db_password_secret_arn` optional, add `enable_cold_storage`, add `local_root_path` |
| 2 | `distributed/stores/files.py` | Move `aiobotocore` import inside `ScalableDistributedFileSystem1.__init__()`, extract `LocalFileSystem` class |
| 3 | `distributed/stores/json.py` | Guard `boto3` import, create `JsonStorageInterface` ABC, rename current class to `DynamoDBJsonStorage`, create `LocalJsonStorage` |
| 4 | `distributed/stores/objects.py` | Guard `boto3` import, add disabled mode |
| 5 | `distributed/stores/git.py` | Guard `boto3` import, create `NoOpGitColdStorage`, use `enable_cold_storage` config |
| 6 | `distributed/stores/databases.py` | Fix `polymathera` import, guard `boto3` import, add `db_password` direct config, make `db_password_secret_arn` optional |
| 7 | `distributed/storage.py` | Remove dead auth import, use config-driven backend selection for filesystem/json/object/git |
| 8 | `colony/pyproject.toml` | Add `aws` extras group with `boto3`, `botocore`, `aiobotocore`; remove them from required deps |
| 9 | `cli/deploy/docker/docker-compose.yml` | Add PostgreSQL service |

## New Files to Create

| # | File | Contents |
|---|------|----------|
| 1 | (none) | All new classes go into existing files to avoid file bloat |

`LocalFileSystem` goes in `files.py`, `LocalJsonStorage` goes in `json.py`, `NoOpGitColdStorage` goes in `git.py`. No new files needed.

---

## Implementation Order

### Phase 1: Configs & Dependencies (no behavior change)

**Step 1**: `pyproject.toml` — Add `aws` extras group
```toml
aws = ["boto3", "botocore", "aiobotocore"]
```
Remove `boto3` from implicit required deps (currently not listed as required — it's used but not declared, relying on transitive deps). Add `aiobotocore` as optional.

**Step 2**: `configs.py` — Add enums and new config fields
- Add `FileSystemBackendType`, `JsonStorageBackendType`, `ObjectStorageBackendType`, `GitColdStorageBackendType` enums
- Add `backend` field to `DistributedFileSystemConfig` (default `"efs"`)
- Add `local_root_path` field to `DistributedFileSystemConfig` (default `/tmp/colony_storage`)
- Add `backend` field to `JsonStorageConfig` (default `"dynamodb"`)
- Add `backend` field to `ObjectStorageConfig` (default `"s3"`)
- Add `enable_cold_storage` field to `GitColdStorageConfig` (default `True`)
- Add `db_password` field to `RelationalStorageConfig` (default `None`, env `RDS_PASSWORD`)
- Make `db_password_secret_arn` default to `None` instead of required (`...`)
- Make all `RelationalStorageConfig` fields default to `None` instead of required (for local mode they come from env vars)

### Phase 2: Guard all boto3 imports

**Step 3**: `files.py` — Move `import aiobotocore.session` from line 12 into `ScalableDistributedFileSystem1.__init__()`:
```python
# At top level: remove `import aiobotocore.session`
# Inside ScalableDistributedFileSystem1.__init__():
try:
    import aiobotocore.session
    self.session = aiobotocore.session.get_session()
except ImportError:
    raise ImportError("aiobotocore is required for ScalableDistributedFileSystem1. Install with: pip install aiobotocore")
```

**Step 4**: `objects.py` — Guard boto3:
```python
from __future__ import annotations
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
```
Add check in `initialize()`: if backend is `"s3"` and `boto3 is None`, raise ImportError.

**Step 5**: `json.py` — Guard boto3 (same pattern)

**Step 6**: `git.py` — Guard boto3 (same pattern)

**Step 7**: `databases.py` — Guard boto3, fix import:
```python
# Fix: from polymathera.schema.vmr → from ...schema.vmr
from ...schema.vmr import (
    VirtualMonorepo,
    Repository,
    RepositoryDependency,
    VMRRepositoryLink,
)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
```

### Phase 3: Local implementations

**Step 8**: `files.py` — Add `LocalFileSystem(FileSystemInterface)`:
- Constructor takes `DistributedFileSystemConfig`
- `initialize()`: creates root dir at `config.local_root_path`, logs
- All methods: standard pathlib/shutil operations
- Reuse `_get_path_for_namespace()` hash logic from `ScalableDistributedFileSystem`
- ~100 lines

**Step 9**: `json.py` — Add `LocalJsonStorage`:
- Stores at `{root}/json_storage/{md5_hash}.json`
- `save(data, metadata)`: write JSON file + set Redis cache (if available)
- `load(metadata)`: check Redis cache → read JSON file
- Root path from config field `local_storage_path` (add to `JsonStorageConfig`)
- ~60 lines

**Step 10**: `git.py` — Add `NoOpGitColdStorage`:
- All methods are no-ops or return empty/False
- `repository_exists()` → `False`
- `store_repository()` → log + no-op
- ~30 lines

**Step 11**: `databases.py` — Modify `_resolve_db_password()`:
```python
def _resolve_db_password(self) -> str | None:
    # 1. Direct password from config/env
    if self.config.db_password:
        return self.config.db_password

    # 2. AWS Secrets Manager
    arn = self.config.db_password_secret_arn
    if arn and arn != "" and not arn.startswith("placeholder"):
        if boto3 is None:
            raise ImportError("boto3 is required for AWS Secrets Manager password resolution")
        sm = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION", "us-east-1"))
        # ... existing Secrets Manager code ...

    return None
```

### Phase 4: Wire up in Storage orchestrator

**Step 12**: `storage.py` — Config-driven backend selection:
- Remove `from .auth import AuthToken, DistributedAuthManager`
- Remove `self.auth_manager` and `_verify_access` usage (or stub `AuthToken = Any`)
- In `initialize()`:
  ```python
  # Filesystem
  from .stores.files import LocalFileSystem, ScalableDistributedFileSystem
  if config.distributed_file_system and config.distributed_file_system.backend == "local":
      self.distributed_file_system = LocalFileSystem(config.distributed_file_system)
  else:
      self.distributed_file_system = ScalableDistributedFileSystem(config.distributed_file_system)

  # Object storage
  if config.object_storage and config.object_storage.backend != "disabled":
      self.object_storage = ObjectStorage(config.object_storage)
  # else: self.object_storage remains None

  # JSON storage
  if config.json_storage and config.json_storage.backend == "local":
      self.json_storage = LocalJsonStorage(config.json_storage)
  else:
      self.json_storage = JsonStorage(config.json_storage)  # DynamoDB

  # Git storage — cold storage controlled by enable_cold_storage in sub-config
  # (handled inside GitFileStorage.initialize())
  ```
- Guard methods that use `self.object_storage` or `self.auth_manager` with `if self.X is not None` checks

### Phase 5: Docker Compose & Environment

**Step 13**: `docker-compose.yml` — Add PostgreSQL service:
```yaml
  postgres:
    image: postgres:16-alpine
    container_name: colony-postgres
    environment:
      - POSTGRES_USER=colony
      - POSTGRES_PASSWORD=colony_dev
      - POSTGRES_DB=colony
    ports:
      - "${COLONY_POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - colony-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U colony"]
      interval: 5s
      timeout: 3s
      retries: 5
```
Add `postgres-data` to volumes. Add PostgreSQL env vars to ray-head/ray-worker:
```yaml
- RDS_HOST=postgres
- RDS_PORT=5432
- RDS_USER=colony
- RDS_PASSWORD=colony_dev
- RDS_DB_NAME=colony
- POLYMATHERA_RUNNING_LOCALLY=true
```

**Step 14**: Set local-mode defaults in docker-compose environment:
```yaml
# Filesystem
- EFS_MOUNT_PATH=/mnt/shared
# Object storage disabled
- OBJECT_STORAGE_BACKEND=disabled
# JSON storage local
- JSON_STORAGE_BACKEND=local
# Cold storage disabled
- GIT_COLD_STORAGE_ENABLED=false
```

---

## Verification

1. **Import test**: `python -c "from colony.distributed.storage import Storage"` should succeed without boto3 installed
2. **Config test**: Create a YAML config with local backends, verify `check_or_get_component` returns correct config
3. **Docker test**: `colony-env up` → `colony-env run /path/to/repo` — full integration test with PostgreSQL, Redis, local filesystem, no AWS
4. **Backward compat**: Existing cloud deployments continue to work with current configs (all defaults are the current cloud values)
5. **Module-level**: Each modified file should be importable in isolation without boto3: `python -c "from colony.distributed.stores.git import GitFileStorage"` etc.

---

## YAML Config Example (Local Mode)

```yaml
distributed:
  storage:
    enable_auth: false
    object_storage:
      backend: disabled
    json_storage:
      backend: local
      local_storage_path: /mnt/shared/json_storage
    distributed_file_system:
      backend: local
      local_root_path: /mnt/shared/filesystem
    git_storage:
      cold_storage_config:
        enable_cold_storage: false
    relational_storage:
      db_host: postgres
      db_port: 5432
      db_user: colony
      db_password: colony_dev
      db_name: colony
```
