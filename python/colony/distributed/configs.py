from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import Field

from .caching.simple import CacheConfig
from .config import ConfigComponent, register_polymathera_config
from .redis_utils.client import RedisConfig


@register_polymathera_config()
class ObjectStorageConfig(ConfigComponent):
    region_name: str | None = Field(default=None)
    access_key: str | None = Field(default=None)
    secret_key: str | None = Field(default=None)

    CONFIG_PATH: ClassVar[str] = "distributed.object_storage"



@register_polymathera_config()
class JsonStorageConfig(ConfigComponent):
    cache_ttl: int = Field(default=3600)
    dynamodb_table: str | None = Field(default="JsonStorage", json_schema_extra={"env": "JSON_STORAGE_DDB_TABLE_NAME"})
    cache_config: CacheConfig | None = Field(default=None)
    aws_region: str = Field(default="us-east-1", json_schema_extra={"env": "AWS_REGION"})

    CONFIG_PATH: ClassVar[str] = "json_storage"



@register_polymathera_config()
class GitCacheManagerConfig(ConfigComponent):
    cache_ttl: int = Field(default=3600)
    lock_ttl: int = Field(default=300)
    clone_cache_config: CacheConfig | None = Field(default=None)
    operations_cache_config: CacheConfig | None = Field(default=None)
    references_cache_config: CacheConfig | None = Field(default=None)
    locks_cache_config: CacheConfig | None = Field(default=None)

    CONFIG_PATH: ClassVar[str] = "distributed.git_cache_manager"


@register_polymathera_config()
class GitColdStorageConfig(ConfigComponent):
    s3_buckets: list[str] = Field(
        json_schema_extra={"env": "GIT_COLD_STORAGE_S3_BUCKETS", "parser": lambda x: x.split(",")}
    )
    repo_metadata_table: str = Field(default="RepoMetadata", json_schema_extra={"env": "REPO_METADATA_DDB_TABLE_NAME"})
    s3_monitor_interval: int = Field(default=3600)
    aws_region: str = Field(default="us-east-1", json_schema_extra={"env": "AWS_REGION"})

    CONFIG_PATH: ClassVar[str] = "distributed.git_cold_storage"



@register_polymathera_config()
class GitFileStorageConfig(ConfigComponent):
    max_concurrent_clones: int = 100
    prune_interval: int = 86400  # 24 hours
    gc_interval: int = 86400  # 24 hours
    namespace: str = "polymathera_git_repos_storage"
    update_threshold: int = Field(
        default=86400, description="Threshold for updating repositories"
    )
    update_interval: int = Field(
        default=3600, description="Interval for updating repositories"
    )
    cache_manager_config: GitCacheManagerConfig | None = None
    cold_storage_config: GitColdStorageConfig | None = None

    CONFIG_PATH: ClassVar[str] = "distributed.git_file_storage"


@register_polymathera_config()
class FileStorageConfig(ConfigComponent):
    namespace: str = Field(default="polymathera_file_storage")

    CONFIG_PATH: ClassVar[str] = "distributed.file_storage"



@register_polymathera_config()
class DistributedFileSystemConfig(ConfigComponent):
    efs_mount_path: str = Field(
        default="/mnt/efs",
        description="The mount path for the EFS file system.",
        json_schema_extra={"env": "EFS_MOUNT_PATH"},
    )
    compression_level: int = Field(
        default=6,
        description="The compression level for the EFS file system.",
        json_schema_extra={"env": "EFS_COMPRESSION_LEVEL", "optional": True},
    )

    CONFIG_PATH: ClassVar[str] = "distributed.file_system"



@register_polymathera_config()
class DistributedFileSystemConfig1(ConfigComponent):
    shard_count: int = 10
    regions: list[str] = ["us-east-1"]
    compression_level: int = 6
    cache_ttl: int = 3600  # 1 hour # TODO: Implement Caching
    scale_up_throughput: int = 100
    security_group_id: str = Field(default="polymathera-security-group")

    CONFIG_PATH: ClassVar[str] = "distributed.file_system1"



@register_polymathera_config()
class AuthConfig(ConfigComponent):
    jwt_secret: str = Field(default="polymathera-jwt-secret")
    permissions_table: str = Field(default="PolymatheraAuthPermissions", json_schema_extra={"env": "AUTH_PERMISSIONS_DDB_TABLE_NAME"})
    audit_table: str = Field(default="PolymatheraAudits", json_schema_extra={"env": "AUTH_AUDIT_DDB_TABLE_NAME"})
    cache_config: CacheConfig | None = Field(default=None)
    aws_region: str = Field(default="us-east-1", json_schema_extra={"env": "AWS_REGION"})

    CONFIG_PATH: ClassVar[str] = "distributed.auth"



@register_polymathera_config()
class RelationalStorageConfig(ConfigComponent):
    db_user: str                = Field(..., json_schema_extra={"env": "RDS_USER"})
    db_password_secret_arn: str = Field(..., json_schema_extra={"env": "RDS_SECRET_ARN"})
    db_host: str                = Field(..., json_schema_extra={"env": "RDS_HOST"})
    db_port: int                = Field(..., json_schema_extra={"env": "RDS_PORT"})
    db_name: str                = Field(..., json_schema_extra={"env": "RDS_DB_NAME"})

    @property
    def database_url(self) -> str:
        # Note: This is a simplified way to get the password.
        # In a real production system, you'd fetch this from Secrets Manager
        # at runtime, not expose it directly in the URL.
        # However, for the purpose of constructing the URL for SQLAlchemy,
        # this is a common pattern. The actual secret fetching should be
        # handled by the application logic that uses this URL.
        # The database URL postgresql+asyncpg:// scheme tells SQLAlchemy to use the asyncpg driver instead of psycopg2 for async operations.
        return f"postgresql+asyncpg://{self.db_user}:DB_PASSWORD_PLACEHOLDER@{self.db_host}:{self.db_port}/{self.db_name}"

    CONFIG_PATH: ClassVar[str] = "distributed.relational_storage"



@register_polymathera_config()
class StorageConfig(ConfigComponent):
    """Configuration for the Storage class"""

    object_storage: ObjectStorageConfig | None = Field(default=None)
    relational_storage: RelationalStorageConfig | None = Field(default=None)
    distributed_file_system: DistributedFileSystemConfig | None = Field(default=None)
    json_storage: JsonStorageConfig | None = Field(default=None)
    file_storage: FileStorageConfig | None = Field(default=None)
    git_storage: GitFileStorageConfig | None = Field(default=None)
    auth_config: AuthConfig | None = Field(default=None)
    enable_auth: bool = Field(default=False)

    CONFIG_PATH: ClassVar[str] = "distributed.storage"



@register_polymathera_config()
class UserChatServiceConfig(ConfigComponent):
    slack_bot_token: str = Field(default="polymathera-slack-bot-token")
    slack_app_token: str = Field(default="polymathera-slack-app-token")

    CONFIG_PATH: ClassVar[str] = "chat.user_chat_service"



@register_polymathera_config()
class ObservabilityConfig(ConfigComponent):
    """
    Configuration for the Observability class.
    NOTE: Config field names must be in all lowercase (and may contain underscores) to
    match the environment variable names and, hence, be configurable via environment variables.
    """

    service_name: str = Field(default="polymathera-agent")
    environment: str = Field(default="development")
    use_otlp: bool = Field(default=False)
    telemetry_endpoint: str = Field(default="http://jaeger:4317")
    prometheus_port: int = Field(default=8000)
    cloudwatch_namespace: str = Field(default="Polymathera")

    CONFIG_PATH: ClassVar[str] = "distributed.observability"



@register_polymathera_config()
class AuthServiceConfig(ConfigComponent):
    CONFIG_PATH: ClassVar[str] = "distributed.auth_service"



@register_polymathera_config()
class SecurityManagerConfig(ConfigComponent):
    permissions_table: str = Field(default="PolymatheraSecPermissions", json_schema_extra={"env": "SEC_MANAGER_PERMISSIONS_DDB_TABLE_NAME"})
    vmr_access_table: str = Field(default="VMRAccess", json_schema_extra={"env": "SEC_MANAGER_VMR_ACCESS_DDB_TABLE_NAME"})
    knowledge_permissions_table: str = Field(default="KnowledgePermissions", json_schema_extra={"env": "SEC_MANAGER_KNOWLEDGE_PERMISSIONS_DDB_TABLE_NAME"})

    CONFIG_PATH: ClassVar[str] = "distributed.security_manager"



@register_polymathera_config()
class ServiceRegistryConfig(ConfigComponent):
    service_registry_type: str = Field(default="etcd")
    etcd_host: str = Field(default="localhost", json_schema_extra={"env": "ETCD_HOST"})
    etcd_port: int = Field(default=2379, json_schema_extra={"env": "ETCD_PORT"})
    aws_region: str = Field(default="us-east-1", json_schema_extra={"env": "AWS_REGION"})
    load_balancer_type: str = Field(default="haproxy")

    CONFIG_PATH: ClassVar[str] = "distributed.service_registry"



@register_polymathera_config()
class SystemMessagingConfig(ConfigComponent):
    kafka_producer_config: dict[str, Any] = Field(default_factory=dict)
    kafka_consumer_config: dict[str, Any] = Field(default_factory=dict)

    CONFIG_PATH: ClassVar[str] = "distributed.system_messaging"



class StateStorageBackendType(Enum):
    """Supported storage backends for state management"""

    REDIS = "redis"
    ETCD = "etcd"


@register_polymathera_config()
class StateStorageConfig(ConfigComponent):
    """Configuration for state storage"""

    backend: StateStorageBackendType = Field(default=StateStorageBackendType.REDIS)
    namespace: str = Field(default="polymathera")
    ttl: int = 3600  # 1 hour default TTL
    max_retries: int = 3
    retry_delay: float = 0.1  # seconds

    # Redis specific - use environment variables with fallbacks
    redis_host: str = Field(
        default="localhost",
        description="Redis host to connect to",
        json_schema_extra={"env": "REDIS_HOST"},
    )
    redis_port: int = Field(
        default=6379,
        description="Redis port to connect to",
        json_schema_extra={"env": "REDIS_PORT"},
    )
    redis_db: int = 0
    redis_password: str | None = None
    redis_ssl: bool = False

    # Etcd specific - use environment variables with fallbacks
    etcd_host: str = Field(
        default="localhost",
        description="Etcd host to connect to",
        json_schema_extra={"env": "ETCD_HOST"},
    )
    etcd_port: int = Field(
        default=2379,
        description="Etcd port to connect to",
        json_schema_extra={"env": "ETCD_PORT"},
    )
    etcd_timeout: int = 5  # seconds
    etcd_ssl: bool = False
    etcd_ca_cert: str | None = None
    etcd_cert_key: str | None = None

    CONFIG_PATH: ClassVar[str] = "distributed.state_storage"



@register_polymathera_config()
class DistributedStateConfig(ConfigComponent):
    """Configuration for distributed state management across the system"""

    # Storage configuration
    storage: StateStorageConfig = Field(
        default_factory=lambda: StateStorageConfig(
            backend=StateStorageBackendType.REDIS, namespace="polymathera"
        )
    )

    # Distributed locking settings
    lock_timeout: int = 30  # seconds
    lock_retry_count: int = 3
    lock_retry_delay: float = 0.1  # seconds

    # State synchronization settings
    sync_interval: int = 5  # seconds
    sync_batch_size: int = 100
    sync_timeout: int = 30  # seconds

    # Cache settings
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    max_cache_size: int = 10000  # entries

    # Consistency settings
    consistency_level: str = "eventual"  # eventual, strong
    replication_factor: int = 2
    quorum_size: int = 2

    # Performance tuning
    max_concurrent_operations: int = 100
    operation_timeout: int = 30  # seconds
    batch_size: int = 50

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60  # seconds

    # Monitoring and observability
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds

    CONFIG_PATH: ClassVar[str] = "distributed.distributed_state"



@register_polymathera_config()
class SystemConfig(ConfigComponent):
    """Configuration for the Polymathera system"""

    name: str = Field(default="polymathera")
    version: str = Field(default="1.0.0")
    architecture: str = Field(default="x86_64")
    storage: StorageConfig | None = Field(default=None)
    chat: UserChatServiceConfig | None = Field(default=None)
    redis: RedisConfig | None = Field(default=None)
    security: SecurityManagerConfig | None = Field(default=None)
    auth_service: AuthServiceConfig | None = Field(default=None)
    observability: ObservabilityConfig | None = Field(default=None)
    service_registry: ServiceRegistryConfig | None = Field(default=None)
    messaging: SystemMessagingConfig | None = Field(default=None)
    distributed_state: DistributedStateConfig | None = Field(default=None)

    CONFIG_PATH: ClassVar[str] = "system"

