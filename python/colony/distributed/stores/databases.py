from datetime import datetime, timedelta, timezone
import asyncio
import logging
from typing import Any
from sqlmodel import SQLModel, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker, selectinload
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError

from ..configs import RelationalStorageConfig
from polymathera.schema.vmr import (
    VirtualMonorepo,
    Repository,
    RepositoryDependency,
    VMRRepositoryLink,
)
from ...utils.retry import standard_retry

import boto3
from botocore.exceptions import ClientError
import os

logger = logging.getLogger(__name__)

class RelationalStorage:
    """Relational storage implementation using PostgreSQL via SQLAlchemy async."""

    def __init__(self, config: RelationalStorageConfig | None = None, create_database_schema: bool = True):
        self.config: RelationalStorageConfig | None = config
        self.create_database_schema: bool = create_database_schema
        # Track engines per event loop to avoid "attached to different loop" errors
        self._loop_engines: dict[int, AsyncEngine] = {}
        self._loop_sessions: dict[int, sessionmaker] = {}
        # Track initialization state
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Initialize the relational storage."""
        logger.info("________ RelationalStorage.initialize() called.")
        self.config = await RelationalStorageConfig.check_or_get_component(self.config)
        if self._initialized:
            logger.info("________ RelationalStorage already initialized.")
            return  # Already initialized

        # Mark as initialized before creating engines to avoid circular dependency
        self._initialized = True

        # Create tables if they don't exist yet (using current loop engine)
        if self.create_database_schema:
            engine = self._get_current_loop_engine()
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            logger.info("________ Database schema creation completed")
        else:
            logger.info("Skipping database schema creation as it was already created during CDK deployment")
        logger.info("________ Initialized relational storage with event loop-aware engines")

        # Migrations are handled during deployment; do not attempt runtime schema patching here.

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        from ...utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
        except Exception as e:
            logger.warning(f"Error cleaning up RelationalStorage tasks: {e}")

        # Dispose of all per-loop engines
        for loop_id, engine in list(self._loop_engines.items()):
            try:
                await engine.dispose()
                logger.debug(f"Disposed AsyncEngine for event loop {loop_id}")
            except Exception as e:
                logger.warning(f"Error disposing engine for loop {loop_id}: {e}")

        # Clear the loop-specific collections and reset initialization
        self._loop_engines.clear()
        self._loop_sessions.clear()
        self._initialized = False
        logger.info("RelationalStorage cleanup completed")

    @standard_retry(logger)
    async def add_repo_to_vmr(self, repo: Repository, vmr: VirtualMonorepo) -> bool:
        raise NotImplementedError("add_repo_to_vmr: Not implemented")

    @standard_retry(logger)
    async def remove_repo_from_vmr(self, repo: Repository, vmr: VirtualMonorepo) -> bool:
        raise NotImplementedError("remove_repo_from_vmr: Not implemented")

    @standard_retry(logger)
    async def update_repo(self, repo: Repository) -> bool:
        logger.info(f"________ RelationalStorage.update_repo() called for repo: {repo.id}")
        await self.initialize()

        async with self._get_current_loop_session_maker()() as session:
            async with session.begin():
                try:
                    # Check if this exact repo version already exists
                    existing_repo_stmt = select(Repository).where(
                        Repository.origin_url == repo.origin_url,
                        Repository.branch == repo.branch,
                        Repository.commit == repo.commit,
                        Repository.is_direct_external_dependency == False,
                    )
                    existing_repo = (
                        await session.execute(existing_repo_stmt)
                    ).scalar_one_or_none()
                    logger.info(f"________ Checking for existing repo: {repo.origin_url}. Found: {existing_repo is not None}")

                    if not existing_repo:
                        raise ValueError(f"Repository {repo.id} not found. Cannot update repo.")

                    # Use merge to handle potential session conflicts
                    # This ensures the repo object is properly associated with the current session
                    merged_repo = await session.merge(repo)

                    await session.flush()
                    logger.info(f"________ Successfully updated repo {repo.id}")
                    # await session.refresh(repo)  # Refresh the instance to ensure we have the latest data
                    return True
                except IntegrityError as e:
                    await session.rollback()
                    logger.error(f"Failed to update repo due to integrity error: {e!s}")
                    raise
                except Exception as e:
                    await session.rollback()
                    logger.error(f"________ Failed to update repo: {e!s}", exc_info=True)
                    raise

            return False

    @standard_retry(logger)
    async def store_vmr(self, **vmr_data: Any) -> str:
        logger.info(f"________ RelationalStorage.store_vmr() called for VMR: {vmr_data.get('name')}")
        await self.initialize()

        # Create a fresh copy of vmr_data for each retry attempt
        vmr_data_copy = vmr_data.copy()

        async with self._get_current_loop_session_maker()() as session:
            async with session.begin():
                # ------------------------------------------------------------------
                # explode repo_spec  →  Repository rows + relationship links
                # ------------------------------------------------------------------
                try:
                    # Extract repository and candidate repository specs
                    repo_specs = vmr_data_copy.pop("repo_spec", [])
                    cand_repo_mapping = vmr_data_copy.pop("candidate_repositories", {})
                    logger.info(f"________ Storing VMR with {len(repo_specs)} repositories.")

                    # Create the VirtualMonorepo instance
                    vmr = VirtualMonorepo(**vmr_data_copy)

                    # Handle primary repositories with proper deduplication
                    for spec in repo_specs:
                        # Check if this exact repo version already exists
                        existing_repo_stmt = select(Repository).where(
                            Repository.origin_url == spec["origin_url"],
                            Repository.branch == spec.get("branch"),
                            Repository.commit == spec.get("commit"),
                            Repository.is_direct_external_dependency == False,
                        )
                        existing_repo = (
                            await session.execute(existing_repo_stmt)
                        ).scalar_one_or_none()
                        logger.info(f"________ Checking for existing repo: {spec['origin_url']}. Found: {existing_repo is not None}")

                        if existing_repo:
                            vmr.add_repository(existing_repo)  # Use deduplication method
                        else:
                            new_repo = Repository(**spec, is_direct_external_dependency=False)
                            # Only add to session if the repository was actually added to VMR
                            if vmr.add_repository(new_repo):  # Use deduplication method
                                session.add(new_repo)

                    # Handle candidate repositories and dependencies
                    for repo in vmr.repositories:
                        if repo.is_direct_external_dependency:
                            continue  # Skip dependencies of dependencies for now

                        # Find its dependencies in the original spec
                        # This assumes cand_repo_mapping uses a URL that can be matched
                        # This part of the logic is complex and may need adjustment based on how
                        # dependencies are specified for primary repos.
                        # For now, we assume we can find dependencies in cand_repo_mapping
                        for url_str, deps in cand_repo_mapping.items():
                            for dep_spec in deps:
                                # Find or create the dependency as a candidate repository
                                dep_repo_stmt = select(Repository).where(
                                    Repository.origin_url == dep_spec["origin_url"],
                                    Repository.is_direct_external_dependency == True,
                                )
                                dep_repo = (
                                    await session.execute(dep_repo_stmt)
                                ).scalar_one_or_none()

                                if not dep_repo:
                                    dep_repo = Repository(
                                        origin_url=dep_spec["origin_url"],
                                        is_direct_external_dependency=True,
                                    )
                                    session.add(dep_repo)

                                # Add to candidate repositories
                                if dep_repo not in vmr.candidate_repositories:
                                    vmr.candidate_repositories.append(dep_repo)

                                # Create the dependency link
                                dep_link = RepositoryDependency(
                                    dependent_repo=repo,
                                    dependency_repo=dep_repo,
                                    **dep_spec,
                                )
                                session.add(dep_link)

                    session.add(vmr)
                    await session.flush()
                    vmr_id = str(vmr.id)  # Return the newly assigned ID as string
                    logger.info(f"________ Successfully stored VMR {vmr_data.get('name')} with ID: {vmr_id}")
                    # await session.refresh(vmr)  # Refresh the instance to ensure we have the latest data
                except IntegrityError as e:
                    await session.rollback()
                    error_msg = str(e)
                    if "duplicate key value violates unique constraint" in error_msg:
                        logger.error(f"Database schema mismatch detected, such as an incorrect unique constraint. Error: {e!s}")
                        logger.error("Please recreate the database or run a migration to fix the schema.")
                        # Try to provide a more helpful error message
                        raise ValueError(
                            "Database schema error: There's an incorrect unique constraint on a database column. "
                            "Please recreate the database or run a migration to fix this."
                        ) from e
                    else:
                        logger.error(f"Failed to store VMR due to integrity error: {e!s}")
                        raise
                except Exception as e:
                    await session.rollback()
                    logger.error(f"________ Failed to store VMR: {e!s}", exc_info=True)
                    raise

            return vmr_id

    @standard_retry(logger)
    async def update_vmr(self, vmr_id: str, **vmr_updates: Any):
        await self.initialize()
        async with self._get_current_loop_session_maker()() as session:
            async with session.begin():
                try:
                    result = await session.execute(
                        select(VirtualMonorepo).where(VirtualMonorepo.id == vmr_id)
                    )
                    vmr: VirtualMonorepo | None = result.scalar_one_or_none()
                    if vmr:
                        # Handle repo_spec and candidate_repositories updates
                        if "repo_spec" in vmr_updates:
                            repo_specs = vmr_updates.pop("repo_spec")
                            # This is a simplification. A real implementation would need to
                            # reconcile the existing list with the new one.
                            vmr.repositories.clear()
                            for repo_spec in repo_specs:
                                repo_instance = Repository(**repo_spec)
                                # Only add to session if the repository was actually added to VMR
                                if vmr.add_repository(repo_instance):  # Use deduplication method
                                    session.add(repo_instance)

                        if "candidate_repositories" in vmr_updates:
                            # This logic also needs careful implementation to update dependencies.
                            # For now, we'll just log a warning.
                            logger.warning(
                                "Updating candidate_repositories directly is not fully supported yet in this simplified model."
                            )
                            vmr_updates.pop("candidate_repositories")

                        # Update remaining VMR attributes
                        for key, value in vmr_updates.items():
                            setattr(vmr, key, value)
                        vmr.updated_at = datetime.now(timezone.utc)
                        session.add(vmr)
                    else:
                        raise ValueError(f"VMR with ID {vmr_id} not found")
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Failed to update VMR: {e!s}")
                    raise

    @standard_retry(logger)
    async def load_vmr(self, vmr_id: str) -> VirtualMonorepo:
        await self.initialize()
        async with self._get_current_loop_session_maker()() as session:
            try:
                # Use eager loading to fetch repositories and their dependencies
                stmt = (
                    select(VirtualMonorepo)
                    .where(VirtualMonorepo.id == vmr_id)
                    .options(
                        selectinload(VirtualMonorepo.repositories).options(
                            selectinload(Repository.dependencies).options(
                                selectinload(RepositoryDependency.dependency_repo)
                            )
                        )
                    )
                )

                result = await session.execute(stmt)
                vmr: VirtualMonorepo | None = result.scalar_one_or_none()

                if not vmr:
                    raise ValueError(f"VMR with ID {vmr_id} not found")

                return vmr
            except Exception as e:
                logger.error(f"Failed to retrieve VMR from database: {e!s}")
                raise

    @standard_retry(logger)
    async def update_vmr_status(self, vmr_id: str, status: str):
        await self.initialize()
        async with self._get_current_loop_session_maker()() as session:
            try:
                result = await session.execute(select(VirtualMonorepo).where(VirtualMonorepo.id == vmr_id))
                vmr = result.scalar_one_or_none()
                if vmr:
                    vmr.status = status
                    vmr.updated_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update VMR status in database: {e!s}")
                raise

    @standard_retry(logger)
    async def get_vmr_status(self, vmr_id: str):
        await self.initialize()
        async with self._get_current_loop_session_maker()() as session:
            try:
                result = await session.execute(select(VirtualMonorepo).where(VirtualMonorepo.id == vmr_id))
                vmr = result.scalar_one_or_none()
                return vmr.status if vmr else None
            except Exception as e:
                logger.error(f"Failed to get VMR status from database: {e!s}")
                raise

    @standard_retry(logger)
    async def delete_vmr(self, vmr_id: str):
        await self.initialize()
        async with self._get_current_loop_session_maker()() as session:
            try:
                result = await session.execute(select(VirtualMonorepo).where(VirtualMonorepo.id == vmr_id))
                vmr = result.scalar_one_or_none()
                if vmr:
                    await session.delete(vmr)
                    await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete VMR from database: {e!s}")
                raise

    @standard_retry(logger)
    async def list_vmrs(self) -> list[VirtualMonorepo]:
        await self.initialize()
        async with self._get_current_loop_session_maker()() as session:
            try:
                # Load VMRs with their relationships
                stmt = select(VirtualMonorepo).options(
                    selectinload(VirtualMonorepo.repositories).options(
                        selectinload(Repository.dependencies).options(
                            selectinload(RepositoryDependency.dependency_repo)
                        )
                    )
                )
                result = await session.execute(stmt)
                vmrs = result.scalars().all()

                return vmrs
            except Exception as e:
                logger.error(f"Failed to list VMRs from database: {e!s}")
                raise

    @standard_retry(logger)
    async def cleanup_all_data(self) -> None:
        """Delete all VMRs, repositories, and dependencies from the database.

        This method is useful for testing to ensure a clean state.
        WARNING: This will delete ALL data in the database!
        """
        logger.info("________ RelationalStorage.cleanup_all_data() called - DELETING ALL DATA")
        await self.initialize()

        async with self._get_current_loop_session_maker()() as session:
            async with session.begin():
                try:
                    # Delete in the correct order to respect foreign key constraints
                    # 1. Delete repository dependencies first
                    await session.execute(delete(RepositoryDependency))
                    logger.info("________ Deleted all repository dependencies")

                    # 2. Delete VMR-Repository link table
                    await session.execute(delete(VMRRepositoryLink))
                    logger.info("________ Deleted all VMR-Repository links")

                    # 3. Delete VMRs
                    result = await session.execute(delete(VirtualMonorepo))
                    vmr_count = result.rowcount
                    logger.info(f"________ Deleted {vmr_count} VMRs")

                    # 4. Delete all repositories
                    result = await session.execute(delete(Repository))
                    repo_count = result.rowcount
                    logger.info(f"________ Deleted {repo_count} repositories")

                    logger.info("________ Successfully cleaned up all data from database")

                except Exception as e:
                    await session.rollback()
                    logger.error(f"________ Failed to cleanup all data: {e!s}", exc_info=True)
                    raise

    def _get_current_loop_engine(self) -> AsyncEngine:
        """Get or create an AsyncEngine for the current event loop."""
        loop = asyncio.get_running_loop()
        loop_id = id(loop)

        engine = self._loop_engines.get(loop_id)
        if engine is None:
            if not self._initialized:
                raise RuntimeError("RelationalStorage not initialized - call initialize() first")

            # Create a new engine for this loop using the same configuration
            pwd = self._resolve_db_password()
            url = self._build_sqlalchemy_url(pwd)
            engine = create_async_engine(
                url,
                future=True,
                ## pool_size=10,
                ## max_overflow=20,
                ## pool_pre_ping=True,
                ## pool_recycle=3600,
                # Disable pooling to prevent cross-event-loop connection sharing
                poolclass=None,  # Disables connection pooling entirely
                pool_pre_ping=False,  # Not needed without pooling
                echo=False
            )
            self._loop_engines[loop_id] = engine
            logger.debug(f"Created new AsyncEngine for event loop {loop_id}")

        return engine

    def _get_current_loop_session_maker(self) -> sessionmaker:
        """Get or create a sessionmaker for the current event loop."""
        loop = asyncio.get_running_loop()
        loop_id = id(loop)

        session_maker = self._loop_sessions.get(loop_id)
        if session_maker is None:
            engine = self._get_current_loop_engine()
            session_maker = sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            self._loop_sessions[loop_id] = session_maker
            logger.debug(f"Created new sessionmaker for event loop {loop_id}")

        return session_maker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sqlalchemy_url(self, password: str | None = None) -> str:
        """Return a full asyncpg PostgreSQL URL or fall back to in-memory SQLite."""

        missing = [
            key for key, present in {
                "db_user": self.config.db_user,
                "db_host": self.config.db_host,
                "db_port": self.config.db_port,
                "db_name": self.config.db_name,
                "password": password,
            }.items() if not present
        ]
        if missing:
            raise RuntimeError(
                f"RelationalStorage configuration incomplete; missing {', '.join(missing)}"
            )

        return (
            f"postgresql+asyncpg://{self.config.db_user}:{password}@"
            f"{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )

    def _resolve_db_password(self) -> str | None:
        """Fetch the DB password from AWS Secrets Manager if an ARN is configured."""

        arn = self.config.db_password_secret_arn
        logger.debug(f"________ Resolving DB password from SecretsManager: {arn}")
        if not arn or arn == "" or arn.startswith("placeholder"):
            logger.debug(f"________ No DB password secret ARN provided, using placeholder")
            return None

        sm = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION", "us-east-1"))
        logger.debug(f"________ Created SecretsManager client for region: {os.getenv('AWS_REGION', 'us-east-1')}")
        try:
            resp = sm.get_secret_value(SecretId=arn)
            logger.debug(f"________ Retrieved DB password from SecretsManager: {resp}")
            secret_str = resp.get("SecretString", "")
            if secret_str:
                logger.debug(f"________ Parsed DB password from SecretsManager: {secret_str}")
                import json

                parsed = json.loads(secret_str)
                logger.debug(f"________ Parsed DB password from SecretsManager: {parsed}")
                return parsed.get("password") or list(parsed.values())[0]
        except ClientError as exc:
            logger.warning(f"Could not retrieve DB password from SecretsManager: {exc}")
        return None
