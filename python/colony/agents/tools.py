"""Tool system for agent tool discovery, execution, and management.

Provides:
- Tool registry (global view of all available tools)
- Tool discovery (by category, name, capability)
- Tool execution with per-call authentication
- Result caching with TTL
- MCP server integration

Tools can be:
1. Native tools (deployed as Ray Serve deployments)
2. MCP tools (from MCP servers)
3. Custom tools (registered dynamically)
"""

import asyncio
import logging
import time
from typing import Any

from pydantic import Field

from ..distributed import get_polymathera
from ..distributed.state_management import SharedState, StateManager
from ..distributed.ray_utils import serving
from .models import ActionStatus, ToolCall, ToolMetadata

logger = logging.getLogger(__name__)


class ToolSystemState(SharedState):
    """Global tool registry - READ-MOSTLY.

    This state stores:
    - All registered tools (tool_id -> ToolMetadata)
    - Category indices for discovery
    - Tool usage statistics

    This state is NOT for:
    - Tool execution results (use caching layer)
    - Per-call authentication (passed with each call)
    """

    # Tool registry
    tools: dict[str, ToolMetadata] = Field(default_factory=dict)

    # Category indices for discovery
    tools_by_category: dict[str, list[str]] = Field(default_factory=dict)  # category -> tool_ids

    # Statistics
    tool_usage_counts: dict[str, int] = Field(default_factory=dict)  # tool_id -> call count
    tool_error_counts: dict[str, int] = Field(default_factory=dict)  # tool_id -> error count

    @classmethod
    def get_state_key(cls, app_name: str) -> str:
        """Generate state key for this tool system."""
        return f"polymathera:serving:{app_name}:tools:system"


class ToolResultCache:
    """LRU cache for tool results with TTL.

    Caches tool execution results to avoid redundant calls.
    Each cached result has a TTL and is keyed by (tool_id, params_hash).
    """

    def __init__(self, max_size: int = 10000, default_ttl_s: float = 300.0):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached results
            default_ttl_s: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl_s = default_ttl_s

        # Cache: (tool_id, params_hash) -> (result, expiry_time, access_time)
        self.cache: dict[tuple[str, str], tuple[Any, float, float]] = {}

        # Access tracking for LRU
        self.access_times: dict[tuple[str, str], float] = {}

    def _make_key(self, tool_id: str, parameters: dict[str, Any]) -> tuple[str, str]:
        """Generate cache key from tool_id and parameters.

        Args:
            tool_id: Tool identifier
            parameters: Tool parameters

        Returns:
            Cache key tuple (tool_id, params_hash)
        """
        import hashlib
        import json

        # Sort parameters for consistent hashing
        params_json = json.dumps(parameters, sort_keys=True)
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()[:16]
        return (tool_id, params_hash)

    def get(self, tool_id: str, parameters: dict[str, Any]) -> Any | None:
        """Get cached result if available and not expired.

        Args:
            tool_id: Tool identifier
            parameters: Tool parameters

        Returns:
            Cached result or None if not found/expired
        """
        key = self._make_key(tool_id, parameters)

        if key not in self.cache:
            return None

        result, expiry_time, _ = self.cache[key]

        # Check if expired
        if time.time() > expiry_time:
            del self.cache[key]
            self.access_times.pop(key, None)
            return None

        # Update access time
        self.access_times[key] = time.time()
        self.cache[key] = (result, expiry_time, time.time())

        return result

    def put(
        self, tool_id: str, parameters: dict[str, Any], result: Any, ttl_s: float | None = None
    ) -> None:
        """Store result in cache.

        Args:
            tool_id: Tool identifier
            parameters: Tool parameters
            result: Tool result
            ttl_s: TTL in seconds (uses default if None)
        """
        key = self._make_key(tool_id, parameters)

        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        expiry_time = time.time() + ttl

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Find least recently used
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        # Store result
        self.cache[key] = (result, expiry_time, time.time())
        self.access_times[key] = time.time()

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.access_times.clear()


@serving.deployment
class ToolManagerDeployment:
    """Global tool manager - tracks and executes tools.

    Provides:
    - Tool registration and discovery
    - Tool execution with authentication
    - Result caching
    - MCP integration
    - Usage tracking

    Does NOT provide:
    - Tool implementation (tools are separate deployments)
    """

    def __init__(self):
        """Initialize tool manager."""
        # Initialized in initialize()
        self.app_name: str | None = None
        self.state_manager: StateManager[ToolSystemState] | None = None
        self.result_cache = ToolResultCache()

        # MCP integration (optional)
        self.mcp_servers: dict[str, Any] = {}  # server_name -> MCP client

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize state manager and MCP clients."""
        # Get app name from environment
        self.app_name = serving.get_my_app_name()
        logger.info(f"Initializing ToolManagerDeployment for app {self.app_name}")

        # Get Polymathera for state management
        polymathera = get_polymathera()

        # Initialize StateManager
        self.state_manager = await polymathera.get_state_manager(
            state_type=ToolSystemState,
            state_key=ToolSystemState.get_state_key(self.app_name),
        )

        logger.info("ToolManagerDeployment initialized")

    # === Tool Registration ===

    @serving.endpoint
    async def register_tool(self, tool: ToolMetadata) -> None:
        """Register a new tool.

        Args:
            tool: Tool metadata
        """
        async for state in self.state_manager.write_transaction():
            state.tools[tool.tool_id] = tool

            # Register in category index
            if tool.category not in state.tools_by_category:
                state.tools_by_category[tool.category] = []
            if tool.tool_id not in state.tools_by_category[tool.category]:
                state.tools_by_category[tool.category].append(tool.tool_id)

            # Initialize usage counts
            if tool.tool_id not in state.tool_usage_counts:
                state.tool_usage_counts[tool.tool_id] = 0
            if tool.tool_id not in state.tool_error_counts:
                state.tool_error_counts[tool.tool_id] = 0

            logger.info(f"Registered tool {tool.tool_id} (category={tool.category})")

    @serving.endpoint
    async def unregister_tool(self, tool_id: str) -> None:
        """Unregister a tool.

        Args:
            tool_id: Tool identifier
        """
        async for state in self.state_manager.write_transaction():
            if tool_id in state.tools:
                tool = state.tools[tool_id]

                # Remove from category index
                if tool.category in state.tools_by_category:
                    if tool_id in state.tools_by_category[tool.category]:
                        state.tools_by_category[tool.category].remove(tool_id)
                    if not state.tools_by_category[tool.category]:
                        del state.tools_by_category[tool.category]

                # Remove from main registry
                del state.tools[tool_id]

                logger.info(f"Unregistered tool {tool_id}")

    # === Tool Discovery ===

    @serving.endpoint
    async def list_all_tools(self) -> list[str]:
        """List all tool IDs.

        Returns:
            List of tool IDs
        """
        async for state in self.state_manager.read_transaction():
            return list(state.tools.keys())

    @serving.endpoint
    async def get_tool_metadata(self, tool_id: str) -> ToolMetadata | None:
        """Get tool metadata.

        Args:
            tool_id: Tool identifier

        Returns:
            ToolMetadata if exists, None otherwise
        """
        async for state in self.state_manager.read_transaction():
            return state.tools.get(tool_id)

    @serving.endpoint
    async def find_tools_by_category(self, category: str) -> list[str]:
        """Find all tools in a category.

        Args:
            category: Tool category

        Returns:
            List of tool IDs
        """
        async for state in self.state_manager.read_transaction():
            return state.tools_by_category.get(category, []).copy()

    @serving.endpoint
    async def search_tools(self, query: str) -> list[ToolMetadata]:
        """Search tools by name or description.

        Args:
            query: Search query

        Returns:
            List of matching tool metadata
        """
        query_lower = query.lower()
        results = []

        async for state in self.state_manager.read_transaction():
            for tool in state.tools.values():
                if (
                    query_lower in tool.name.lower()
                    or query_lower in tool.description.lower()
                    or any(query_lower in tip.lower() for tip in tool.usage_tips)
                ):
                    results.append(tool)

        return results

    # === Tool Execution ===

    @serving.endpoint
    async def execute_tool(self, tool_call: ToolCall) -> ToolCall:
        """Execute a tool call.

        Args:
            tool_call: Tool call with parameters and auth

        Returns:
            Updated ToolCall with result or error
        """
        # Mark as running
        tool_call.status = ActionStatus.RUNNING

        try:
            # Get tool metadata
            tool_metadata = await self.get_tool_metadata(tool_call.tool_id)
            if not tool_metadata:
                tool_call.status = ActionStatus.FAILED
                tool_call.error = f"Tool {tool_call.tool_id} not found"
                return tool_call

            # Check cache first (if no auth required or same auth)
            cached_result = self.result_cache.get(tool_call.tool_id, tool_call.parameters)
            if cached_result is not None:
                tool_call.status = ActionStatus.COMPLETED
                tool_call.result = cached_result
                tool_call.completed_at = time.time()
                logger.debug(f"Tool {tool_call.tool_id} result retrieved from cache")
                return tool_call

            # Get tool deployment handle
            try:
                tool_handle = serving.get_deployment(
                    app_name=tool_metadata.deployment_app_name,
                    deployment_name=tool_metadata.deployment_name,
                )
            except Exception as e:
                tool_call.status = ActionStatus.FAILED
                tool_call.error = f"Failed to get tool deployment: {e}"
                await self._record_tool_error(tool_call.tool_id)
                return tool_call

            # Execute tool (assuming tool has an 'execute' endpoint)
            try:
                # Pass authentication if required
                if tool_metadata.requires_auth:
                    if not tool_call.auth_token:
                        tool_call.status = ActionStatus.FAILED
                        tool_call.error = "Tool requires authentication but no token provided"
                        await self._record_tool_error(tool_call.tool_id)
                        return tool_call

                    result = await tool_handle.execute(
                        parameters=tool_call.parameters, auth_token=tool_call.auth_token
                    )
                else:
                    result = await tool_handle.execute(parameters=tool_call.parameters)

                # Store result
                tool_call.result = result
                tool_call.status = ActionStatus.COMPLETED
                tool_call.completed_at = time.time()

                # Cache result
                self.result_cache.put(tool_call.tool_id, tool_call.parameters, result)

                # Record usage
                await self._record_tool_usage(tool_call.tool_id)

                logger.debug(f"Tool {tool_call.tool_id} executed successfully")

            except Exception as e:
                tool_call.status = ActionStatus.FAILED
                tool_call.error = f"Tool execution failed: {e}"
                tool_call.completed_at = time.time()
                await self._record_tool_error(tool_call.tool_id)
                logger.error(f"Tool {tool_call.tool_id} execution failed: {e}")

        except Exception as e:
            tool_call.status = ActionStatus.FAILED
            tool_call.error = f"Unexpected error: {e}"
            tool_call.completed_at = time.time()
            logger.error(f"Unexpected error executing tool {tool_call.tool_id}: {e}")

        return tool_call

    async def _record_tool_usage(self, tool_id: str) -> None:
        """Record tool usage (internal).

        Args:
            tool_id: Tool identifier
        """
        async for state in self.state_manager.write_transaction():
            if tool_id in state.tool_usage_counts:
                state.tool_usage_counts[tool_id] += 1

    async def _record_tool_error(self, tool_id: str) -> None:
        """Record tool error (internal).

        Args:
            tool_id: Tool identifier
        """
        async for state in self.state_manager.write_transaction():
            if tool_id in state.tool_error_counts:
                state.tool_error_counts[tool_id] += 1

    # === MCP Integration ===

    @serving.endpoint
    async def register_mcp_server(self, server_name: str, server_url: str) -> None:
        """Register an MCP server for tool discovery.

        Args:
            server_name: MCP server name
            server_url: MCP server URL
        """
        # TODO: Implement MCP client integration
        # This would connect to MCP server and auto-register its tools
        logger.info(f"MCP server registration: {server_name} at {server_url}")
        logger.warning("MCP integration not yet implemented")

    # === Monitoring ===

    @serving.endpoint
    async def get_tool_stats(self) -> dict[str, Any]:
        """Get tool system statistics.

        Returns:
            Dictionary with stats
        """
        async for state in self.state_manager.read_transaction():
            return {
                "total_tools": len(state.tools),
                "tools_by_category": {
                    category: len(tool_ids)
                    for category, tool_ids in state.tools_by_category.items()
                },
                "tool_usage_counts": dict(state.tool_usage_counts),
                "tool_error_counts": dict(state.tool_error_counts),
                "cache_size": len(self.result_cache.cache),
            }
