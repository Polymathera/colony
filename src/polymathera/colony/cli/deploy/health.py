"""Health check utilities for colony-env deployment."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable


async def tcp_check(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def redis_ping(host: str, port: int, timeout: float = 2.0) -> bool:
    """Send a Redis PING and check for PONG response."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.write(b"PING\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return response.strip() == b"+PONG"
    except (OSError, asyncio.TimeoutError):
        return False


async def docker_container_healthy(container_name: str) -> bool:
    """Check if a Docker container exists and is healthy."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "inspect", "--format", "{{.State.Health.Status}}", container_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode == 0 and stdout.decode().strip() == "healthy"


async def docker_container_running(container_name: str) -> bool:
    """Check if a Docker container exists and is running."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "inspect", "--format", "{{.State.Running}}", container_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode == 0 and stdout.decode().strip() == "true"


async def wait_until_ready(
    check_fn: Callable[[], Awaitable[bool]],
    timeout: float = 60.0,
    interval: float = 2.0,
    description: str = "",
) -> bool:
    """Poll a check function until it returns True or timeout is reached."""
    elapsed = 0.0
    while elapsed < timeout:
        if await check_fn():
            return True
        await asyncio.sleep(interval)
        elapsed += interval
    return False
