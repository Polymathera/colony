import abc
import asyncio
import contextlib
import difflib
import functools
import inspect
import io
import json
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any
import concurrent.futures

import docker
from termcolor import colored

from rich.logging import RichHandler
from rich.console import Console


def setup_logger(name: str, level: int = logging.INFO, use_rich: bool = False, capture_logs: bool = False) -> logging.Logger | tuple[logging.Logger, io.StringIO]:
    """Set up a logger with consistent formatting and handlers.

    This function provides a universal logging setup that should be used
    throughout the codebase instead of logging.basicConfig().

    Args:
        name: The logger name, typically __name__
        level: The logging level (default: INFO)
        use_rich: Whether to use RichHandler for better formatting (default: True)
        capture_logs: Whether to capture logs to a StringIO stream (default: False)

    Returns:
        Configured logging.Logger instance, or tuple of (logger, log_stream) if capture_logs=True
    """
    logger = logging.getLogger(name)
    log_stream: io.StringIO | None = None

    # Check if root logger already has handlers to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Root logger is already configured
        # so just return the named logger - it will inherit the configuration
        if not capture_logs:
            return logger
        capture_handler: logging.StreamHandler | None = None
        if logger.handlers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    capture_handler = handler.stream
                    break
        if capture_handler is None:
            log_stream = io.StringIO()
            capture_handler = logging.StreamHandler(log_stream)
            capture_handler.setLevel(level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            capture_handler.setFormatter(formatter)
            logger.addHandler(capture_handler)
            root_logger.addHandler(capture_handler)
            # Also add to root logger to capture all logs
            root_logger.addHandler(capture_handler)
        else:
            log_stream = capture_handler.stream
        return logger, log_stream

    # Configure the root logger if it hasn't been configured yet
    if use_rich:
        handler = RichHandler(rich_tracebacks=True)
        formatter = logging.Formatter("%(message)s")
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(message)s",
    #     datefmt="[%X]",
    #     handlers=[RichHandler(rich_tracebacks=True)],
    # )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Add capture handler if requested
    if capture_logs:
        log_stream = io.StringIO()
        capture_handler = logging.StreamHandler(log_stream)
        capture_handler.setLevel(level)
        capture_handler.setFormatter(formatter)
        logger.addHandler(capture_handler)
        root_logger.addHandler(capture_handler)
        return logger, log_stream

    return logger


def get_ray_logger() -> logging.Logger:
    """Get the Ray logger
    Use Ray's recommended logging pattern - get the root logger
    This ensures logs are properly forwarded to the driver when log_to_driver=True
    Reference: https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
    """
    return logging.getLogger()


# Asyncio utility functions


def async_partial(
    func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """
    Create a partial function that works with both synchronous and asynchronous functions.

    Args:
        func: The function to be partially applied. Can be sync or async.
        *args: Positional arguments to be partially applied.
        **kwargs: Keyword arguments to be partially applied.

    Returns:
        A new partial function that can be called with additional arguments.

    Raises:
        TypeError: If func is not callable.
    """
    if not callable(func):
        raise TypeError("func must be callable")

    @functools.wraps(func)
    async def wrapper(*inner_args: Any, **inner_kwargs: Any) -> Any:
        all_kwargs = {**kwargs, **inner_kwargs}
        try:
            result = func(*args, *inner_args, **all_kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            logging.error(f"Error in async_partial: {e!s}")
            raise

    return wrapper


async def with_timeout(coro, timeout=10):
    try:
        return await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError:
        return None


async def call_async(func, *args, **kwargs):
    """Run a blocking (e.g., boto3) call in the default executor to avoid
    blocking the asyncio event loop."""
    return await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))

async def call_async_in_executor(executor: concurrent.futures.Executor, func, *args, **kwargs):
    """Run a blocking (e.g., boto3) call in the given executor to avoid
    blocking the asyncio event loop."""
    return await asyncio.get_event_loop().run_in_executor(executor, lambda: func(*args, **kwargs))




def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine synchronously.

    WARNING: This runs the coroutine in a separate thread with its own event loop.
    Ensure the coroutine doesn't access thread-unsafe shared state.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    import concurrent.futures

    def run_in_thread():
        return asyncio.run(coro)

    # This blocks until the coroutine completes, but runs in separate thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()

def run_sync1(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're inside an async context
        return loop.run_until_complete(coro)  # loop.create_task(coro)
    except RuntimeError:
        # No event loop running - we're in a sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def stop_event_loop():
    loop = asyncio.get_event_loop()
    loop.stop()


def schedule_callback(callback, *args, **kwargs: Any):
    loop = asyncio.get_event_loop()
    # loop.call_soon(functools.partial(callback, *args, **kwargs))
    loop.call_soon_threadsafe(functools.partial(callback, *args, **kwargs))


def create_dynamic_asyncio_task(owner, coro: Awaitable):
    """
    Must Save a reference to the result of calls to asyncio.create_task,
    to avoid a task disappearing mid-execution.
    """
    assert inspect.iscoroutine(coro), "coro must be a coroutine"
    assert asyncio.iscoroutine(coro), "coro must be a coroutine"
    assert (
        asyncio.get_event_loop() is asyncio.get_running_loop()
    ), "must be called within an event loop"
    if not hasattr(owner, "_dynamic_tasks"):
        owner._dynamic_tasks = []

    # Wrap the coro coroutine into a Task and schedule its execution.
    task = asyncio.create_task(coro)
    # Create a strong reference.
    owner._dynamic_tasks.append(task)
    # To prevent keeping references to finished tasks forever,
    # make each task remove its own reference from the set after completion:
    task.add_done_callback(owner._dynamic_tasks.remove)
    return task  # So we can await the result of this task.


def cancel_dynamic_asyncio_tasks(owner):
    if not hasattr(owner, "_dynamic_tasks"):
        return
    for task in owner._dynamic_tasks:
        if not task.done():
            task.cancel()


async def cleanup_dynamic_asyncio_tasks(owner, raise_exceptions=True):
    if not hasattr(owner, "_dynamic_tasks"):
        return
    cancel_dynamic_asyncio_tasks(owner)

    results = await asyncio.gather(
        *owner._dynamic_tasks, return_exceptions=raise_exceptions
    )
    # Check for exceptions in the results
    if not raise_exceptions:
        return
    for result in results:
        if isinstance(result, Exception) and not isinstance(
            result, asyncio.CancelledError
        ):
            raise result


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass to ensure only one instance of a class is created.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs: Any):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Define a type variable for the file path
FilePath = str | Path


def print_files_compare(f1: dict[FilePath, str], f2: dict[FilePath, str]):
    def colored_diff(s1, s2):
        lines1 = s1.splitlines()
        lines2 = s2.splitlines()

        diff = difflib.unified_diff(lines1, lines2, lineterm="")

        RED = "\033[38;5;202m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        colored_lines = []
        for line in diff:
            if line.startswith("+"):
                colored_lines.append(GREEN + line + RESET)
            elif line.startswith("-"):
                colored_lines.append(RED + line + RESET)
            else:
                colored_lines.append(line)

        return "\n".join(colored_lines)

    for file in sorted(set(f1.keys()) | set(f2.keys())):
        diff_lines = colored_diff(f1.get(file, ""), f2.get(file, ""))
        if diff_lines:
            print(f"Changes to {file}:")
            print(diff_lines)


def prompt_yesno(prompt) -> bool:
    output = f'{prompt} ({colored("y", "green")}/{colored("n", "red")}) '
    while True:
        response = input(output).strip().lower()
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
            return False
        print("Please answer 'y[es]' or 'n[o]'")


def get_system_info():
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "packages": format_installed_packages(get_installed_packages()),
    }
    return system_info


def format_installed_packages(packages):
    return "\n".join([f"{name}: {version}" for name, version in packages.items()])


def get_installed_packages():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
        )
        packages = json.loads(result.stdout)
        return {pkg["name"]: pkg["version"] for pkg in packages}
    except Exception as e:
        return str(e)


@contextlib.contextmanager
def suppress_logging(logger_name, level=logging.ERROR):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()

    try:
        logger.setLevel(level)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ), contextlib.suppress(UserWarning):
            yield
    finally:
        logger.setLevel(original_level)


def concatenate_paths(base_path, sub_path):
    # Compute the relative path from base_path to sub_path
    relative_path = os.path.relpath(sub_path, base_path)

    # If the relative path is not in the parent directory, use the original sub_path
    if not relative_path.startswith(".."):
        return sub_path

    # Otherwise, concatenate base_path and sub_path
    return os.path.normpath(os.path.join(base_path, sub_path))


def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")


def is_docker_available() -> bool:
    """Check if Docker is available and supports Linux containers

    Returns:
        bool: True if Docker is available and supports Linux containers, False otherwise
    """
    try:
        client = docker.from_env()
        docker_info = client.info()
        return docker_info["OSType"] == "linux"
    except Exception:
        return False
