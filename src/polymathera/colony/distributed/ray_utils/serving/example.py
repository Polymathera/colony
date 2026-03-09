"""Example usage of the Polymathera serving framework.

This file demonstrates how to use the serving framework to create
deployments and build applications.
"""

import asyncio
import logging
import json
import os


import ray

from polymathera.colony.distributed.ray_utils import serving

# Configure logging for driver
log_level = os.environ.get("POLYMATHERA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Example 1: Simple deployment with single endpoint, max_concurrency, and logging config
@serving.deployment(
    name="GreeterService",
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_queue_length": 3,
    },
    max_concurrency=2,  # Test max concurrency - only 2 concurrent requests per replica
    logging_config=serving.LoggingConfig(level="DEBUG"),  # DEBUG logs for this deployment
)
class GreeterService:
    """Simple greeting service."""

    def __init__(self, greeting_prefix: str = "Hello"):
        self.greeting_prefix = greeting_prefix
        self.request_count = 0

    @serving.endpoint
    async def greet(self, name: str) -> str:
        """Greet someone by name."""
        self.request_count += 1
        await asyncio.sleep(2.0)  # Longer delay to test request queueing and concurrency limits at the deployment proxy
        return f"{self.greeting_prefix}, {name}! (Request #{self.request_count})"

    @serving.endpoint
    async def get_stats(self) -> dict:
        """Get service statistics."""
        return {"request_count": self.request_count}


# Example 2: Deployment with periodic health check
@serving.deployment(name="ConnectionService")
class ConnectionService:
    """Service that maintains a connection and performs periodic health checks."""

    def __init__(self):
        self.connection_healthy = True
        self.health_check_count = 0
        self.reconnect_count = 0

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize the connection."""
        logger.info("Initializing connection...")
        # Simulate connection initialization
        self.connection_healthy = True

    @serving.periodic_health_check(interval_s=10.0)
    async def check_connection_health(self):
        """Periodically check if connection is healthy and reconnect if needed."""
        self.health_check_count += 1
        logger.info(f"Running periodic health check (count: {self.health_check_count})")

        # Simulate occasional connection issues
        import random
        if random.random() < 0.1:  # 10% chance of connection issue
            logger.warning("Connection unhealthy detected, reconnecting...")
            self.connection_healthy = False
            # Simulate reconnection
            await asyncio.sleep(0.1)
            self.connection_healthy = True
            self.reconnect_count += 1
            logger.info(f"Reconnected successfully (reconnect count: {self.reconnect_count})")

    @serving.endpoint
    async def get_connection_status(self) -> dict:
        """Get connection status."""
        return {
            "healthy": self.connection_healthy,
            "health_check_count": self.health_check_count,
            "reconnect_count": self.reconnect_count,
        }

    @serving.cleanup_deployment
    async def cleanup(self):
        """Clean up the connection."""
        logger.info("Cleaning up connection...")


# Example 3: Deployment that calls another deployment
@serving.deployment(name="UppercaseService")
class UppercaseService:
    """Service that converts text to uppercase."""

    @serving.endpoint
    async def uppercase(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()


@serving.deployment(name="CompositeService")
class CompositeService:
    """Service that composes multiple other services."""

    @serving.endpoint
    async def greet_and_shout(self, name: str) -> str:
        """Greet someone and return the result in uppercase."""
        # Discover other deployments
        greeter: serving.DeploymentHandle = serving.get_deployment(
            serving.get_my_app_name(),
            deployment_class=GreeterService,
        )
        uppercaser: serving.DeploymentHandle = serving.get_deployment(
            serving.get_my_app_name(),
            deployment_class=UppercaseService,
        )

        # Call them
        greeting = await greeter.greet(name)
        result = await uppercaser.uppercase(greeting)
        return result


# Example 4: Custom request router
class PrefixRouter(serving.RequestRouter):
    """Route requests based on a prefix in the metadata."""

    async def route_request(
        self,
        request: serving.DeploymentRequest,
        replicas: list[serving.DeploymentReplicaInfo],
    ) -> serving.DeploymentReplicaInfo:
        """Route based on request metadata."""
        if not replicas:
            raise ValueError("No healthy replicas available")

        # Simple hash-based routing for demonstration
        # In practice, you might route based on request content
        prefix = request.metadata.get("prefix", "")
        index = hash(prefix) % len(replicas)
        return replicas[index]


@serving.deployment(
    name="CustomRoutedService",
    router_class=PrefixRouter,
)
class CustomRoutedService:
    """Service with custom routing."""

    @serving.endpoint
    async def process(self, data: str) -> str:
        """Process data."""
        return f"Processed: {data}"


async def main():
    """Run example application."""
    # Initialize Ray with debug logging
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.DEBUG,
            log_to_driver=True,
        )

    # Create application
    app = serving.Application(name="ExampleApp")

    # Add deployments
    app.add_deployment(GreeterService.bind(greeting_prefix="Hi"))
    app.add_deployment(ConnectionService.bind())
    app.add_deployment(UppercaseService.bind())
    app.add_deployment(CompositeService.bind())

    # Start application
    await app.start()

    # Test the deployments with concurrent requests to see queueing in action
    logger.info("Testing GreeterService with concurrent requests...")
    greeter_handle = serving.get_deployment("ExampleApp", "GreeterService")

    # Send 5 concurrent requests - with max_concurrency=2 and 1 replica,
    # 2 will execute immediately, 3 will queue
    logger.info("Sending 5 concurrent requests (max_concurrency=2, so 3 will queue)...")
    start_time = asyncio.get_event_loop().time()

    results = await asyncio.gather(
        greeter_handle.greet("Alice"),
        greeter_handle.greet("Bob"),
        greeter_handle.greet("Charlie"),
        greeter_handle.greet("David"),
        greeter_handle.greet("Eve"),
    )

    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"All 5 requests completed in {elapsed:.2f}s")
    for i, result in enumerate(results):
        logger.info(f"  Result {i+1}: {result}")

    stats = await greeter_handle.get_stats()
    logger.info(f"Stats: {stats}")

    logger.info("\nTesting ConnectionService with periodic health checks...")
    connection_handle = serving.get_deployment("ExampleApp", "ConnectionService")

    # Wait for a few health checks to run
    logger.info("Waiting 25 seconds for periodic health checks to execute...")
    await asyncio.sleep(25)

    # Get connection status
    connection_status = await connection_handle.get_connection_status()
    logger.info(f"Connection status after health checks: {connection_status}")

    logger.info("\nTesting CompositeService...")
    composite_handle = serving.get_deployment("ExampleApp", "CompositeService")
    result = await composite_handle.greet_and_shout("Zara")
    logger.info(f"Result: {result}")

    # Get application stats
    logger.info("\nApplication stats:")
    all_stats = await app.get_all_stats()
    for deployment_name, deployment_stats in all_stats.items():
        logger.info(f"  {deployment_name}: {json.dumps(deployment_stats, indent=2)}")

    # Cleanup
    logger.info("\nStopping application...")
    await app.stop()

    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())
