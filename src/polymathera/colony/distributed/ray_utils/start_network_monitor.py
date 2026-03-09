"""
Script to start the NetworkMonitor Ray actor on a node.

This script is intended to be run in the background (e.g., via nohup)
on each Ray head and worker node as part of the node setup process.
It ensures that the NetworkMonitor actor is running and periodically
updates custom Ray resources with network latency information.
"""

import asyncio
import logging
import signal
import time
import sys
import os

import ray

# Assume NetworkMonitor is defined here. Adjust the import path as necessary.
# Ensure this path is correct relative to how the script is run or PYTHONPATH
try:
    from polymathera.colony.distributed.ray_utils.network_monitor import NetworkMonitor
    from polymathera.colony.utils import setup_logger
except ImportError:
    # Handle case where the script might be run from a different context
    # This might require adjusting PYTHONPATH when running the script
    logger = setup_logger(__name__)
    logger.error("Failed to import NetworkMonitor. Ensure PYTHONPATH is set correctly or adjust import path.", exc_info=True)
    sys.exit(1)


# --- Configuration ---
ACTOR_CHECK_INTERVAL_S = 60  # How often to check if the actor is alive
RETRY_DELAY_BASE_S = 5       # Base delay for retries
MAX_ACTOR_CREATE_RETRIES = 5 # Max attempts to create/get actor

# --- Logging Setup ---
# Send logs to a file specific to this script for easier debugging
# Use PID to differentiate logs if multiple instances accidentally run
pid = os.getpid()
log_file = f"/tmp/start_network_monitor_{pid}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout) # Also print to stdout/stderr
    ]
)
logger = logging.getLogger(__name__)

# --- Global State ---
monitor_actor = None
main_task = None
keep_running = True
shutdown_event = asyncio.Event() # Used to signal shutdown completion

async def main():
    """Initializes Ray and starts/manages the NetworkMonitor actor."""
    global monitor_actor, keep_running

    try:
        # --- Ray Connection ---
        ray_connected = False
        ray_init_retries = 12 # Retry for up to ~1 minute (5*1 + 5*2 + ...)
        for i in range(ray_init_retries):
            if not keep_running: return # Exit if shutdown signaled

            if ray.is_initialized():
                logger.info("Ray already initialized.")
                ray_connected = True
                break

            logger.info(f"Attempting Ray connection (address='auto', attempt {i+1}/{ray_init_retries})...")
            try:
                # Set dashboard_host to prevent warning logs if dashboard isn't needed here
                ray.init(address='auto', ignore_reinit_error=True, dashboard_host="")
                logger.info("Ray connection initialized successfully.")
                ray_connected = True
                break # Exit retry loop on success
            except (ConnectionError, ray.exceptions.RaySystemError) as e:
                logger.warning(f"Ray connection attempt {i+1} failed: {e}")
                if i < ray_init_retries - 1:
                    sleep_time = RETRY_DELAY_BASE_S * (i + 1) # Linear backoff for init
                    logger.info(f"Retrying Ray connection in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Failed to initialize Ray connection after {ray_init_retries} attempts. Exiting.")
                    return # Cannot proceed without Ray

        # Ensure connection succeeded
        if not ray_connected or not ray.is_initialized():
             logger.error("Ray connection failed or lost after retries. Exiting.")
             return

        node_id = ray.get_runtime_context().get_node_id()
        # Sanitize node_id for use in actor name if necessary (e.g., remove invalid chars)
        safe_node_id = node_id.replace("-", "_") # Example sanitization
        actor_name = f"network_monitor_{safe_node_id}"
        logger.info(f"Node ID: {node_id}. Target Actor Name: {actor_name}")

        # --- Get or Create the NetworkMonitor Actor ---
        actor_ready = False
        for attempt in range(MAX_ACTOR_CREATE_RETRIES):
            try:
                logger.info(f"Attempting to get or create NetworkMonitor actor: {actor_name} (Attempt {attempt + 1}/{MAX_ACTOR_CREATE_RETRIES})" )
                actor_options = {
                    "name": actor_name,
                    "lifetime": "detached",
                    "num_cpus": 0,
                    # TODO: Add resource requirements if NetworkMonitor needs GPUs/memory
                }

                # Try getting first in case it exists (e.g., script restart)
                try:
                    monitor_actor = ray.get_actor(actor_name)
                    logger.info(f"Found existing NetworkMonitor actor '{actor_name}'. Verifying...")
                    await monitor_actor.ping_self.remote()
                    logger.info(f"Existing actor '{actor_name}' is responsive.")
                    actor_ready = True

                except ValueError: # Actor doesn't exist
                    logger.info(f"Creating new NetworkMonitor actor: '{actor_name}'")
                    # TODO: Pass necessary config to NetworkMonitor if its constructor requires it
                    monitor_actor = ray.remote(NetworkMonitor).options(**actor_options).remote(
                        update_interval = 60,
                        sampling_ratio = 0.1,
                        tokens_per_second = 100,
                        burst = 1000,
                    )
                    # Allow some time for actor process to start
                    await asyncio.sleep(3)
                    await monitor_actor.ping_self.remote()
                    logger.info(f"Successfully created and verified new NetworkMonitor actor: '{actor_name}'")
                    actor_ready = True

                except Exception as e:
                     logger.warning(f"Verification of actor '{actor_name}' failed (Attempt {attempt + 1}): {e}", exc_info=True)
                     # Actor might exist but failed ping - try recreating next time if possible

                if actor_ready:
                    break # Exit retry loop

            except Exception as e:
                logger.error(f"Error during actor get/create/verify for '{actor_name}' (Attempt {attempt + 1}): {e}", exc_info=True)

            if not actor_ready and attempt < MAX_ACTOR_CREATE_RETRIES - 1:
                 delay = RETRY_DELAY_BASE_S * (2 ** attempt) # Exponential backoff
                 logger.info(f"Retrying in {delay} seconds...")
                 await asyncio.sleep(delay)
            elif not actor_ready:
                 logger.critical(f"Max retries ({MAX_ACTOR_CREATE_RETRIES}) reached. Failed to ensure NetworkMonitor actor '{actor_name}' is running. Exiting.")
                 return # Exit main function

        # --- Keep Script Alive & Monitor Actor ---
        logger.info(f"NetworkMonitor actor '{actor_name}' confirmed running. Entering keep-alive loop.")
        while keep_running:
            # Periodically check if the actor is still alive
            try:
                await asyncio.sleep(ACTOR_CHECK_INTERVAL_S) # Wait first
                if not keep_running: break # Check flag after sleep

                await monitor_actor.ping_self.remote()
                logger.debug(f"NetworkMonitor actor '{actor_name}' is alive.")
            except ray.exceptions.RayActorError:
                logger.error(f"NetworkMonitor actor '{actor_name}' is no longer reachable. Exiting script.")
                keep_running = False # Signal exit
            except asyncio.CancelledError:
                logger.info("Keep-alive loop cancelled.")
                keep_running = False # Ensure flag is set
            except Exception as e:
                 logger.exception(f"Unexpected error during actor health check for '{actor_name}': {e}")
                 # Decide if we should break or continue based on error severity
                 # For now, we'll keep running unless it's RayActorError
                 if not isinstance(e, ray.exceptions.RayError): # Check if it's a ray error besides ActorError
                      pass # Continue for non-fatal errors
                 else:
                      logger.error(f"Exiting keep-alive loop due to Ray error: {e}")
                      keep_running = False # Exit for other Ray errors

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception as e:
        logger.exception(f"Critical error in NetworkMonitor startup/management: {e}")
    finally:
        logger.info("start_network_monitor main task finished or errored. keep_running={keep_running}")
        # Signal that shutdown is complete
        shutdown_event.set()


def handle_signal(sig, frame):
    """Gracefully handle termination signals."""
    global keep_running
    # Prevent multiple signal handling
    if not keep_running:
        logger.info(f"Signal {sig} received, but shutdown already in progress.")
        return
    logger.info(f"Received signal {sig}. Initiating shutdown sequence. Setting keep_running=False.")
    keep_running = False # Signal the main loop to exit
    # If main_task exists and is running, cancel it
    if main_task and not main_task.done():
         logger.info(f"Cancelling main task due to signal {sig}.")
         main_task.cancel()
    # The main loop should exit due to keep_running flag or cancellation

if __name__ == "__main__":
    # --- Signal Handling Setup ---
    # Use loop.add_signal_handler for better asyncio integration if possible,
    # but standard signal.signal is more portable in basic scripts.
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info(f"Starting Network Monitor deployment script (PID: {pid}). Logging to {log_file}")

    loop = asyncio.get_event_loop()
    main_task = loop.create_task(main())

    try:
        loop.run_until_complete(main_task)
        # If main exits normally (e.g., actor died), wait for potential cleanup signal
        # loop.run_until_complete(shutdown_event.wait())
    except KeyboardInterrupt:
         logger.info("KeyboardInterrupt caught, stopping.")
         # Ensure handle_signal is called if loop was interrupted
         if keep_running: # Check if signal handler already ran
              handle_signal(signal.SIGINT, None)
         # Wait for main task to finish cancellation
         loop.run_until_complete(main_task)
    except asyncio.CancelledError:
         logger.info("Main task was cancelled externally.")
         # Allow finally block in main to run
    except Exception as e:
         logger.exception("Unhandled exception during script execution.")
    finally:
        # Final cleanup if needed
        if ray.is_initialized():
            # Generally not needed to disconnect here when run as a background process
            # ray.disconnect()
            pass
        logger.info(f"Network Monitor deployment script (PID: {pid}) finished.")
