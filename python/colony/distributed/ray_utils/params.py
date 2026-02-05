import os
from pathlib import Path
import yaml

# Get the directory containing this file
current_dir = Path(__file__).parent

# Load configuration from ray_params.yaml in the same directory
with open(current_dir / "ray_params.yaml") as f:
    content = f.read()

# Replace template variables with environment variables
app_mount_path = os.getenv("APP_MOUNT_PATH", "/app")
content = content.replace("${APP_MOUNT_PATH}", app_mount_path)

ray_params = yaml.safe_load(content)

ray_vmr_exec_config = ray_params.get("ray_vmr_exec_config", {})
ray_repo_stats_collector_config = ray_params.get("ray_repo_stats_collector_config", {})
ray_global_code_analysis_config = ray_params.get("ray_global_code_analysis_config", {})
ray_local_code_analysis_config = ray_params.get("ray_local_code_analysis_config", {})
