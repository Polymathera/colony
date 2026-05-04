"""Custom-deployment plug-in surface (HPC, AWS-CDK, Slurm, ...).

Importing this package registers :class:`CustomDeploymentsConfig` with the
shared config registry (side-effect of ``configs.py``) and exposes the
``CustomDeployment`` protocol + ``register_custom_deployment`` decorator.
"""

from .configs import CustomDeploymentSpec, CustomDeploymentsConfig
from .custom import (
    CustomDeployment,
    DeploymentContext,
    get_custom_deployment_class,
    list_custom_deployments,
    register_custom_deployment,
)

__all__ = (
    "CustomDeployment",
    "CustomDeploymentSpec",
    "CustomDeploymentsConfig",
    "DeploymentContext",
    "get_custom_deployment_class",
    "list_custom_deployments",
    "register_custom_deployment",
)
