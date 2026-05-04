"""Typed config slot for custom deployments (HPC, AWS-CDK, ...).

The operator YAML names instances and points each at a registered handler::

    custom_deployments:
      deployments:
        cps_hpc_aero:
          handler: aws_cdk_hpc          # name passed to register_custom_deployment
          auto_provision: true
          params:
            stack_name: polymath-hpc-prod
            region: us-west-2
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..distributed.config import (
    ConfigComponent,
    Mutability,
    Ownership,
    Tier,
    register_polymathera_config,
    tier_metadata,
)


class CustomDeploymentSpec(BaseModel):
    """Per-instance entry in :class:`CustomDeploymentsConfig`."""

    handler: str
    enabled: bool = True
    auto_provision: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


@register_polymathera_config(path="custom_deployments")
class CustomDeploymentsConfig(ConfigComponent):
    """Mapping ``instance_name → CustomDeploymentSpec``.

    The instance name doubles as the L4 ``runtime_overlays`` scope key under
    which a successful ``provision()`` writes its returned runtime values.
    Empty by default — colony ships no built-in custom deployments; CPS or
    other extensions populate this via operator YAML.
    """

    deployments: dict[str, CustomDeploymentSpec] = Field(
        default_factory=dict,
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR,
            ownership=Ownership.EXTENSION,
            mutability=Mutability.RELOADABLE,
        ),
    )


__all__ = (
    "CustomDeploymentSpec",
    "CustomDeploymentsConfig",
)
