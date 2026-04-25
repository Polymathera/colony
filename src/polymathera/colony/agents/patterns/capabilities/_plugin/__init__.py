"""Private helpers for ``UserPluginCapability``.

Kept out of the top-level ``capabilities`` namespace so the public
action surface (``UserPluginCapability``) is the only entry point
users import from.
"""

from .schema import PluginSpec, SkillParam, SkillSource, SkillSpec
from .discovery import (
    DiscoveryResult,
    discover_plugins,
    discover_skills,
    parse_frontmatter,
    resolve_skill_path,
)

__all__ = [
    "SkillParam",
    "SkillSource",
    "SkillSpec",
    "PluginSpec",
    "DiscoveryResult",
    "discover_skills",
    "discover_plugins",
    "parse_frontmatter",
    "resolve_skill_path",
]
