from __future__ import annotations

import re
from dataclasses import field
from re import Pattern
from typing import ClassVar

from colony.distributed.config import register_polymathera_config

from ..analyzers.base import AnalyzerConfig


@register_polymathera_config()
class DependencyConfig(AnalyzerConfig):
    """Configuration for dependency analysis
    TODO: Add more language patterns
    TODO: Enhance cross-language detection
    TODO: Add more cross-language path mappings
    TODO: Add more language-specific path resolution
    TODO: Add more sophisticated confidence scoring
    TODO: Implement additional optimization strategies
    TODO: Implement additional file handling features
    """

    # Language-specific patterns
    language_patterns: dict[str, dict[str, list[Pattern]]] = field(
        default_factory=lambda: {
            "python": {
                "import": [
                    re.compile(r"^from\s+([\w.]+)\s+import"),
                    re.compile(r"^import\s+([\w.]+)"),
                ],
                "class": [
                    re.compile(r"class\s+\w+\s*\(([\w.,\s]+)\):"),
                    re.compile(r"@\w+\.(\w+)"),
                ],
                "function": [
                    re.compile(r"def\s+\w+\s*\(([\w\s,.:]+)\)"),
                    re.compile(r"@(\w+[\w.]*)"),
                ],
            },
            "typescript": {
                "import": [
                    re.compile(r'import.*from\s+[\'"](.+)[\'"]'),
                    re.compile(r'require\([\'"](.+)[\'"]\)'),
                ],
                "class": [
                    re.compile(r"class\s+\w+\s+implements\s+([\w,\s]+)"),
                    re.compile(r"class\s+\w+\s+extends\s+([\w]+)"),
                ],
                "interface": [re.compile(r"interface\s+\w+\s+extends\s+([\w,\s]+)")],
            },
            # Add more language patterns as needed
        }
    )

    # Cross-language bindings
    known_bindings: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: {
            "python-typescript": {
                "decorators": ["@typescript", "@js_export"],
                "imports": ["from typescript import", "import js_module"],
                "markers": ["# @ts-check", "# @ts-ignore"],
            },
            "typescript-python": {
                "decorators": ["@python", "@py_import"],
                "imports": ["import * as py from", 'require("python-bridge")'],
                "markers": ["// @py-check", "// @py-ignore"],
            },
            # Add more language pairs
        }
    )

    # Analysis settings
    min_confidence: float = 0.6
    enable_deep_analysis: bool = True
    batch_size: int = 100

    CONFIG_PATH: ClassVar[str] = "llms.sharding.analyzers.dependency"
