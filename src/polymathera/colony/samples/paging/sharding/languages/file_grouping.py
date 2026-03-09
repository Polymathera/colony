"""
Language-specific configuration for file grouping
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from re import Pattern

###############################################################################
# Language Configs
###############################################################################

# Common language binding pairs (extensible)
STRONG_LANGUAGE_BINDINGS = {
    frozenset(["python", "c"]),
    frozenset(["python", "cpp"]),
    frozenset(["typescript", "javascript"]),
    frozenset(["java", "kotlin"]),
    frozenset(["c", "cpp"]),
    frozenset(["objective-c", "cpp"]),
    # TODO: Add more common language pairs
}


class LanguageFeature(Enum):
    NESTED_FUNCTIONS = auto()
    DECORATORS = auto()
    ASYNC_SUPPORT = auto()
    TEMPLATES = auto()
    MACROS = auto()
    IMPORTS = auto()
    ANNOTATIONS = auto()
    GENERICS = auto()
    LAMBDAS = auto()
    PREPROCESSOR = auto()
    ATTRIBUTES = auto()
    INNER_CLASSES = auto()
    MODULES = auto()
    INTERFACES = auto()
    MIXINS = auto()
    PARTIAL_CLASSES = auto()


@dataclass
class LanguageConfig:
    """Language-specific configuration"""

    features: set[LanguageFeature]
    file_extensions: set[str]
    import_patterns: list[Pattern]
    scope_patterns: dict[str, Pattern]
    context_patterns: dict[Pattern, int]
    merge_patterns: list[Pattern]
    skip_patterns: list[Pattern]
    dependency_patterns: dict[str, Pattern]

    # Token estimation factors
    avg_tokens_per_char: float = 0.25
    avg_tokens_per_line: float = 8.0

    # Language-specific limits
    max_line_length: int = 120
    max_function_length: int = 100
    max_class_length: int = 1000

    @classmethod
    def create_defaults(cls) -> dict[str, LanguageConfig]:
        """Create default configurations for supported languages"""
        return {
            "python": cls(
                features={
                    LanguageFeature.NESTED_FUNCTIONS,
                    LanguageFeature.DECORATORS,
                    LanguageFeature.ASYNC_SUPPORT,
                    LanguageFeature.ANNOTATIONS,
                    LanguageFeature.LAMBDAS,
                    LanguageFeature.INNER_CLASSES,
                    LanguageFeature.MODULES,
                },
                file_extensions={".py", ".pyi", ".pyx"},
                import_patterns=[
                    re.compile(r"^import\s+.*$"),
                    re.compile(r"^from\s+.*\s+import\s+.*$"),
                ],
                scope_patterns={
                    "class": re.compile(r"^\s*class\s+\w+.*:$"),
                    "function": re.compile(r"^\s*(?:async\s+)?def\s+\w+.*:$"),
                    "method": re.compile(r"^\s*(?:async\s+)?def\s+\w+.*:$"),
                },
                context_patterns={
                    re.compile(r"^\s*@\w+"): 1,  # Decorators
                    re.compile(r"^\s*class\s+.*\(.*\):\s*$"): 1,  # Class inheritance
                    re.compile(
                        r"^\s*def\s+.*\(.*\)\s*->\s*.+:\s*$"
                    ): 1,  # Type annotations
                },
                merge_patterns=[
                    re.compile(r"^\s*elif\s+.*:$"),
                    re.compile(r"^\s*else\s*:$"),
                    re.compile(r"^\s*except\s+.*:$"),
                    re.compile(r"^\s*finally\s*:$"),
                ],
                skip_patterns=[
                    re.compile(r"^\s*#.*$"),
                    re.compile(r'^\s*""".*?"""$', re.DOTALL),
                    re.compile(r"^\s*$"),
                ],
                dependency_patterns={
                    "class_usage": re.compile(r"\b(\w+)\s*\([^)]*\)"),
                    "function_call": re.compile(r"\b(\w+)\s*\([^)]*\)"),
                    "inheritance": re.compile(r"class\s+\w+\s*\(\s*(\w+)\s*[,)]"),
                    "type_annotation": re.compile(r":\s*(\w+)(?:\[.*\])?"),
                },
            ),
            "typescript": cls(
                features={
                    LanguageFeature.INTERFACES,
                    LanguageFeature.GENERICS,
                    LanguageFeature.DECORATORS,
                    LanguageFeature.MODULES,
                    LanguageFeature.LAMBDAS,
                },
                file_extensions={".ts", ".tsx"},
                import_patterns=[
                    re.compile(r"^import\s+.*$"),
                    re.compile(r"^export\s+.*$"),
                ],
                scope_patterns={
                    "interface": re.compile(r"^\s*interface\s+\w+.*{$"),
                    "class": re.compile(r"^\s*class\s+\w+.*{$"),
                    "function": re.compile(r"^\s*(?:async\s+)?function\s+\w+.*{$"),
                    "method": re.compile(r"^\s*(?:async\s+)?\w+\s*\(.*\)\s*{$"),
                },
                context_patterns={
                    re.compile(r"^\s*@\w+"): 1,  # Decorators
                    re.compile(
                        r"^\s*interface\s+.*extends\s+.*{$"
                    ): 1,  # Interface inheritance
                    re.compile(r"^\s*type\s+.*=\s*.*$"): 1,  # Type aliases
                },
                merge_patterns=[
                    re.compile(r"^\s*else\s+if\s*\(.*\)\s*{$"),
                    re.compile(r"^\s*else\s*{$"),
                    re.compile(r"^\s*catch\s*\(.*\)\s*{$"),
                    re.compile(r"^\s*finally\s*{$"),
                ],
                skip_patterns=[
                    re.compile(r"^\s*//.*$"),
                    re.compile(r"^\s*/\*.*?\*/$", re.DOTALL),
                    re.compile(r"^\s*$"),
                ],
                dependency_patterns={
                    "type_usage": re.compile(r":\s*(\w+)(?:<.*>)?"),
                    "interface_impl": re.compile(r"implements\s+(\w+)"),
                    "class_usage": re.compile(r"new\s+(\w+)\s*\("),
                    "import_type": re.compile(r"import\s+type\s*{\s*(\w+)\s*}"),
                },
            ),
            # TODO: Add more languages...
        }
