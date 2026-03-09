from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache


logger = logging.getLogger(__name__)

################################################################################
# Language Optimizations
################################################################################


class LanguageFeature(Enum):
    """Language-specific features that affect splitting
    Features that can be enabled/disabled per language"""

    IMPORTS = auto()
    FUNCTIONS = auto()
    CLASSES = auto()
    STATEMENTS = auto()
    DOCSTRINGS = auto()
    COMMENTS = auto()
    DECORATORS = auto()
    TEMPLATES = auto()
    MACROS = auto()
    PREPROCESSOR = auto()
    NESTED_FUNCTIONS = auto()
    ASYNC_SUPPORT = auto()
    ANNOTATIONS = auto()
    GENERICS = auto()
    LAMBDAS = auto()
    ATTRIBUTES = auto()
    INNER_CLASSES = auto()


@dataclass
class LanguageOptimization:
    """Language-specific optimization settings"""

    features: set[LanguageFeature]
    scope_markers: dict[str, str]  # start/end markers for scopes
    import_patterns: list[str]
    skip_patterns: list[str]
    merge_patterns: list[str]
    context_patterns: dict[str, int]  # pattern -> required context lines
    max_scope_depth: int
    parse_timeout: float


class LanguageOptimizer:
    """Handles language-specific code splitting optimizations"""

    def __init__(self):
        self.optimizations = {
            "python": LanguageOptimization(
                features={
                    LanguageFeature.NESTED_FUNCTIONS,
                    LanguageFeature.DECORATORS,
                    LanguageFeature.ASYNC_SUPPORT,
                    LanguageFeature.ANNOTATIONS,
                    LanguageFeature.LAMBDAS,
                    LanguageFeature.INNER_CLASSES,
                },
                scope_markers={
                    "def": ":",
                    "class": ":",
                    "async def": ":",
                    "with": ":",
                    "if": ":",
                    "for": ":",
                    "while": ":",
                },
                import_patterns=[r"^import\s+.*$", r"^from\s+.*\s+import\s+.*$"],
                skip_patterns=[
                    r"^\s*#.*$",  # Comments
                    r'^\s*""".*?"""$',  # Docstrings
                    r"^\s*$",  # Empty lines
                ],
                merge_patterns=[
                    r"@\w+(\(.*\))?$",  # Decorators
                    r"^elif\s+.*:$",  # elif statements
                    r"^except\s+.*:$",  # except clauses
                ],
                context_patterns={
                    r"@\w+": 1,  # Decorators need previous line
                    r"else\s*:": 1,  # else needs if
                    r"elif\s+.*:": 1,  # elif needs if/elif
                    r"except\s+.*:": 1,  # except needs try
                    r"finally\s*:": 1,  # finally needs try
                },
                max_scope_depth=10,
                parse_timeout=5.0,
            ),
            "javascript": LanguageOptimization(
                features={
                    LanguageFeature.ASYNC_SUPPORT,
                    LanguageFeature.DECORATORS,
                    LanguageFeature.LAMBDAS,
                    LanguageFeature.INNER_CLASSES,
                },
                scope_markers={
                    "function": "{",
                    "class": "{",
                    "if": "{",
                    "for": "{",
                    "while": "{",
                    "try": "{",
                    "switch": "{",
                },
                import_patterns=[
                    r"^import\s+.*$",
                    r"^export\s+.*$",
                    r'require\([\'"].*[\'"]\)',
                ],
                skip_patterns=[
                    r"^\s*//.*$",  # Line comments
                    r"^\s*/\*.*?\*/\s*$",  # Block comments
                    r"^\s*$",  # Empty lines
                ],
                merge_patterns=[
                    r"@\w+(\(.*\))?$",  # Decorators
                    r"^\s*else\s*{$",  # else blocks
                    r"^\s*catch\s*\(.*\)\s*{$",  # catch blocks
                ],
                context_patterns={
                    r"else\s*{": 1,
                    r"catch\s*\(.*\)\s*{": 1,
                    r"finally\s*{": 1,
                },
                max_scope_depth=8,
                parse_timeout=4.0,
            ),
            "java": LanguageOptimization(
                features={
                    LanguageFeature.ANNOTATIONS,
                    LanguageFeature.GENERICS,
                    LanguageFeature.INNER_CLASSES,
                    LanguageFeature.LAMBDAS,
                },
                scope_markers={
                    "class": "{",
                    "interface": "{",
                    "enum": "{",
                    "record": "{",
                    "if": "{",
                    "for": "{",
                    "while": "{",
                },
                import_patterns=[r"^import\s+.*$", r"^package\s+.*$"],
                skip_patterns=[r"^\s*//.*$", r"^\s*/\*.*?\*/\s*$", r"^\s*$"],
                merge_patterns=[
                    r"@\w+(\(.*\))?$",
                    r"^\s*else\s*{$",
                    r"^\s*catch\s*\(.*\)\s*{$",
                ],
                context_patterns={r"@\w+": 1, r"else\s*{": 1, r"catch\s*\(.*\)\s*{": 1},
                max_scope_depth=6,
                parse_timeout=3.0,
            ),
            "cpp": LanguageOptimization(
                features={
                    LanguageFeature.TEMPLATES,
                    LanguageFeature.MACROS,
                    LanguageFeature.PREPROCESSOR,
                    LanguageFeature.NESTED_FUNCTIONS,
                },
                scope_markers={
                    "class": "{",
                    "struct": "{",
                    "namespace": "{",
                    "if": "{",
                    "for": "{",
                    "while": "{",
                },
                import_patterns=[r'#include\s+[<"].*[>"]', r"using\s+namespace\s+.*$"],
                skip_patterns=[r"^\s*//.*$", r"^\s*/\*.*?\*/\s*$", r"^\s*$"],
                merge_patterns=[r"#if\s+.*$", r"#else\s*$", r"#elif\s+.*$"],
                context_patterns={r"#else": 1, r"#elif": 1, r"else\s*{": 1},
                max_scope_depth=5,
                parse_timeout=3.0,
            ),
            # TODO: Add optimizations for:
            # - Rust
            # - Go
            # - TypeScript
            # - Ruby
            # - PHP
            # - C#
            # - Swift
            # - Kotlin
            # - Scala
        }

        self._compile_patterns()

    def _compile_patterns(self):
        """Precompile regex patterns for performance"""
        for lang, opt in self.optimizations.items():
            opt.import_patterns = [re.compile(p) for p in opt.import_patterns]
            opt.skip_patterns = [re.compile(p) for p in opt.skip_patterns]
            opt.merge_patterns = [re.compile(p) for p in opt.merge_patterns]
            opt.context_patterns = {
                re.compile(p): n for p, n in opt.context_patterns.items()
            }

    @lru_cache(maxsize=1000)
    def get_optimization(self, language: str) -> LanguageOptimization | None:
        """Get optimization settings for a language"""
        return self.optimizations.get(language)

    def should_merge_lines(self, lines: list[str], language: str) -> bool:
        """Check if lines should be merged based on language patterns"""
        opt = self.get_optimization(language)
        if not opt:
            return False

        for pattern in opt.merge_patterns:
            if pattern.match(lines[-1].strip()):
                return True
        return False

    def get_required_context(self, line: str, language: str) -> int:
        """Get number of context lines needed for this line"""
        opt = self.get_optimization(language)
        if not opt:
            return 0

        for pattern, context_lines in opt.context_patterns.items():
            if pattern.match(line.strip()):
                return context_lines
        return 0

    def is_scope_start(self, line: str, language: str) -> bool:
        """Check if line starts a new scope"""
        opt = self.get_optimization(language)
        if not opt:
            return False

        line = line.strip()
        return any(
            line.startswith(start) and line.endswith(end)
            for start, end in opt.scope_markers.items()
        )

    # TODO: Add methods for:
    # - Template/generic parameter handling
    # - Macro expansion tracking
    # - Import dependency analysis
    # - Scope nesting analysis
    # - Language-specific comment handling
    # - Custom splitting rules per language


################################################################################
# Language Rules
################################################################################


@dataclass
class LanguageConfig:
    """Per-language configuration"""

    enabled_features: set[LanguageFeature] = field(default_factory=set)
    max_node_size: int = 1024 * 512  # 512KB max for single node
    min_node_size: int = 1024  # 1KB min to avoid over-splitting
    parse_timeout: int = 5  # seconds
    cache_parsed_ast: bool = True

    # Performance tuning
    skip_large_nodes: bool = True
    large_node_threshold: int = 1024 * 256  # 256KB
    enable_parallel_parsing: bool = False

    # Custom patterns
    additional_split_patterns: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)


class LanguageRules:
    """Language-specific splitting rules and patterns"""

    def __init__(self):
        self.rules = self._init_language_rules()

    def _init_language_rules(self) -> dict[str, dict]:
        """Initialize rules for all supported languages"""
        return {
            "python": {
                "splittable_nodes": {
                    "function_definition",
                    "class_definition",
                    "if_statement",
                    "for_statement",
                    "while_statement",
                    "try_statement",
                    "with_statement",
                    "match_statement",  # Python 3.10+
                    "async_function_definition",
                    "async_with_statement",
                    "async_for_statement",
                },
                "import_patterns": [r"^import\s+.*$", r"^from\s+.*\s+import\s+.*$"],
                "docstring_patterns": [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
                # TODO: Add more Python-specific patterns
            },
            "javascript": {
                "splittable_nodes": {
                    "function_declaration",
                    "class_declaration",
                    "method_definition",
                    "arrow_function",
                    "if_statement",
                    "for_statement",
                    "while_statement",
                    "try_statement",
                    "switch_statement",
                    "export_statement",
                },
                "import_patterns": [
                    r"^import\s+.*$",
                    r"^export\s+.*$",
                    r'require\([\'"].*[\'"]\)',
                ],
                # TODO: Add JSDoc patterns
            },
            "java": {
                "splittable_nodes": {
                    "method_declaration",
                    "class_declaration",
                    "interface_declaration",
                    "constructor_declaration",
                    "static_initializer",
                    "if_statement",
                    "for_statement",
                    "while_statement",
                    "try_statement",
                    "switch_statement",
                },
                "import_patterns": [r"^import\s+.*$", r"^package\s+.*$"],
                "annotation_patterns": [r"@\w+(\s*\([^)]*\))?"],
                # TODO: Add Javadoc patterns
            },
            "cpp": {
                "splittable_nodes": {
                    "function_definition",
                    "class_specifier",
                    "namespace_definition",
                    "if_statement",
                    "for_statement",
                    "while_statement",
                    "try_statement",
                    "switch_statement",
                    "template_declaration",
                },
                "preprocessor_patterns": [
                    r'#include\s+[<"].*[>"]',
                    r"#define\s+.*$",
                    r"#ifdef\s+.*$",
                    r"#ifndef\s+.*$",
                ],
                # TODO: Add more C++-specific patterns
            },
            "rust": {
                "splittable_nodes": {
                    "function_item",
                    "struct_item",
                    "enum_item",
                    "impl_item",
                    "trait_item",
                    "if_expression",
                    "for_expression",
                    "while_expression",
                    "loop_expression",
                    "match_expression",
                },
                "attribute_patterns": [r"#\[.*\]"],
                # TODO: Add more Rust-specific patterns
            },
            # TODO: Add support for:
            # - Go
            # - TypeScript
            # - Ruby
            # - PHP
            # - C#
            # - Swift
            # - Kotlin
            # - Scala
            # - And more...
        }

    def get_language_rules(self, language: str, config: LanguageConfig) -> dict:
        """Get rules for a specific language with configuration applied"""
        base_rules = self.rules.get(language, {})
        if not base_rules:
            return {}

        # Apply configuration
        rules = base_rules.copy()

        # Filter features based on config
        if LanguageFeature.IMPORTS not in config.enabled_features:
            rules.pop("import_patterns", None)
        if LanguageFeature.DOCSTRINGS not in config.enabled_features:
            rules.pop("docstring_patterns", None)
        # TODO: ... etc for other features

        # Add custom patterns
        if config.additional_split_patterns:
            rules["custom_split_patterns"] = config.additional_split_patterns

        return rules

    def should_split_node(
        self, node_type: str, language: str, config: LanguageConfig
    ) -> bool:
        """Determine if a node should be split based on rules and config"""
        rules = self.get_language_rules(language, config)

        # Skip large nodes if configured
        if (
            config.skip_large_nodes
            and hasattr(node_type, "size")
            and node_type.size > config.large_node_threshold
        ):
            return True

        return node_type in rules.get("splittable_nodes", set()) or any(
            re.match(pattern, node_type)
            for pattern in rules.get("custom_split_patterns", [])
        )

    # TODO: Add methods for:
    # - Detecting and handling imports
    # - Processing docstrings and comments
    # - Handling language-specific features (decorators, annotations, etc.)
    # - Optimizing parsing for specific languages
    # - Caching frequently used patterns
    # - Parallel processing of large files
    # - Custom splitting rules for specific projects/organizations


@dataclass
class CustomRule:
    """Defines a custom splitting rule"""

    name: str
    pattern: str | re.Pattern
    priority: int = 0  # Higher priority rules are checked first
    min_context_lines: int = 0  # Lines of context to keep before/after match
    max_size: int | None = None  # Max size for segments created by this rule
    languages: set[str] = field(default_factory=set)  # Empty means all languages
    condition: Callable[[str, str], bool] | None = None  # Optional custom condition

    def __post_init__(self):
        if isinstance(self.pattern, str):
            self.compiled_pattern = re.compile(self.pattern, re.MULTILINE | re.DOTALL)
        else:
            # Pattern is already compiled
            self.compiled_pattern = self.pattern


@dataclass
class RuleMatch:
    """Represents a match found by a splitting rule"""

    start_pos: int
    end_pos: int
    start_line: int
    end_line: int
    rule: CustomRule
    context: str
    priority: int


class RuleManager:
    """Manages custom splitting rules"""

    def __init__(self):
        self.rules: dict[str, CustomRule] = {}
        self.language_optimizer = LanguageOptimizer()
        self._add_default_rules()

    def _add_basic_rules(self):
        """Add default splitting rules"""
        self.add_rule(
            CustomRule(
                name="class_definition",
                pattern=r"^\s*class\s+\w+[^\n]*:",
                priority=100,
                min_context_lines=1,
                languages={"python"},
            )
        )

        self.add_rule(
            CustomRule(
                name="function_definition",
                pattern=r"^\s*(?:async\s+)?def\s+\w+[^\n]*:",
                priority=90,
                min_context_lines=1,
                languages={"python"},
            )
        )

        self.add_rule(
            CustomRule(
                name="js_class",
                pattern=r"^\s*(?:export\s+)?class\s+\w+[^\n]*{",
                priority=100,
                min_context_lines=1,
                languages={"javascript", "typescript"},
            )
        )

        # Add more default rules as needed

    def _add_default_rules(self):
        """Add default and language-specific rules"""
        # Add basic rules
        self._add_basic_rules()

        # Add language-specific rules from optimizer
        for language, opt in self.language_optimizer.optimizations.items():
            # Convert scope markers to rules
            for scope_start, scope_end in opt.scope_markers.items():
                self.add_rule(
                    CustomRule(
                        name=f"{language}_scope_{scope_start}",
                        pattern=f"^\\s*{scope_start}.*{scope_end}",
                        priority=100,
                        min_context_lines=1,
                        languages={language},
                    )
                )

            # Convert import patterns to rules
            for pattern in opt.import_patterns:
                self.add_rule(
                    CustomRule(
                        name=f"{language}_import_{hash(pattern)}",
                        pattern=pattern,
                        priority=90,
                        languages={language},
                    )
                )

            # Add merge patterns as negative rules (prevent splitting)
            for pattern in opt.merge_patterns:
                self.add_rule(
                    CustomRule(
                        name=f"{language}_merge_{hash(pattern)}",
                        pattern=pattern,
                        priority=-1,  # Negative priority to prevent splits
                        min_context_lines=opt.context_patterns.get(pattern, 0),
                        languages={language},
                    )
                )

    def get_rules_for_language(self, language: str) -> list[CustomRule]:
        """Get all rules applicable to a language"""
        # Get basic rules
        rules = [
            rule
            for rule in self.rules.values()
            if not rule.languages or language in rule.languages
        ]

        # Get optimization settings
        opt = self.language_optimizer.get_optimization(language)
        if opt:
            # Add dynamic rules based on language features
            if LanguageFeature.TEMPLATES in opt.features:
                rules.extend(self._get_template_rules(language))
            if LanguageFeature.MACROS in opt.features:
                rules.extend(self._get_macro_rules(language))
            if LanguageFeature.PREPROCESSOR in opt.features:
                rules.extend(self._get_preprocessor_rules(language))

        return rules

    def add_rule(self, rule: CustomRule):
        """Add a new custom rule"""
        if rule.name in self.rules:
            logger.warning(f"Overwriting existing rule: {rule.name}")
        self.rules[rule.name] = rule

    def remove_rule(self, rule_name: str):
        """Remove a custom rule"""
        self.rules.pop(rule_name, None)

    def get_rules_for_language(self, language: str) -> list[CustomRule]:
        """Get all rules applicable to a language"""
        return [
            rule
            for rule in self.rules.values()
            if not rule.languages or language in rule.languages
        ]

    # TODO: Add methods for:
    # - Rule validation and optimization
    # - Rule performance monitoring
    # - Rule caching
    # - Custom rule import/export
    # - Project-specific rule sets
