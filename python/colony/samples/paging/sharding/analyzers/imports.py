from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from re import Pattern
from typing import Any, ClassVar

from overrides import override

from colony.distributed.config import register_polymathera_config
from colony.distributed import get_polymathera
from colony.distributed.metrics.common import BaseMetricsMonitor
from colony.utils import setup_logger, run_method_once

from ..languages.imports import get_language_configs
from ..languages.utils import detect_language, is_comment
from .base import AnalyzerConfig, BaseAnalyzer, FileContentCache

logger = setup_logger(__name__)


@register_polymathera_config()
class ImportConfig(AnalyzerConfig):
    """Configuration for import analysis"""

    # Analysis settings
    analyze_type_imports: bool = True
    analyze_runtime_imports: bool = True
    track_transitive_imports: bool = True
    max_transitive_depth: int = 3

    # Performance settings
    use_parallel_resolution: bool = True

    # Language-specific settings
    language_configs: dict[str, dict] = field(
        default_factory=lambda: get_language_configs()
    )

    CONFIG_PATH: ClassVar[str] = "llms.sharding.analyzers.imports"


# TODO: Implement language-specific import parsing methods
# TODO: Add support for more languages
# TODO: Enhance cross-language import detection
# TODO: Add import validation and verification
# TODO: Add circular import detection
# TODO: Enhance the import resolution
# TODO: Implement relative import resolution
# TODO: Add more language-specific resolution strategies
# TODO: Add more sophisticated path resolution
# TODO: Enhance caching for large codebases
# TODO: Add more sophisticated caching
# TODO: Implement caching for resolved paths
# TODO: Enhance the error handling



class ImportAnalyzerMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing ImportAnalyzerMetricsMonitor instance {id(self)}...")
        self.import_counts = self.create_counter(
            "import_analyzer_imports_total",
            "Number of imports analyzed",
            labelnames=["language", "type"],
        )
        self.analysis_duration = self.create_histogram(
            "import_analyzer_duration_seconds",
            "Time spent analyzing imports",
            labelnames=["language"],
        )
        self.resolver_calls = self.create_counter(
            "import_resolver_calls_total",
            "Number of resolver calls",
            labelnames=["language", "status"],
        )
        self.resolver_latency = self.create_histogram(
            "import_resolver_latency_seconds",
            "Resolver operation latency",
            labelnames=["language"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
        self.resolver_errors = self.create_counter(
            "import_resolver_errors_total",
            "Number of resolver errors",
            labelnames=["language", "error_type"],
        )
        # Add batch-specific metrics
        self.batch_size = self.create_histogram(
            "import_resolver_batch_size",
            "Size of import resolution batches",
            labelnames=["language"],
        )
        self.batch_duration = self.create_histogram(
            "import_resolver_batch_duration_seconds",
            "Time to resolve import batches",
            labelnames=["language"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        self.batch_success_rate = self.create_gauge(
            "import_resolver_batch_success_rate",
            "Success rate of batch import resolution",
            labelnames=["language"],
        )
        self.concurrent_batches = self.create_gauge(
            "import_resolver_concurrent_batches",
            "Number of concurrent import resolution batches",
            labelnames=["language"],
        )


class ImportAnalyzer(BaseAnalyzer):
    """Analyzes module and package imports across languages

    1. ImportAnalyzer: Focuses on explicit module/package imports
        - Module imports
        - Package imports
        - Type imports
        - Runtime vs compile-time imports
    2. DependencyAnalyzer: Focuses on code-level dependencies
        - Function calls
        - Class inheritance
        - Interface implementations
        - Variable usage
        - Cross-language bindings
    """

    def __init__(self, file_content_cache: FileContentCache, config: ImportConfig | None = None):
        super().__init__("imports", file_content_cache)
        self.config = config
        self.patterns: dict = {}
        self.metrics = ImportAnalyzerMetricsMonitor()

        # Create resolver map
        self.resolvers = {
            "python": self._resolve_python_import,
            "typescript": self._resolve_js_import,
            "javascript": self._resolve_js_import,
            "java": self._resolve_java_import,
            "kotlin": self._resolve_kotlin_import,
            "rust": self._resolve_rust_import,
            "go": self._resolve_go_import,
            "swift": self._resolve_swift_import,
        }

    async def initialize(self):
        self.config = await ImportConfig.check_or_get_component(self.config)
        self._compile_patterns()
        await super().initialize()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        self.patterns = {}
        for lang, config in self.config.language_configs.items():
            self.patterns[lang] = {
                "imports": [re.compile(p) for p in config["import_patterns"]],
                "types": [re.compile(p) for p in config.get("type_patterns", [])],
            }

    @override
    async def _analyze_file_impl(
        self, file_path: str, content: str, language: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Analyze imports in a file.

        Performance: O(n) where n is file lines
        Memory: O(m) where m is number of imports

        Returns:
            dict containing:
                - 'direct': Set of direct imports
                - 'types': Set of type imports
                - 'runtime': Set of runtime imports
                - 'transitive': Set of transitive imports
        """
        imports = await self._analyze_file_imports(
            content, language, file_path
        )

        if self.config.track_transitive_imports:
            imports["transitive"] = await self._analyze_transitive_imports(
                imports["direct"], language, file_path, depth=0
            )

        return imports

    async def _analyze_file_imports(
        self, content: str, language: str, file_path: str
    ) -> dict[str, set[str]]:
        """
        Analyze imports in file content with enhanced metadata for FileGrouper

        Returns:
            dict containing:
            - direct: Set of direct import paths
            - types: Set of type-only imports
            - runtime: Set of runtime imports
            - cross_language: Set of cross-language imports
            - metadata: dict with additional import info:
                - import_type: 'direct' | 'relative' | 'wildcard' | 'aliased'
                - is_optional: bool  # Optional/dynamic imports
                - is_conditional: bool  # Conditional imports
                - line_numbers: list[int]  # Where imports occur
                - context: str | None  # Import context (e.g., inside function)
        """
        if self.config.use_parallel_resolution:
            imports = await self._analyze_imports_uncached_parallel(
                content, language, file_path
            )
        else:
            imports = await self._analyze_imports_uncached(
                content, language, file_path
            )

        return imports

    @run_method_once
    def _warn_missing_language_patterns(self, language: str) -> None:
        """Warn if no patterns are found for a language"""
        logger.warning(f"No patterns found for language: {language}")

    async def _analyze_imports_uncached(
        self, content: str, language: str, file_path: str
    ) -> dict[str, set[str]]:
        """
        Analyze imports in file content without using cache

        Args:
            content: File content to analyze
            language: Programming language of the file
            file_path: Path to the file (for resolution)

        Returns:
            dict containing:
            - direct: All direct imports
            - types: Type-only imports
            - runtime: Runtime imports
            - cross_language: Cross-language imports
        """
        try:
            imports = {
                "direct": set(),
                "types": set(),
                "runtime": set(),
                "cross_language": set(),
                "metadata": None,
            }

            lang_patterns = self.patterns.get(language, {})
            if not lang_patterns:
                self._warn_missing_language_patterns(language)
                return imports

            # Collect metadata across all imports
            metadata = {
                "import_style": "direct",  # Will be updated if we find relative/wildcard imports
                "is_optional": False,     # Will be set to True if we find dynamic imports
                "is_conditional": False,  # Will be set to True if we find conditional imports
                "line_numbers": [],
                "context": None,
            }

            for line_num, line in enumerate(content.splitlines(), 1):
                # Skip empty lines and comments
                stripped_line = line.strip()
                if not stripped_line or is_comment(stripped_line, language):
                    continue

                # Check all import patterns
                for pattern in lang_patterns.get("imports", []):
                    match = pattern.search(line)
                    if match:
                        import_path = self._parse_import(match, line, language)
                        if not import_path:
                            continue

                        # Analyze import characteristics
                        import_characteristics = self._analyze_import_characteristics(line, language, line_num) # TODO: Merge with _parse_import

                        # Update metadata based on this import
                        self._update_imports_metadata(metadata, import_characteristics)

                        metadata["line_numbers"].append(line_num)

                        # Resolve the import path
                        resolved_path = await self._resolve_import_path(
                            import_path, language, file_path
                        )

                        if not resolved_path:
                            logger.warning(
                                f"Could not resolve import {import_path} "
                                f"in {file_path}"
                            )
                            continue

                        # Determine import type
                        is_type = self._is_type_import(
                            line, lang_patterns.get("types", [])
                        )

                        # Add to appropriate sets
                        imports["direct"].add(resolved_path)
                        if is_type:
                            imports["types"].add(resolved_path)
                            self.metrics.import_counts.labels(language, "type").inc()
                        else:
                            imports["runtime"].add(resolved_path)
                            self.metrics.import_counts.labels(
                                language, "runtime"
                            ).inc()

                        # Check for cross-language imports
                        if self._is_cross_language_import(resolved_path, language):
                            imports["cross_language"].add(resolved_path)
                            self.metrics.import_counts.labels(
                                language, "cross_language"
                            ).inc()

            # Update the imports metadata
            imports["metadata"] = metadata

            return imports

        except Exception as e:
            logger.error(f"Error analyzing imports: {e}", exc_info=True)
            return self._get_fallback_result()

    def _update_imports_metadata(self, metadata: dict, import_characteristics: dict):
        """Update the imports metadata based on the import characteristics"""
        # Update metadata based on this import
        if import_characteristics["is_relative"] and metadata["import_style"] == "direct":
            metadata["import_style"] = "relative"
        elif import_characteristics["is_wildcard"]:
            metadata["import_style"] = "wildcard"
        elif import_characteristics["is_aliased"] and metadata["import_style"] not in ["relative", "wildcard"]:
            metadata["import_style"] = "aliased"

        if import_characteristics["is_optional"]:
            metadata["is_optional"] = True
        if import_characteristics["is_conditional"]:
            metadata["is_conditional"] = True

    async def _analyze_imports_uncached_parallel(
        self, content: str, language: str, file_path: str
    ) -> dict[str, set[str]]:
        """Analyze imports in file content with parallel resolution"""
        try:
            imports = {
                "direct": set(),
                "types": set(),
                "runtime": set(),
                "cross_language": set(),
            }

            lang_patterns = self.patterns.get(language, {})
            if not lang_patterns:
                self._warn_missing_language_patterns(language)
                return imports

            # Collect metadata across all imports
            metadata = {
                "import_style": "direct",
                "is_optional": False,
                "is_conditional": False,
                "line_numbers": [],
                "context": None,
            }

            # Collect all imports first
            import_tasks = []
            for line_num, line in enumerate(content.splitlines(), 1):
                stripped_line = line.strip()
                if not stripped_line or is_comment(stripped_line, language):
                    continue

                for pattern in lang_patterns.get("imports", []):
                    match = pattern.search(line)
                    if match:
                        import_path = self._parse_import(match, line, language)
                        if not import_path:
                            continue

                        # Analyze import characteristics
                        import_characteristics = self._analyze_import_characteristics(line, language, line_num) # TODO: Merge with _parse_import

                        # Update metadata based on this import
                        self._update_imports_metadata(metadata, import_characteristics)

                        metadata["line_numbers"].append(line_num)

                        import_tasks.append(
                            {
                                "path": import_path,
                                "line": line,
                                "line_num": line_num,
                                "is_type": self._is_type_import(
                                    line, lang_patterns.get("types", [])
                                ),
                            }
                        )

            # Process imports in batches
            batch_size = self.config.batch_size
            import_batches = [
                import_tasks[i : i + batch_size]
                for i in range(0, len(import_tasks), batch_size)
            ]

            resolved_imports: list[dict[str, Any]] = []
            async with asyncio.TaskGroup() as tg:
                batch_tasks = [
                    tg.create_task(
                        self._resolve_import_batch(batch, language, file_path)
                    )
                    for batch in import_batches
                ]

            for task in batch_tasks:
                resolved_imports.extend(task.result())

            # Process resolved imports
            for imp in resolved_imports:
                resolved_path = imp["resolved_path"]
                imports["direct"].add(resolved_path)

                if imp["is_type"]:
                    imports["types"].add(resolved_path)
                else:
                    imports["runtime"].add(resolved_path)

                if self._is_cross_language_import(resolved_path, language):
                    imports["cross_language"].add(resolved_path)

            # Update the imports metadata
            imports["metadata"] = metadata

            return imports

        except Exception as e:
            logger.error(f"Error analyzing imports: {e}", exc_info=True)
            return self._get_fallback_result()

    async def _resolve_import_batch(
        self, batch: list[dict], language: str, file_path: str
    ) -> list[dict]:
        """Resolve a batch of imports with metrics"""
        start_time = time.time()
        batch_size = len(batch)

        try:
            self.metrics.batch_size.labels(language=language).observe(batch_size)
            self.metrics.concurrent_batches.labels(language=language).inc()

            # Resolve imports in parallel
            resolved_imports = []
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._resolve_import_path(imp_info["path"], language, file_path)
                    )
                    for imp_info in batch
                ]

            for task, imp_info in zip(tasks, batch, strict=True):
                try:
                    resolved_path = task.result()
                    if not resolved_path:
                        logger.warning(
                            f"Could not resolve import {imp_info['path']} "
                            f"in {file_path}"
                        )
                        continue
                    resolved_imports.append(
                        {**imp_info, "resolved_path": resolved_path}
                    )
                except Exception as e:
                    logger.error(
                        f"Import resolution error at line "
                        f"{imp_info['line_num']}: {e}"
                    )
                    self.metrics.resolver_errors.labels(
                        language=language, error_type=type(e).__name__
                    ).inc()

            success_rate = len(resolved_imports) / batch_size
            self.metrics.batch_success_rate.labels(language=language).set(
                success_rate
            )

            return resolved_imports

        finally:
            duration = time.time() - start_time
            self.metrics.batch_duration.labels(language=language).observe(duration)
            self.metrics.concurrent_batches.labels(language=language).dec()

    def _analyze_import_characteristics(self, line: str, language: str, line_num: int) -> dict[str, bool]:
        """Analyze characteristics of an import statement"""
        # TODO: Merge with _parse_import
        characteristics = {
            "is_relative": False,
            "is_wildcard": False,
            "is_aliased": False,
            "is_optional": False,
            "is_conditional": False,
        }

        line_lower = line.lower().strip()

        if language == "python":
            # Relative imports
            characteristics["is_relative"] = "from ." in line or "from .." in line
            # Wildcard imports
            characteristics["is_wildcard"] = "import *" in line
            # Aliased imports
            characteristics["is_aliased"] = " as " in line
            # Optional/dynamic imports
            characteristics["is_optional"] = (
                "importlib.import_module" in line or
                "__import__" in line or
                "try:" in line_lower  # Heuristic for try/except imports
            )
            # Conditional imports
            characteristics["is_conditional"] = (
                line.strip().startswith("if ") or
                "import" in line and ("if " in line_lower or "else:" in line_lower)
            )

        elif language in ("javascript", "typescript"):
            # Relative imports
            characteristics["is_relative"] = (
                'from "./' in line or 'from "../' in line or
                'require("./' in line or 'require("../' in line
            )
            # Wildcard imports
            characteristics["is_wildcard"] = "import *" in line or "* as " in line
            # Aliased imports
            characteristics["is_aliased"] = " as " in line
            # Optional/dynamic imports
            characteristics["is_optional"] = (
                "import(" in line or  # Dynamic import()
                "require.resolve" in line or
                "await import" in line
            )
            # Conditional imports
            characteristics["is_conditional"] = (
                line.strip().startswith("if ") or
                "import" in line and ("if (" in line_lower or "? " in line)
            )

        elif language == "java":
            # Wildcard imports
            characteristics["is_wildcard"] = "import " in line and ".*;" in line
            # Static imports
            characteristics["is_aliased"] = "import static" in line  # Treat static as aliased

        elif language == "kotlin":
            # Wildcard imports
            characteristics["is_wildcard"] = "import " in line and ".*" in line
            # Aliased imports
            characteristics["is_aliased"] = " as " in line

        elif language == "rust":
            # Relative imports (crate-relative)
            characteristics["is_relative"] = (
                "use crate::" in line or
                "use super::" in line or
                "use self::" in line
            )
            # Wildcard imports
            characteristics["is_wildcard"] = "use " in line and "::*" in line
            # Aliased imports
            characteristics["is_aliased"] = " as " in line

        elif language == "go":
            # Aliased imports
            characteristics["is_aliased"] = (
                line.strip().startswith("import ") and
                " " in line.split('"')[0] and
                not line.strip().startswith("import (")
            )

        elif language == "swift":
            # Testable imports (treat as aliased)
            characteristics["is_aliased"] = "@testable import" in line

        return characteristics

    def _is_type_import(self, line: str, type_patterns: list[Pattern]) -> bool:
        """Check if import is type-only"""
        return any(pattern.search(line) for pattern in type_patterns)

    def _is_cross_language_import(self, import_path: str, source_language: str) -> bool:
        """Check if import is cross-language"""
        try:
            # Get file extension
            ext = Path(import_path).suffix.lower()

            # Get language extensions
            lang_exts = {
                lang: config.get("extensions", [])
                for lang, config in self.config.language_configs.items()
            }

            # Find target language
            target_lang = next(
                (
                    lang
                    for lang, exts in lang_exts.items()
                    if ext in exts and lang != source_language
                ),
                None,
            )

            return bool(target_lang)

        except Exception as e:
            logger.error(f"Cross-language check error: {e}")
            return False

    async def _analyze_transitive_imports(
        self, direct_imports: set[str], language: str, file_path: str, depth: int
    ) -> set[str]:
        """
        Analyze transitive imports recursively.

        Performance: O(d*m) where d is depth and m is imports per file
        Memory: O(t) where t is total transitive imports
        """
        if depth >= self.config.max_transitive_depth:
            return set()

        try:
            transitive = set()
            for import_path in direct_imports:
                # Resolve actual file path
                resolved_path = await self._resolve_import_path(import_path, language, file_path)
                if not resolved_path:
                    continue

                # Analyze imported file
                content = await self.file_content_cache.read_file(resolved_path)
                if content:
                    imports = await self._analyze_file_imports(
                        content, language, resolved_path
                    )
                    transitive.update(imports["direct"])

                    # Recurse for next level
                    next_level = await self._analyze_transitive_imports(
                        imports["direct"], language, resolved_path, depth + 1
                    )
                    transitive.update(next_level)

            return transitive

        except Exception as e:
            logger.error(f"Error analyzing transitive imports: {e}", exc_info=True)
            return set()

    def _parse_type_import(
        self, match: re.Match, line: str, language: str, file_path: str
    ) -> str | None:
        """Parse type import statement"""
        try:
            if language == "python":
                return self._parse_python_type_import(match, line)
            elif language in ("typescript", "javascript"):
                return self._parse_js_type_import(match, line)
            # TODO: Add more language-specific parsing
            return None
        except Exception as e:
            logger.error(f"Type import parsing error: {e}")
            return None

    def _parse_runtime_import(
        self, match: re.Match, line: str, language: str, file_path: str
    ) -> str | None:
        """Parse runtime import statement"""
        try:
            if language == "python":
                return self._parse_python_runtime_import(match, line)
            elif language in ("typescript", "javascript"):
                return self._parse_js_runtime_import(match, line)
            # TODO: Add more language-specific parsing
            return None
        except Exception as e:
            logger.error(f"Runtime import parsing error: {e}")
            return None

    def _parse_python_type_import(self, match: re.Match, line: str) -> str | None:
        """Parse Python type import statement"""
        try:
            import_path = match.group(1).strip()

            # Handle from ... import ... syntax
            if "from" in line:
                # For "from xx.yy import zz", we want to resolve "xx.yy"
                # The import resolution should point to the module file, not the specific symbol
                base_module = import_path
                ### imports = line.split("import")[1].strip()
                ### # Handle multiple imports
                ### for imp in imports.split(","):
                ###     if "as" in imp:
                ###         imp = imp.split("as")[0]
                ###     return f"{base_module}.{imp.strip()}"
                return base_module.strip()

            # Handle direct import
            if "as" in import_path:
                import_path = import_path.split("as")[0]

            return import_path.strip()

        except Exception as e:
            logger.error(f"Python type import parsing error: {e}")
            return None

    def _parse_js_type_import(self, match: re.Match, line: str) -> str | None:
        """Parse TypeScript/JavaScript type import statement"""
        try:
            # Extract path from quotes
            if "from" in line:
                path = line.split("from")[1].strip()
                path = path.strip("'").strip('"')
            else:
                path = match.group(1)

            # Normalize path
            if not path.endswith(".ts") and not path.endswith(".js"):
                path += ".ts"

            return path

        except Exception as e:
            logger.error(f"JS type import parsing error: {e}")
            return None

    def _parse_python_runtime_import(self, match: re.Match, line: str) -> str | None:
        """Parse Python runtime import statement"""
        try:
            import_path = match.group(1).strip()

            # Handle relative imports
            if import_path.startswith("."):
                # TODO: Resolve relative imports based on file location
                pass

            # Handle from ... import ... syntax
            if "from" in line:
                # For "from xx.yy import zz", we want to resolve "xx.yy"
                # The import resolution should point to the module file, not the specific symbol
                base_module = import_path
                ### imports = line.split("import")[1].strip()
                ### # Handle multiple imports
                ### for imp in imports.split(","):
                ###     if "as" in imp:
                ###         imp = imp.split("as")[0]
                ###     return f"{base_module}.{imp.strip()}"
                return base_module.strip()

            # Handle direct import
            if "as" in import_path:
                import_path = import_path.split("as")[0]

            return import_path.strip()

        except Exception as e:
            logger.error(f"Python runtime import parsing error: {e}")
            return None

    def _parse_js_runtime_import(self, match: re.Match, line: str) -> str | None:
        """Parse TypeScript/JavaScript runtime import statement"""
        try:
            # Handle require syntax
            if "require(" in line:
                path = match.group(1)
            # Handle import syntax
            else:
                path = line.split("from")[1].strip()
                path = path.strip("'").strip('"')

            # Normalize path
            if not path.endswith(".js"):
                path += ".js"

            return path

        except Exception as e:
            logger.error(f"JS runtime import parsing error: {e}")
            return None

    def _parse_import(self, match: re.Match, line: str, language: str) -> str | None:
        """Parse import based on language"""
        parsers = {
            "python": self._parse_python_import,
            "typescript": self._parse_js_import,
            "javascript": self._parse_js_import,
            "java": self._parse_java_import,
            "kotlin": self._parse_kotlin_import,
            "rust": self._parse_rust_import,
            "go": self._parse_go_import,
            "swift": self._parse_swift_import,
        }

        parser = parsers.get(language)
        if parser:
            try:
                return parser(match, line)
            except Exception as e:
                logger.error(f"Import parsing error for {language}: {e}")
                return None

        logger.warning(f"No parser found for language: {language}")
        return None

    def _parse_python_import(self, match: re.Match, line: str) -> str | None:
        """Parse Python import statement"""
        try:
            import_path = match.group(1).strip()

            # Handle from ... import ... syntax
            if "from" in line:
                # For "from xx.yy import zz", we want to resolve "xx.yy"
                # The import resolution should point to the module file, not the specific symbol
                base_module = import_path
                ### imports = line.split("import")[1].strip()
                ### # Handle multiple imports
                ### for imp in imports.split(","):
                ###     if "as" in imp:
                ###         imp = imp.split("as")[0]
                ###     return f"{base_module}.{imp.strip()}"
                return base_module.strip()

            # Handle direct import
            if "as" in import_path:
                import_path = import_path.split("as")[0]

            return import_path.strip()

        except Exception as e:
            logger.error(f"Python import parsing error: {e}")
            return None

    def _parse_js_import(self, match: re.Match, line: str) -> str | None:
        """Parse TypeScript/JavaScript import statement"""
        try:
            # Handle require syntax
            if "require(" in line:
                path = match.group(1)
            # Handle import syntax
            else:
                path = line.split("from")[1].strip()
                path = path.strip("'").strip('"')

            # Normalize path
            if not path.endswith(".js") and not path.endswith(".ts"):
                path += ".js"

            return path

        except Exception as e:
            logger.error(f"JS import parsing error: {e}")
            return None

    def _parse_java_import(self, match: re.Match, line: str) -> str | None:
        """Parse Java import statement"""
        try:
            is_static = bool(match.group(1))
            import_path = match.group(2)

            # Handle wildcard imports
            if import_path.endswith(".*"):
                import_path = import_path[:-2]

            # Convert package notation to path
            path = import_path.replace(".", "/")

            # Add extension if not static import
            if not is_static and not path.endswith(".java"):
                path += ".java"

            return path

        except Exception as e:
            logger.error(f"Java import parsing error: {e}")
            return None

    def _parse_kotlin_import(self, match: re.Match, line: str) -> str | None:
        """Parse Kotlin import statement"""
        try:
            import_path = match.group(1)

            # Handle wildcard imports
            if import_path.endswith(".*"):
                import_path = import_path[:-2]

            # Convert package notation to path
            path = import_path.replace(".", "/")

            # Add extension
            if not path.endswith(".kt"):
                path += ".kt"

            return path

        except Exception as e:
            logger.error(f"Kotlin import parsing error: {e}")
            return None

    def _parse_rust_import(self, match: re.Match, line: str) -> str | None:
        """Parse Rust import statement"""
        try:
            import_path = match.group(1)

            # Handle crate imports
            if line.startswith("extern crate"):
                return f"extern/{import_path}/lib.rs"

            # Handle mod declarations
            if line.startswith("mod"):
                return f"{import_path}.rs"

            # Handle use statements
            path = import_path.replace("::", "/")
            if not path.endswith(".rs"):
                path += ".rs"

            return path

        except Exception as e:
            logger.error(f"Rust import parsing error: {e}")
            return None

    def _parse_go_import(self, match: re.Match, line: str) -> str | None:
        """Parse Go import statement"""
        try:
            import_path = match.group(1)

            # Handle vendored dependencies
            if not import_path.startswith("."):
                return f"vendor/{import_path}"

            # Handle relative imports
            if not import_path.endswith(".go"):
                import_path += ".go"

            return import_path

        except Exception as e:
            logger.error(f"Go import parsing error: {e}")
            return None

    def _parse_swift_import(self, match: re.Match, line: str) -> str | None:
        """Parse Swift import statement"""
        try:
            import_path = match.group(1)

            # Handle testable imports
            if line.startswith("@testable"):
                return f"Tests/{import_path}.swift"

            # Handle framework imports
            if import_path[0].isupper():
                return f"Frameworks/{import_path}.framework"

            # Handle regular imports
            return f"{import_path}.swift"

        except Exception as e:
            logger.error(f"Swift import parsing error: {e}")
            return None

    def _resolve_python_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve Python import to file path"""
        try:
            # Handle relative imports (starting with dots)
            if import_path.startswith('.'):
                return self._resolve_relative_import(import_path, file_path)

            # Check if it's a standard library module first
            if self._is_standard_library_module(import_path):
                return f"<stdlib>/{import_path}"

            # Convert dots to path separators
            path_parts = import_path.split(".")

            # Get the directory containing the current file for relative resolution
            current_file_dir = Path(file_path).parent

            # Build comprehensive list of search paths
            search_paths = self._get_python_search_paths(current_file_dir)

            # Try to resolve the import in each search path
            for search_path in search_paths:
                resolved_path = self._try_resolve_in_path(search_path, path_parts)
                if resolved_path:
                    return str(resolved_path)

            # If not found in project paths, check if it's an external package
            external_path = self._resolve_external_package(import_path)
            if external_path:
                return external_path

            return None

        except Exception as e:
            logger.error(f"Python import resolution error for '{import_path}': {e}")
            return None

    def _is_standard_library_module(self, module_name: str) -> bool:
        """Check if a module is part of Python's standard library"""
        # Get the top-level module name
        top_level = module_name.split('.')[0]

        # Common standard library modules
        stdlib_modules = {
            # Built-ins and core modules
            'sys', 'os', 'io', 'time', 'datetime', 'math', 'random', 'json', 'csv',
            'urllib', 'http', 'html', 'xml', 'email', 'base64', 'hashlib', 'hmac',
            'uuid', 'secrets', 'struct', 'codecs', 'locale', 'calendar',

            # Data structures and algorithms
            'collections', 'itertools', 'functools', 'operator', 'copy', 'pickle',
            'shelve', 'dbm', 'sqlite3', 'heapq', 'bisect', 'array', 'weakref',

            # Text processing
            're', 'string', 'textwrap', 'unicodedata', 'stringprep',

            # File and path handling
            'pathlib', 'fileinput', 'stat', 'filecmp', 'tempfile', 'glob', 'fnmatch',
            'linecache', 'shutil',

            # Data compression and archiving
            'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',

            # Persistence
            'pickle', 'copyreg', 'shelve', 'marshal', 'dbm', 'sqlite3',

            # Numeric and mathematical
            'numbers', 'math', 'cmath', 'decimal', 'fractions', 'random', 'statistics',

            # Functional programming
            'itertools', 'functools', 'operator',

            # File formats
            'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib',

            # Cryptographic services
            'hashlib', 'hmac', 'secrets',

            # Generic OS services
            'os', 'io', 'time', 'argparse', 'getopt', 'logging', 'getpass', 'curses',
            'platform', 'errno', 'ctypes',

            # Concurrent execution
            'threading', 'multiprocessing', 'concurrent', 'subprocess', 'sched', 'queue',
            'asyncio', '_thread',

            # Networking and IPC
            'socket', 'ssl', 'select', 'selectors', 'asyncio', 'signal',

            # Internet protocols and support
            'webbrowser', 'urllib', 'http', 'ftplib', 'poplib', 'imaplib', 'nntplib',
            'smtplib', 'smtpd', 'telnetlib', 'uuid', 'socketserver', 'xmlrpc',

            # Multimedia services
            'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr', 'sndhdr',
            'ossaudiodev',

            # Internationalization
            'gettext', 'locale',

            # Program frameworks
            'turtle', 'cmd', 'shlex',

            # GUI
            'tkinter', 'turtle',

            # Development tools
            'typing', 'pydoc', 'doctest', 'unittest', 'test', 'lib2to3',

            # Debugging and profiling
            'bdb', 'faulthandler', 'pdb', 'timeit', 'trace', 'tracemalloc',

            # Software packaging and distribution
            'distutils', 'ensurepip', 'venv', 'zipapp',

            # Runtime services
            'sys', 'sysconfig', 'builtins', '__main__', 'warnings', 'dataclasses',
            'contextlib', 'abc', 'atexit', 'traceback', 'gc', 'inspect', 'site',

            # Custom Python interpreters
            'code', 'codeop',

            # Importing modules
            'zipimport', 'pkgutil', 'modulefinder', 'runpy', 'importlib',

            # Language services
            'parser', 'ast', 'symtable', 'symbol', 'token', 'keyword', 'tokenize',
            'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools',
            'formatter',

            # MS Windows specific
            'msilib', 'msvcrt', 'winreg', 'winsound',

            # Unix specific
            'posix', 'pwd', 'spwd', 'grp', 'crypt', 'termios', 'tty', 'pty', 'fcntl',
            'pipes', 'resource', 'nis', 'syslog',

            # Superseded modules (but still in stdlib)
            'optparse', 'imp',
        }

        return top_level in stdlib_modules

    def _resolve_external_package(self, import_path: str) -> str | None:
        """Resolve external package imports (numpy, torch, etc.)"""
        try:
            # Get the top-level package name
            top_level = import_path.split('.')[0]

            # Don't resolve stdlib modules as external packages
            if self._is_standard_library_module(top_level):
                return None

            # Try to find the package in site-packages or virtual environment
            import importlib.util

            try:
                spec = importlib.util.find_spec(top_level)
                if spec and spec.origin:
                    # Return a marker for external packages with the actual path
                    return f"<external>/{import_path}@{spec.origin}"
                elif spec and spec.submodule_search_locations:
                    # Package with __init__.py
                    for location in spec.submodule_search_locations:
                        init_file = Path(location) / "__init__.py"
                        if init_file.exists():
                            return f"<external>/{import_path}@{init_file}"
            except (ImportError, ModuleNotFoundError, ValueError):
                pass

            # Fallback: check common package locations
            search_paths = []

            # Virtual environment site-packages
            if 'VIRTUAL_ENV' in os.environ:
                venv_path = Path(os.environ['VIRTUAL_ENV'])
                for python_ver in ['python3.11', 'python3.10', 'python3.9', 'python3.8']:
                    site_packages = venv_path / 'lib' / python_ver / 'site-packages'
                    if site_packages.exists():
                        search_paths.append(site_packages)
                        break

            # System site-packages (filtered)
            for path_str in sys.path:
                if 'site-packages' in path_str or 'dist-packages' in path_str:
                    path = Path(path_str)
                    if path.exists():
                        search_paths.append(path)

            # Try to find the package in these locations
            for search_path in search_paths:
                package_path = search_path / top_level
                if package_path.exists():
                    if package_path.is_dir():
                        init_file = package_path / "__init__.py"
                        if init_file.exists():
                            return f"<external>/{import_path}@{init_file}"
                        # Namespace package
                        return f"<external>/{import_path}@{package_path}"
                    else:
                        # Single file module
                        return f"<external>/{import_path}@{package_path}"

            return None

        except Exception as e:
            logger.debug(f"External package resolution error for '{import_path}': {e}")
            return None

    def _resolve_relative_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve Python relative imports (those starting with dots)"""
        try:
            current_file = Path(file_path)
            current_dir = current_file.parent

            # Count leading dots to determine how many levels to go up
            level = 0
            for char in import_path:
                if char == '.':
                    level += 1
                else:
                    break

            # Extract the module part after the dots
            module_part = import_path[level:] if level < len(import_path) else ""

            # Go up the specified number of levels
            target_dir = current_dir
            for _ in range(level - 1):  # level-1 because one dot means current package
                target_dir = target_dir.parent
                if target_dir == target_dir.parent:  # Hit filesystem root
                    break

            # If there's a module part, resolve it from the target directory
            if module_part:
                module_parts = module_part.split('.')
                resolved_path = self._try_resolve_in_path(target_dir, module_parts)
                if resolved_path:
                    return str(resolved_path)
            else:
                # Just dots (e.g., "from . import something") - return the package directory
                init_file = target_dir / '__init__.py'
                if init_file.exists():
                    return str(init_file)
                # Check for namespace package
                elif any(child.suffix == '.py' for child in target_dir.iterdir() if child.is_file()):
                    return str(target_dir)

            return None

        except Exception as e:
            logger.error(f"Relative import resolution error for '{import_path}': {e}")
            return None

    def _get_python_search_paths(self, current_file_dir: Path) -> list[Path]:
        """Get comprehensive list of Python search paths"""
        search_paths = []

        # 1. Current file directory (for relative imports)
        search_paths.append(current_file_dir)

        # 2. Walk up the directory tree to find package roots
        # (directories containing __init__.py or pyproject.toml or setup.py)
        current_dir = current_file_dir
        while current_dir != current_dir.parent:  # Stop at filesystem root
            # Check if this is a package root
            if any((current_dir / marker).exists() for marker in ['__init__.py', 'pyproject.toml', 'setup.py', 'setup.cfg']):
                search_paths.append(current_dir)
                # Also add parent directory (common pattern for src layouts)
                if current_dir.parent not in search_paths:
                    search_paths.append(current_dir.parent)
            current_dir = current_dir.parent

        # 3. Look for common Python project structures
        project_root = self._find_project_root(current_file_dir)
        if project_root:
            # Add the project root itself (critical for minGPT structure)
            if project_root not in search_paths:
                search_paths.append(project_root)

            # Standard layouts
            for subdir in ['src', 'lib', 'python']:
                potential_path = project_root / subdir
                if potential_path.exists() and potential_path not in search_paths:
                    search_paths.append(potential_path)

        # 4. For Git repositories, add the repository root
        git_root = self._find_git_root(current_file_dir)
        if git_root and git_root not in search_paths:
            search_paths.append(git_root)

        # 5. Environment PYTHONPATH (more reliable than sys.path for static analysis)
        pythonpath = os.environ.get('PYTHONPATH', '')
        if pythonpath:
            for path_str in pythonpath.split(os.pathsep):
                path = Path(path_str).resolve()
                if path.exists() and path not in search_paths:
                    search_paths.append(path)

        # 6. sys.path as fallback (but filter out system paths that won't contain project code)
        for path_str in sys.path:
            if not path_str:  # Empty string means current directory
                continue
            path = Path(path_str).resolve()
            # Skip system/site-packages paths for project-specific resolution
            if any(skip in str(path) for skip in ['site-packages', 'dist-packages', '/usr/lib', '/Library']):
                continue
            if path.exists() and path not in search_paths:
                search_paths.append(path)

        # 7. Virtual environment detection
        venv_paths = self._detect_virtual_env_paths()
        search_paths.extend(venv_paths)

        return search_paths

    def _find_git_root(self, start_path: Path) -> Path | None:
        """Find the Git repository root"""
        current = start_path
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        return None

    def _find_project_root(self, start_path: Path) -> Path | None:
        """Find the project root by looking for common project markers"""
        current = start_path

        # First, try to find explicit project markers
        while current != current.parent:
            # Check for project root indicators
            if any((current / marker).exists() for marker in [
                'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
                'Pipfile', 'poetry.lock', 'conda.yaml', 'environment.yml'
            ]):
                return current
            current = current.parent

        # If no explicit markers found, try Git root
        git_root = self._find_git_root(start_path)
        if git_root:
            return git_root

        # Fallback: look for directories with Python packages
        current = start_path
        while current != current.parent:
            # Check if this directory contains Python packages
            python_dirs = [d for d in current.iterdir()
                          if d.is_dir() and (d / '__init__.py').exists()]
            if len(python_dirs) >= 1:  # At least one Python package
                return current
            current = current.parent

        return None

    def _detect_virtual_env_paths(self) -> list[Path]:
        """Detect virtual environment paths"""
        venv_paths = []

        # Check common virtual env indicators
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            venv_path = Path(virtual_env)
            # Add site-packages from the virtual env
            for python_ver in ['python3.11', 'python3.10', 'python3.9', 'python3.8']:
                site_packages = venv_path / 'lib' / python_ver / 'site-packages'
                if site_packages.exists():
                    venv_paths.append(site_packages)
                    break

        # Check for conda environment
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_path = Path(conda_prefix) / 'lib' / 'python3.11' / 'site-packages'  # Adjust version as needed
            if conda_path.exists():
                venv_paths.append(conda_path)

        return venv_paths

    def _try_resolve_in_path(self, search_path: Path, path_parts: list[str]) -> Path | None:
        """Try to resolve import path parts in a given search directory"""
        try:
            # Build the potential module path
            module_path = search_path
            for part in path_parts:
                module_path = module_path / part

            # Try different file extensions and patterns
            candidates = [
                # Direct file matches
                module_path.with_suffix('.py'),
                module_path.with_suffix('.pyi'),  # Type stub files
                module_path.with_suffix('.pyx'),  # Cython files
                module_path.with_suffix('.so'),   # Compiled extensions
                module_path.with_suffix('.pyd'),  # Windows extensions

                # Package directory with __init__.py
                module_path / '__init__.py',
                module_path / '__init__.pyi',

                # Handle namespace packages (PEP 420)
                module_path,  # Directory without __init__.py
            ]

            for candidate in candidates:
                if candidate.exists():
                    # For directories, ensure it's a valid Python package
                    if candidate.is_dir():
                        # Check if it's a namespace package or has __init__.py
                        if any((candidate / init).exists() for init in ['__init__.py', '__init__.pyi']):
                            return candidate / '__init__.py'
                        # Namespace package - return the directory itself
                        elif any(child.suffix in ['.py', '.pyi'] for child in candidate.iterdir() if child.is_file()):
                            return candidate
                    else:
                        return candidate

            # Special case: try partial path resolution for deeply nested imports
            # e.g., for "mingpt.utils", try finding "mingpt" directory first
            if len(path_parts) > 1:
                partial_path = search_path / path_parts[0]
                if partial_path.exists() and partial_path.is_dir():
                    # Recursively try to resolve the rest
                    remaining_parts = path_parts[1:]
                    return self._try_resolve_in_path(partial_path, remaining_parts)

            return None

        except Exception as e:
            logger.debug(f"Error resolving path in {search_path}: {e}")
            return None

    def _resolve_js_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve JavaScript/TypeScript import to file path"""
        try:
            path = Path(import_path)

            # Handle node_modules
            if not import_path.startswith("."):
                path = Path("node_modules") / path

            # Try extensions
            for ext in [".js", ".ts", ".jsx", ".tsx"]:
                full_path = path.with_suffix(ext)
                if full_path.exists():
                    return str(full_path)

                # Check for index files
                index_path = path / f"index{ext}"
                if index_path.exists():
                    return str(index_path)

            return None

        except Exception as e:
            logger.error(f"JS import resolution error: {e}")
            return None

    def _resolve_java_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve Java import path"""
        try:
            # Check source directories
            source_dirs = ["src/main/java", "src/test/java"]

            for src_dir in source_dirs:
                full_path = Path(src_dir) / import_path
                if full_path.exists():
                    return str(full_path)

            return None

        except Exception as e:
            logger.error(f"Java import resolution error: {e}")
            return None

    def _resolve_kotlin_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve Kotlin import path"""
        try:
            # Check common Kotlin source directories
            source_dirs = [
                "src/main/kotlin",
                "src/test/kotlin",
                "app/src/main/kotlin",
                "app/src/test/kotlin",
            ]

            for src_dir in source_dirs:
                full_path = Path(src_dir) / import_path
                if full_path.exists():
                    return str(full_path)

            return None

        except Exception as e:
            logger.error(f"Kotlin import resolution error: {e}")
            return None

    def _resolve_rust_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve Rust import path"""
        try:
            # Check common Rust source locations
            if import_path.startswith("crate::"):
                # Handle crate-relative paths
                path = import_path.replace("crate::", "src/")
            elif import_path.startswith("super::"):
                # Handle parent module paths
                path = import_path.replace("super::", "../")
            else:
                path = import_path

            # Try with .rs extension
            rs_path = Path(path).with_suffix(".rs")
            if rs_path.exists():
                return str(rs_path)

            # Try mod.rs for directories
            mod_path = Path(path) / "mod.rs"
            if mod_path.exists():
                return str(mod_path)

            return None

        except Exception as e:
            logger.error(f"Rust import resolution error: {e}")
            return None

    def _resolve_go_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve Go import path"""
        try:
            # Check GOPATH
            gopath = os.environ.get("GOPATH", "")
            if gopath:
                full_path = Path(gopath) / "src" / import_path
                if full_path.exists():
                    return str(full_path)

            # Check vendor directory
            vendor_path = Path("vendor") / import_path
            if vendor_path.exists():
                return str(vendor_path)

            # Check local module
            local_path = Path(import_path)
            if local_path.exists():
                return str(local_path)

            return None

        except Exception as e:
            logger.error(f"Go import resolution error: {e}")
            return None

    def _resolve_swift_import(self, import_path: str, file_path: str) -> str | None:
        """Resolve Swift import path"""
        try:
            # Check common Swift source locations
            source_dirs = ["Sources", "Tests", "Package.swift"]

            # Try direct module import
            for src_dir in source_dirs:
                module_path = Path(src_dir) / import_path
                if module_path.exists():
                    return str(module_path)

            # Check for framework
            framework_path = Path("Frameworks") / f"{import_path}.framework"
            if framework_path.exists():
                return str(framework_path)

            return None

        except Exception as e:
            logger.error(f"Swift import resolution error: {e}")
            return None

    def _get_fallback_result(self) -> dict[str, set[str]]:
        """Return safe fallback result on error"""
        return {
            "direct": set(),
            "types": set(),
            "runtime": set(),
            "cross_language": set(),
            "transitive": set(),
            "metadata": {
                "import_style": "direct",
                "is_optional": False,
                "is_conditional": False,
                "line_numbers": [],
                "context": None,
            },
        }


    async def _resolve_import_path(
        self, import_path: str, language: str, file_path: str
    ) -> str | None:
        """Resolve import path with caching"""
        try:
            start_time = time.time()

            # Generate cache key
            cache_key = self._make_cache_key(
                "resolved",
                language,
                import_path,
                file_path
            )

            # Try cache first
            cached_path = await self._get_cached_result(cache_key)
            if cached_path:
                return cached_path

            # Get resolver
            resolver = self.resolvers.get(language)
            if not resolver:
                logger.warning(f"No resolver found for language: {language}")
                self.metrics.resolver_calls.labels(
                    language=language, status="missing"
                ).inc()
                return None

            # Resolve path
            try:
                resolved_path = resolver(import_path, file_path)

                if resolved_path:
                    # Cache result
                    await self._cache_breaker(self.results_cache.set)(cache_key, resolved_path)
                    self.metrics.resolver_calls.labels(
                        language=language, status="success"
                    ).inc()
                else:
                    self.metrics.resolver_calls.labels(
                        language=language, status="not_found"
                    ).inc()

                return resolved_path

            except Exception as e:
                logger.error(f"Resolver error for {language}: {e}")
                self.metrics.resolver_errors.labels(
                    language=language, error_type=type(e).__name__
                ).inc()
                return None

        finally:
            duration = time.time() - start_time
            self.metrics.resolver_latency.labels(language=language).observe(duration)


