"""
Centralized language-extension mapping system for the polymathera codebase.

This module provides a unified way to map file extensions to programming languages
and vice versa, reducing duplication and ensuring consistency across all components.
"""

from dataclasses import dataclass
from enum import Enum


class FileCategory(Enum):
    """Categories of files for different processing strategies"""
    CODE = "code"
    MARKUP = "markup"
    CONFIG = "config"
    DATA = "data"
    BUILD = "build"
    DOCUMENTATION = "documentation"
    BINARY = "binary"
    NOTEBOOK = "notebook"
    UNKNOWN = "unknown"


@dataclass
class LanguageInfo:
    """Information about a programming language"""
    name: str
    extensions: set[str]
    category: FileCategory
    aliases: set[str] = None
    import_patterns: list[str] = None
    type_patterns: list[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = set()
        if self.import_patterns is None:
            self.import_patterns = []
        if self.type_patterns is None:
            self.type_patterns = []


class LanguageExtensionRegistry:
    """Centralized registry for language-extension mappings"""

    def __init__(self):
        self._languages = {}
        self._extension_to_language = {}
        self._filename_to_language = {}
        self._setup_default_languages()

    def _setup_default_languages(self):
        """Setup default language mappings"""

        # Programming languages
        languages = [
            LanguageInfo(
                name="python",
                extensions={".py", ".pyi", ".pyx", ".pyw"},
                category=FileCategory.CODE,
                aliases={"py", "python3"},
                import_patterns=[
                    r"^import\s+(.+)$",
                    r"^from\s+([.\w]+)\s+import",
                    r"^typing\s+import\s+(.+)$",
                ],
                type_patterns=[r"^from\s+typing\s+import", r"^from\s+types\s+import"]
            ),
            LanguageInfo(
                name="javascript",
                extensions={".js", ".jsx", ".mjs", ".cjs"},
                category=FileCategory.CODE,
                aliases={"js", "node"},
                import_patterns=[
                    r'import\s+.*from\s+[\'"](.+)[\'"]',
                    r'require\([\'"](.+)[\'"]\)',
                ],
                type_patterns=[r"import\s+type\s+.*from", r"import\s+\{.*\}\s+from"]
            ),
            LanguageInfo(
                name="typescript",
                extensions={".ts", ".tsx", ".d.ts"},
                category=FileCategory.CODE,
                aliases={"ts"},
                import_patterns=[
                    r'import\s+.*from\s+[\'"](.+)[\'"]',
                    r'require\([\'"](.+)[\'"]\)',
                ],
                type_patterns=[r"import\s+type\s+.*from", r"import\s+\{.*\}\s+from"]
            ),
            LanguageInfo(
                name="java",
                extensions={".java"},
                category=FileCategory.CODE,
                import_patterns=[
                    r"^import\s+(static\s+)?([.\w]+)(?:\s*\*)?;",
                    r"^package\s+([.\w]+);",
                ],
                type_patterns=[r"import\s+([.\w]+)\.(?:[A-Z]\w+);"]
            ),
            LanguageInfo(
                name="kotlin",
                extensions={".kt", ".kts"},
                category=FileCategory.CODE,
                import_patterns=[
                    r"^import\s+([.\w]+)(?:\s*\*)?",
                    r"^package\s+([.\w]+)",
                ],
                type_patterns=[r"import\s+([.\w]+)\.(?:[A-Z]\w+)"]
            ),
            LanguageInfo(
                name="scala",
                extensions={".scala", ".sc"},
                category=FileCategory.CODE,
                import_patterns=[
                    r"^import\s+([.\w]+)",
                    r"^package\s+([.\w]+)",
                ]
            ),
            LanguageInfo(
                name="rust",
                extensions={".rs"},
                category=FileCategory.CODE,
                import_patterns=[
                    r"^use\s+([:\w]+)(?:::\{[^}]+\})?;",
                    r"^mod\s+(\w+);",
                    r"^extern\s+crate\s+(\w+);",
                ],
                type_patterns=[r"use\s+([:\w]+)::(?:[A-Z]\w+)"]
            ),
            LanguageInfo(
                name="go",
                extensions={".go"},
                category=FileCategory.CODE,
                import_patterns=[
                    r"^import\s+[\"(]([^\"]+)[\")]",
                    r'^import\s+\(\s*(?:[^)]+\s+)?["\']([^"\']+)["\']',
                ],
                type_patterns=[]  # Go doesn't have separate type imports
            ),
            LanguageInfo(
                name="swift",
                extensions={".swift"},
                category=FileCategory.CODE,
                import_patterns=[r"^import\s+(\w+)", r"^@testable\s+import\s+(\w+)"],
                type_patterns=[r"import\s+class\s+(\w+)", r"import\s+protocol\s+(\w+)"]
            ),
            LanguageInfo(
                name="c",
                extensions={".c", ".h"},
                category=FileCategory.CODE,
                aliases={"c99", "c11"}
            ),
            LanguageInfo(
                name="cpp",
                extensions={".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++"},
                category=FileCategory.CODE,
                aliases={"c++", "cxx"}
            ),
            LanguageInfo(
                name="csharp",
                extensions={".cs"},
                category=FileCategory.CODE,
                aliases={"c#", "cs"}
            ),
            LanguageInfo(
                name="ruby",
                extensions={".rb", ".rake", ".gemspec"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="php",
                extensions={".php", ".phtml", ".php3", ".php4", ".php5", ".php7"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="r",
                extensions={".r", ".R"},
                category=FileCategory.CODE,
                aliases={"rlang"}
            ),
            LanguageInfo(
                name="matlab",
                extensions={".m", ".mat"},
                category=FileCategory.CODE,
                aliases={"octave"}
            ),
            LanguageInfo(
                name="shell",
                extensions={".sh", ".bash", ".zsh", ".fish", ".ksh"},
                category=FileCategory.CODE,
                aliases={"bash", "zsh", "fish"}
            ),
            LanguageInfo(
                name="powershell",
                extensions={".ps1", ".psm1", ".psd1"},
                category=FileCategory.CODE,
                aliases={"pwsh"}
            ),
            LanguageInfo(
                name="sql",
                extensions={".sql", ".ddl", ".dml"},
                category=FileCategory.CODE,
                aliases={"mysql", "postgresql", "sqlite"}
            ),
            LanguageInfo(
                name="julia",
                extensions={".jl"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="haskell",
                extensions={".hs", ".lhs"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="erlang",
                extensions={".erl", ".hrl"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="elixir",
                extensions={".ex", ".exs"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="clojure",
                extensions={".clj", ".cljs", ".cljc"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="lisp",
                extensions={".lisp", ".cl", ".el"},
                category=FileCategory.CODE,
                aliases={"common-lisp", "elisp"}
            ),
            LanguageInfo(
                name="fortran",
                extensions={".f", ".f90", ".f95", ".f03", ".f08"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="assembly",
                extensions={".asm", ".s"},
                category=FileCategory.CODE,
                aliases={"asm"}
            ),
            LanguageInfo(
                name="dart",
                extensions={".dart"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="solidity",
                extensions={".sol"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="perl",
                extensions={".pl", ".pm", ".t"},
                category=FileCategory.CODE
            ),
            LanguageInfo(
                name="lua",
                extensions={".lua"},
                category=FileCategory.CODE
            ),

            # Web and markup languages
            LanguageInfo(
                name="html",
                extensions={".html", ".htm", ".xhtml"},
                category=FileCategory.MARKUP
            ),
            LanguageInfo(
                name="css",
                extensions={".css"},
                category=FileCategory.MARKUP
            ),
            LanguageInfo(
                name="scss",
                extensions={".scss"},
                category=FileCategory.MARKUP
            ),
            LanguageInfo(
                name="sass",
                extensions={".sass"},
                category=FileCategory.MARKUP
            ),
            LanguageInfo(
                name="less",
                extensions={".less"},
                category=FileCategory.MARKUP
            ),
            LanguageInfo(
                name="xml",
                extensions={".xml", ".xsd", ".xsl", ".xslt"},
                category=FileCategory.MARKUP
            ),
            LanguageInfo(
                name="markdown",
                extensions={".md", ".markdown", ".mdown", ".mkd"},
                category=FileCategory.MARKUP,
                aliases={"md"}
            ),
            LanguageInfo(
                name="rst",
                extensions={".rst", ".rest"},
                category=FileCategory.MARKUP,
                aliases={"restructuredtext"}
            ),
            LanguageInfo(
                name="latex",
                extensions={".tex", ".latex"},
                category=FileCategory.MARKUP
            ),

            # Configuration and data formats
            LanguageInfo(
                name="yaml",
                extensions={".yaml", ".yml"},
                category=FileCategory.CONFIG,
                aliases={"yml"}
            ),
            LanguageInfo(
                name="json",
                extensions={".json", ".jsonc", ".json5"},
                category=FileCategory.CONFIG
            ),
            LanguageInfo(
                name="toml",
                extensions={".toml"},
                category=FileCategory.CONFIG
            ),
            LanguageInfo(
                name="ini",
                extensions={".ini", ".cfg", ".conf"},
                category=FileCategory.CONFIG
            ),
            LanguageInfo(
                name="properties",
                extensions={".properties"},
                category=FileCategory.CONFIG
            ),

            # Build and project files
            LanguageInfo(
                name="dockerfile",
                extensions={".dockerfile"},
                category=FileCategory.BUILD
            ),
            LanguageInfo(
                name="makefile",
                extensions={".makefile"},
                category=FileCategory.BUILD
            ),
            LanguageInfo(
                name="cmake",
                extensions={".cmake"},
                category=FileCategory.BUILD
            ),
            LanguageInfo(
                name="gradle",
                extensions={".gradle"},
                category=FileCategory.BUILD
            ),
            LanguageInfo(
                name="maven",
                extensions={".maven"},
                category=FileCategory.BUILD
            ),

            # Data formats
            LanguageInfo(
                name="csv",
                extensions={".csv"},
                category=FileCategory.DATA
            ),

            # Notebooks
            LanguageInfo(
                name="jupyter",
                extensions={".ipynb"},
                category=FileCategory.NOTEBOOK,
                aliases={"notebook"}
            ),

            # Documentation
            LanguageInfo(
                name="proto",
                extensions={".proto"},
                category=FileCategory.DOCUMENTATION
            ),
            LanguageInfo(
                name="avro",
                extensions={".avsc"},
                category=FileCategory.DOCUMENTATION
            ),
            LanguageInfo(
                name="thrift",
                extensions={".thrift"},
                category=FileCategory.DOCUMENTATION
            ),
            LanguageInfo(
                name="swagger",
                extensions={".swagger"},
                category=FileCategory.DOCUMENTATION
            ),
            LanguageInfo(
                name="raml",
                extensions={".raml"},
                category=FileCategory.DOCUMENTATION
            ),
        ]

        # Register all languages
        for lang in languages:
            self.register_language(lang)

        # Register special filenames
        special_files = {
            "dockerfile": "dockerfile",
            "makefile": "makefile",
            "rakefile": "ruby",
            "gemfile": "ruby",
            "podfile": "ruby",
            "cmakelists.txt": "cmake",
            "package.json": "json",
            "pyproject.toml": "toml",
            "cargo.toml": "toml",
            "requirements.txt": "text",
            "readme.md": "markdown",
            "license": "text",
            "gitignore": "text",
        }

        for filename, language in special_files.items():
            self._filename_to_language[filename.lower()] = language

    def register_language(self, language_info: LanguageInfo):
        """Register a language with its extension mappings"""
        self._languages[language_info.name] = language_info

        # Add aliases
        for alias in language_info.aliases:
            self._languages[alias] = language_info

        # Map extensions to language
        for ext in language_info.extensions:
            self._extension_to_language[ext.lower()] = language_info.name

    def get_language_by_extension(self, extension: str) -> str | None:
        """Get language name by file extension"""
        return self._extension_to_language.get(extension.lower())

    def get_language_by_filename(self, filename: str) -> str | None:
        """Get language name by filename (for special files)"""
        return self._filename_to_language.get(filename.lower())

    def get_language_info(self, language: str) -> LanguageInfo | None:
        """Get language information by name or alias"""
        return self._languages.get(language.lower())

    def get_extensions_for_language(self, language: str) -> set[str]:
        """Get all extensions for a language"""
        lang_info = self.get_language_info(language)
        return lang_info.extensions if lang_info else set()

    def get_languages_by_category(self, category: FileCategory) -> list[str]:
        """Get all languages in a category"""
        return [
            name for name, info in self._languages.items()
            if info.category == category and name == info.name  # Only primary names
        ]

    def get_extensions_by_category(self, category: FileCategory) -> set[str]:
        """Get all extensions for files in a category"""
        extensions = set()
        for name, info in self._languages.items():
            if info.category == category and name == info.name:  # Only primary names
                extensions.update(info.extensions)
        return extensions

    def detect_language(self, file_path: str) -> str | None:
        """Detect language from file path"""
        from pathlib import Path

        path = Path(file_path)

        # Check by filename first (for special files)
        filename_lang = self.get_language_by_filename(path.name)
        if filename_lang:
            return filename_lang

        # Check by extension
        extension = path.suffix.lower()
        return self.get_language_by_extension(extension)

    def is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file"""
        language = self.detect_language(file_path)
        if not language:
            return False

        lang_info = self.get_language_info(language)
        return lang_info and lang_info.category == FileCategory.CODE

    def is_binary_extension(self, extension: str) -> bool:
        """Check if extension typically represents binary files"""
        binary_extensions = {
            ".pyc", ".pyo", ".so", ".dll", ".dylib", ".class", ".exe", ".bin",
            ".pkl", ".pyd", ".o", ".a", ".lib", ".jar", ".war", ".ear",
            ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico",
            ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".wav",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"
        }
        return extension.lower() in binary_extensions

    def is_text_file_for_llm(self, file_path: str) -> bool:
        """Check if a file should be processed as text for LLM token counting."""
        file_path_lower = file_path.lower()

        # Skip .git directory and its contents
        if '/.git/' in file_path or file_path.startswith('.git/'):
            return False

        # Get file extension
        if '.' in file_path:
            extension = '.' + file_path.split('.')[-1]
        else:
            extension = ''

        # Skip binary files
        if self.is_binary_extension(extension):
            return False

        # Skip system files
        filename = file_path.split('/')[-1].lower()
        if filename in {'.ds_store', '.thumbs.db', 'desktop.ini'}:
            return False

        # Allow common config and documentation files
        if filename in {'.gitignore', '.env', '.envrc', '.editorconfig', 'readme', 'license', 'changelog'}:
            return True

        # Skip other hidden files except those we specifically allow
        if filename.startswith('.'):
            return False

        # Check if it's a recognized text-based category
        language_info = self._extension_to_language.get(extension.lower())
        if language_info is not None:
            return language_info.category in {
                FileCategory.CODE,
                FileCategory.MARKUP,
                FileCategory.CONFIG,
                FileCategory.DATA,
                FileCategory.BUILD,
                FileCategory.DOCUMENTATION,
                FileCategory.NOTEBOOK
            }

        # Default to true for unknown extensions (could be text files)
        return True

    def get_import_patterns_for_language(self, language: str) -> list[str]:
        """Get import patterns for a language"""
        lang_info = self.get_language_info(language)
        return lang_info.import_patterns if lang_info else []

    def get_type_patterns_for_language(self, language: str) -> list[str]:
        """Get type import patterns for a language"""
        lang_info = self.get_language_info(language)
        return lang_info.type_patterns if lang_info else []


# Global registry instance
_registry = LanguageExtensionRegistry()


# Convenience functions for backward compatibility
def get_language_by_extension(extension: str) -> str | None:
    """Get language name by file extension"""
    return _registry.get_language_by_extension(extension)


def detect_language(file_path: str) -> str | None:
    """Detect language from file path"""
    return _registry.detect_language(file_path)


def is_code_file(file_path: str) -> bool:
    """Check if file is a code file"""
    return _registry.is_code_file(file_path)


def get_extensions_for_language(language: str) -> set[str]:
    """Get all extensions for a language"""
    return _registry.get_extensions_for_language(language)


def get_extensions_by_category(category: FileCategory) -> set[str]:
    """Get all extensions for files in a category"""
    return _registry.get_extensions_by_category(category)


def is_binary_extension(extension: str) -> bool:
    """Check if extension typically represents binary files"""
    return _registry.is_binary_extension(extension)


def is_text_file_for_llm(file_path: str) -> bool:
    """Check if a file should be processed as text for LLM token counting"""
    return _registry.is_text_file_for_llm(file_path)


def get_registry() -> LanguageExtensionRegistry:
    """Get the global language registry"""
    return _registry