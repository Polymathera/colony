"""
This implementation provides:
1. Comprehensive File Type Support:
    - Programming languages (20+ languages)
    - Notebooks (Jupyter)
    - Markup (Markdown, RST)
    - Config (JSON, YAML)
    - Data (CSV, XML)
    - Documentation (Proto, Avro)
    - Build files (Dockerfile, package configs)
    - Database (SQL)
2. Intelligent Processing:
    - Language-specific handling
    - Structure preservation
    - Size limits
    - Error handling
3. Extensible Design:
    - Easy to add new file types
    - Configurable features
    - Fallback mechanisms
4. Special Handling:
    - Binary files
    - Large files
    - Unknown formats
    - Encoding detection
"""

import base64
import csv
import io
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import yaml

try:
    import chardet # Optional library
except ImportError:
    chardet = None

try:
    import html2text # Optional library
except ImportError:
    html2text = None

try:
    import markdown # Optional library
except ImportError:
    markdown = None

try:
    import nbformat # Optional library
except ImportError:
    nbformat = None

try:
    import sqlparse # Optional library
except ImportError:
    sqlparse = None

try:
    from PIL import Image  # Optional libray
except ImportError:
    Image = None

from pygments import highlight
from pygments.formatters import NullFormatter
from pygments.lexers import get_lexer_for_filename

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported file types for preprocessing"""

    CODE = auto()  # Programming languages
    NOTEBOOK = auto()  # Jupyter notebooks
    MARKUP = auto()  # Markdown, RST, etc.
    CONFIG = auto()  # YAML, TOML, JSON, etc.
    DATA = auto()  # CSV, XML, etc.
    DOCUMENTATION = auto()  # API docs, specs
    BUILD = auto()  # Build files, package configs
    DATABASE = auto()  # SQL, migrations
    BINARY = auto()  # Compiled code, executables
    UNKNOWN = auto()


class PreprocessingFeature(Enum):
    # Code cleaning
    REMOVE_COMMENTS = auto()
    NORMALIZE_WHITESPACE = auto()
    REMOVE_EMPTY_LINES = auto()
    MINIFY_CODE = auto()
    EXTRACT_DOCSTRINGS = auto()

    # Data normalization
    NORMALIZE_JSON = auto()
    NORMALIZE_YAML = auto()
    NORMALIZE_XML = auto()
    FLATTEN_DATA = auto()

    # Content extraction
    EXTRACT_SCHEMAS = auto()
    EXTRACT_DEPENDENCIES = auto()
    EXTRACT_IMPORTS = auto()
    EXTRACT_FUNCTIONS = auto()

    # Special handling
    CONVERT_NOTEBOOKS = auto()
    PROCESS_DOCUMENTATION = auto()
    HANDLE_TEMPLATES = auto()


class NotebookFeature(Enum):
    KEEP_MARKDOWN = auto()
    KEEP_CODE = auto()
    KEEP_OUTPUT = auto()
    KEEP_PLOTS = auto()
    KEEP_WIDGETS = auto()
    KEEP_METADATA = auto()


@dataclass
class PreprocessingConfig:
    enabled_features: list[PreprocessingFeature]
    notebook_features: list[NotebookFeature]

    # Size limits
    max_output_size: int = 1024
    max_plot_size: int = 2048
    max_file_size: int = 1024 * 1024  # 1MB

    # File type settings
    process_hidden_files: bool = False
    binary_file_extensions: set[str] = field(
        default_factory=lambda: {
            ".pyc",
            ".pyo",
            ".so",
            ".dll",
            ".dylib",
            ".class",
            ".exe",
            ".bin",
            ".pkl",
            ".pyd",
        }
    )

    # Language-specific settings - populated from centralized registry
    code_file_extensions: dict[str, set[str]] = field(default_factory=dict)


class ContentPreprocessor:
    """Preprocesses content before sharding"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._setup_language_extensions()
        self._setup_processors()

    def _setup_language_extensions(self):
        """Initialize language-extension mappings from centralized registry"""
        if not self.config.code_file_extensions:
            from .languages.extensions import get_registry, FileCategory

            registry = get_registry()
            code_languages = registry.get_languages_by_category(FileCategory.CODE)

            for language in code_languages:
                extensions = registry.get_extensions_for_language(language)
                if extensions:
                    self.config.code_file_extensions[language] = extensions

    def _setup_processors(self):
        """Initialize file type specific processors"""
        self.processors = {
            FileType.CODE: self._process_code,
            FileType.NOTEBOOK: self._process_notebook,
            FileType.MARKUP: self._process_markup,
            FileType.CONFIG: self._process_config,
            FileType.DATA: self._process_data,
            FileType.DOCUMENTATION: self._process_documentation,
            FileType.BUILD: self._process_build,
            FileType.DATABASE: self._process_database,
            FileType.BINARY: self._process_binary,
            FileType.UNKNOWN: self._process_unknown,
        }

    def preprocess(self, content: str, file_path: str) -> str:
        """Preprocess content based on file type"""
        try:
            file_type = self._detect_file_type(file_path)
            processor = self.processors[file_type]
            return processor(content, file_path)
        except Exception as e:
            logger.warning(f"Error preprocessing {file_path}: {e}")
            return content

    def _detect_file_type(self, file_path: str) -> FileType:
        """Detect file type based on extension and content"""
        path = Path(file_path)
        ext = path.suffix.lower()

        # Check binary files first
        if ext in self.config.binary_file_extensions:
            return FileType.BINARY

        # Check code files
        for lang, extensions in self.config.code_file_extensions.items():
            if ext in extensions:
                return FileType.CODE

        # Check other known types
        if ext == ".ipynb":
            return FileType.NOTEBOOK
        elif ext in {".md", ".rst", ".txt", ".adoc"}:
            return FileType.MARKUP
        elif ext in {".json", ".yaml", ".yml", ".toml", ".ini"}:
            return FileType.CONFIG
        elif ext in {".csv", ".xml", ".xlsx", ".parquet"}:
            return FileType.DATA
        elif ext in {".sql", ".ddl", ".dml"}:
            return FileType.DATABASE
        elif path.name in {"Dockerfile", "Makefile", "CMakeLists.txt"}:
            return FileType.BUILD
        elif ext in {".proto", ".avsc", ".thrift", ".swagger", ".raml"}:
            return FileType.DOCUMENTATION

        return FileType.UNKNOWN

    def _apply_feature(self, content: str, feature: PreprocessingFeature) -> str:
        """Apply preprocessing feature to content"""
        processors = {
            PreprocessingFeature.REMOVE_COMMENTS: self._remove_comments,
            PreprocessingFeature.NORMALIZE_WHITESPACE: self._normalize_whitespace,
            PreprocessingFeature.REMOVE_EMPTY_LINES: self._remove_empty_lines,
            PreprocessingFeature.MINIFY_CODE: self._minify_code,
            PreprocessingFeature.EXTRACT_DOCSTRINGS: self._extract_docstrings,
            PreprocessingFeature.NORMALIZE_JSON: self._normalize_json,
            PreprocessingFeature.NORMALIZE_YAML: self._normalize_yaml,
            PreprocessingFeature.NORMALIZE_XML: self._normalize_xml,
            PreprocessingFeature.FLATTEN_DATA: self._flatten_data,
            PreprocessingFeature.EXTRACT_SCHEMAS: self._extract_schemas,
            PreprocessingFeature.EXTRACT_DEPENDENCIES: self._extract_dependencies,
            PreprocessingFeature.EXTRACT_IMPORTS: self._extract_imports,
            PreprocessingFeature.EXTRACT_FUNCTIONS: self._extract_functions,
            PreprocessingFeature.CONVERT_NOTEBOOKS: self._convert_notebooks,
            PreprocessingFeature.PROCESS_DOCUMENTATION: self._process_documentation_feature,
            PreprocessingFeature.HANDLE_TEMPLATES: self._handle_templates,
        }

        processor = processors.get(feature)
        if processor:
            try:
                return processor(content)
            except Exception as e:
                logger.warning(f"Error applying feature {feature}: {e}")
                return content
        return content

    def _process_code(self, content: str, file_path: str) -> str:
        """Process code files with language-specific handling"""
        try:
            lexer = get_lexer_for_filename(file_path)

            # Apply enabled features
            processed = content
            for feature in self.config.enabled_features:
                processed = self._apply_feature(processed, feature)

            # Format based on language
            processed = highlight(processed, lexer, NullFormatter())

            return processed
        except Exception as e:
            logger.warning(f"Error processing code file {file_path}: {e}")
            return content

    def _process_notebook_old(self, content: str) -> str:
        notebook = json.loads(content)
        cells = []

        for cell in notebook["cells"]:
            if (
                cell["cell_type"] == "markdown"
                and NotebookFeature.KEEP_MARKDOWN in self.config.notebook_features
            ):
                cells.append(self._process_markdown_cell(cell))

            elif cell["cell_type"] == "code":
                if NotebookFeature.KEEP_CODE in self.config.notebook_features:
                    cells.append(self._process_code_cell(cell))

        return "\n\n".join(cells)

    def _process_notebook(self, content: str) -> str:
        """Process Jupyter notebooks"""
        if nbformat is None:
            return content
        try:
            notebook = nbformat.reads(content, as_version=4)
            cells = []

            for cell in notebook.cells:
                if (
                    cell.cell_type == "markdown"
                    and NotebookFeature.KEEP_MARKDOWN in self.config.notebook_features
                ):
                    cells.append(self._process_markdown_cell(cell))

                elif cell.cell_type == "code":
                    if NotebookFeature.KEEP_CODE in self.config.notebook_features:
                        processed_cell = self._process_code_cell(cell)
                        if processed_cell:
                            cells.append(processed_cell)

            return "\n\n".join(cells)
        except Exception as e:
            logger.warning(f"Error processing notebook: {e}")
            return content

    def _process_markdown_cell(self, cell: dict) -> str:
        """Process markdown cell from notebook"""
        if html2text is None or markdown is None:
            return cell.source
        try:
            # Convert markdown to plain text while preserving structure
            h = html2text.HTML2Text()
            h.body_width = 0  # Don't wrap lines
            return h.handle(markdown.markdown(cell.source))
        except Exception as e:
            logger.warning(f"Error processing markdown cell: {e}")
            return cell.source

    def _process_code_cell(self, cell: dict) -> str | None:
        """Process code cell from notebook"""
        try:
            parts = []

            # Add source code
            if cell.source.strip():
                parts.append(f"```\n{cell.source.strip()}\n```")

            # Add outputs if enabled
            if NotebookFeature.KEEP_OUTPUT in self.config.notebook_features:
                for output in cell.outputs:
                    if output.output_type == "stream":
                        if len(output.text) <= self.config.max_output_size:
                            parts.append(f"Output:\n{output.text.strip()}")

                    elif output.output_type == "display_data":
                        if NotebookFeature.KEEP_PLOTS in self.config.notebook_features:
                            for mime_type, data in output.data.items():
                                if mime_type.startswith("image/") and Image is not None:
                                    try:
                                        img_data = base64.b64decode(data)
                                        img = Image.open(io.BytesIO(img_data))
                                        if (
                                            img.size[0]
                                            * img.size[1]
                                            * len(img.getbands())
                                            <= self.config.max_plot_size
                                        ):
                                            parts.append(
                                                f"[Plot: {img.size[0]}x{img.size[1]} {img.mode}]"
                                            )
                                    except Exception as e:
                                        logger.warning(f"Error processing plot: {e}")

            return "\n".join(parts) if parts else None
        except Exception as e:
            logger.warning(f"Error processing code cell: {e}")
            return None

    def _process_markup(self, content: str, file_path: str) -> str:
        """Process markup files (Markdown, RST, etc.)"""
        if html2text is None or markdown is None:
            return content
        try:
            if file_path.endswith(".md"):
                h = html2text.HTML2Text()
                h.body_width = 0
                return h.handle(markdown.markdown(content))
            # Add support for other markup formats
            return content
        except Exception as e:
            logger.warning(f"Error processing markup file {file_path}: {e}")
            return content

    def _process_config(self, content: str, file_path: str) -> str:
        """Process configuration files"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext in {".json", ".yaml", ".yml"}:
                # Parse and normalize
                if ext == ".json":
                    data = json.loads(content)
                else:
                    data = yaml.safe_load(content)
                return json.dumps(data, indent=2, sort_keys=True)
            return content
        except Exception as e:
            logger.warning(f"Error processing config file {file_path}: {e}")
            return content

    def _process_data(self, content: str, file_path: str) -> str:
        """Process data files (CSV, XML, etc.)"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext == ".csv":
                # Process CSV while preserving structure
                reader = csv.reader(content.splitlines())
                return "\n".join(",".join(row) for row in reader)
            elif ext == ".xml":
                # Format XML
                root = ET.fromstring(content)
                return ET.tostring(root, encoding="unicode", method="xml")
            return content
        except Exception as e:
            logger.warning(f"Error processing data file {file_path}: {e}")
            return content

    def _process_documentation(self, content: str, file_path: str) -> str:
        """Process documentation files"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext in {".proto", ".avsc"}:
                # Extract schema information
                return self._extract_schemas(content)
            return content
        except Exception as e:
            logger.warning(f"Error processing documentation file {file_path}: {e}")
            return content

    def _process_build(self, content: str, file_path: str) -> str:
        """Process build and configuration files"""
        try:
            filename = Path(file_path).name
            if filename == "Dockerfile":
                # Extract key information from Dockerfile
                return self._process_dockerfile(content)
            elif filename in {"package.json", "setup.py", "pom.xml"}:
                # Extract dependencies and metadata
                return self._extract_dependencies(content)
            return content
        except Exception as e:
            logger.warning(f"Error processing build file {file_path}: {e}")
            return content

    def _process_database(self, content: str, file_path: str) -> str:
        """Process database-related files"""
        if sqlparse is None:
            return content
        try:
            if file_path.endswith(".sql"):
                # Format SQL
                return sqlparse.format(content, reindent=True, keyword_case="upper")
            return content
        except Exception as e:
            logger.warning(f"Error processing database file {file_path}: {e}")
            return content

    def _process_binary(self, content: str, file_path: str) -> str:
        """Process binary files (extract metadata only)"""
        try:
            # Return file metadata instead of content
            path = Path(file_path)
            return f"Binary file: {path.name}\nSize: {len(content)} bytes\nType: {path.suffix}"
        except Exception as e:
            logger.warning(f"Error processing binary file {file_path}: {e}")
            return f"Binary file: {file_path}"

    def _process_unknown(self, content: str, file_path: str) -> str:
        """Process unknown file types"""
        if chardet is None:
            return content
        try:
            # Try to detect encoding
            raw_bytes = content.encode() if isinstance(content, str) else content
            result = chardet.detect(raw_bytes)
            if result["confidence"] > 0.7:
                return content.decode(result["encoding"])
            return content
        except Exception as e:
            logger.warning(f"Error processing unknown file {file_path}: {e}")
            return content
