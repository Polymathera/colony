import os
import re
import json
import requests
from pathlib import Path
from typing import Any
from overrides import override

from ..languages.dependency import DependencyConfig
from ......utils import setup_logger
from .base import BaseAnalyzer, FileContentCache

logger = setup_logger(__name__)


class DependencyAnalyzer(BaseAnalyzer):
    """Analyzes code dependencies with cross-language support"""

    def __init__(self, file_content_cache: FileContentCache, config: DependencyConfig | None = None):
        super().__init__("dependency", file_content_cache)
        self.config = config

    async def initialize(self):
        self.config = await DependencyConfig.check_or_get_component(self.config)
        await super().initialize()

    @override
    async def _analyze_file_impl(
        self,
        file_path: str,
        content: str,
        language: str | None = None,
        **kwargs
    ) -> dict[str, dict[str, Any]]:
        """
        Analyze file dependencies including cross-language bindings.

        Returns:
            dict mapping dependent files to their dependency info:
            {
                'file_path': {
                    'type': str,  # import, class, function, etc.
                    'confidence': float,
                    'cross_language': bool,
                    'binding_type': str | None,
                    'locations': list[int]  # Line numbers
                }
            }
        """
        try:
            cross_language = kwargs.get("cross_language", True)

            # Get language-specific patterns
            patterns = self.config.language_patterns.get(language, {})
            if not patterns:
                return self._get_fallback_result()

            dependencies = {}

            # Analyze each line for dependencies
            for line_num, line in enumerate(content.splitlines(), 1):
                # Check each pattern type
                for dep_type, type_patterns in patterns.items():
                    for pattern in type_patterns:
                        for match in pattern.finditer(line):
                            dep_info = self._process_dependency_match(
                                match, dep_type, language, line_num
                            )
                            if dep_info:
                                dep_file = dep_info.pop("file")
                                if dep_file in dependencies:
                                    # Merge with existing dependency
                                    self._merge_dependency_info(
                                        dependencies[dep_file], dep_info
                                    )
                                else:
                                    dependencies[dep_file] = dep_info

            # Check for cross-language bindings if enabled
            if cross_language:
                cross_deps = await self._analyze_cross_language_deps(
                    content, language, file_path
                )
                dependencies.update(cross_deps)

            return dependencies

        except Exception as e:
            logger.error(f"Dependency analysis error: {e}", exc_info=True)
            self.base_metrics.errors.labels("analysis").inc()
            return {}

    @override
    def _get_fallback_result(self) -> dict[str, Any]:
        """Return safe fallback result on error"""
        return {}

    async def _analyze_cross_language_deps(
        self, content: str, language: str, file_path: str
    ) -> dict[str, dict[str, Any]]:
        """Analyze cross-language dependencies"""
        cross_deps = {}

        # Check each language pair that includes our language
        for pair, bindings in self.config.known_bindings.items():
            langs: list[str] = pair.split("-")
            if language not in langs:
                continue

            other_lang = langs[1] if langs[0] == language else langs[0]

            # Check each binding type
            for binding_type, patterns in bindings.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        dep_info = self._process_cross_language_match(
                            match, binding_type, language, other_lang, file_path
                        )
                        if dep_info:
                            dep_file = dep_info.pop("file")
                            cross_deps[dep_file] = dep_info

        return cross_deps

    def _process_dependency_match(
        self, match: re.Match, dep_type: str, language: str, line_num: int
    ) -> dict[str, Any] | None:
        """Process a dependency match and return dependency info"""
        try:
            dep_name = match.group(1)
            if not dep_name:
                return None

            return {
                "file": self._resolve_dependency_path(dep_name, language),
                "type": dep_type,
                "confidence": self._calculate_confidence(dep_type, language),
                "cross_language": False,
                "locations": [line_num],
            }

        except Exception as e:
            logger.error(f"Error processing match: {e}")
            return None

    def _process_cross_language_match(
        self,
        match: re.Match,
        binding_type: str,
        source_lang: str,
        target_lang: str,
        file_path: str,
    ) -> dict[str, Any] | None:
        """Process a cross-language binding match"""
        try:
            dep_path = self._resolve_cross_language_path(
                match.group(1), source_lang, target_lang, file_path
            )

            return {
                "file": dep_path,
                "type": "cross_language",
                "binding_type": binding_type,
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence": self._calculate_cross_language_confidence(
                    binding_type, source_lang, target_lang
                ),
                "cross_language": True,
                "locations": [match.start()],
            }

        except Exception as e:
            logger.error(f"Error processing cross-language match: {e}")
            return None

    def _merge_dependency_info(self, existing: dict[str, Any], new: dict[str, Any]):
        """Merge new dependency info into existing entry"""
        existing["locations"].extend(new["locations"])
        existing["confidence"] = max(existing["confidence"], new["confidence"])
        if new.get("cross_language"):
            existing["cross_language"] = True
            existing["binding_type"] = new.get("binding_type")

    def _resolve_dependency_path(self, dep_name: str, language: str) -> str:
        """Resolve dependency name to actual file path"""
        try:
            # Language-specific path resolution
            if language == "python":
                return self._resolve_python_import(dep_name)
            elif language in ("javascript", "typescript"):
                return self._resolve_js_import(dep_name)
            elif language in ("java", "kotlin"):
                return self._resolve_jvm_import(dep_name)
            else:
                return dep_name.replace(".", "/")

        except Exception as e:
            logger.error(f"Path resolution error: {e}", exc_info=True)
            return dep_name

    def _resolve_python_import(self, import_name: str) -> str:
        """Resolve Python import to file path"""
        # Convert dotted import to path
        parts = import_name.split(".")

        # Handle relative imports
        if parts[0] == "":  # Leading dot
            parts = parts[1:]

        # Add .py extension if missing
        if not parts[-1].endswith(".py"):
            parts[-1] += ".py"

        return "/".join(parts)

    def _resolve_js_import(self, import_name: str) -> str:
        """Resolve JavaScript/TypeScript import to file path"""
        # Remove quotes and leading/trailing whitespace
        path = import_name.strip("'\"").strip()

        # Handle node_modules imports
        if not path.startswith(".") and not path.startswith("/"):
            return f"node_modules/{path}"

        # Add extension if missing
        if not Path(path).suffix:
            path += ".js"

        return path

    def _resolve_jvm_import(self, import_name: str) -> str:
        """Resolve Java/Kotlin import to file path"""
        # Convert package notation to path
        path = import_name.replace(".", "/")

        # Add .java extension if missing
        if not path.endswith(".java"):
            path += ".java"

        return path

    def _resolve_cross_language_path(
        self, dep_name: str, source_lang: str, target_lang: str, source_file: str
    ) -> str:
        """Resolve cross-language dependency path"""
        try:
            # Get source and target directories
            source_dir = Path(source_file).parent

            # Language-specific resolution
            if f"{source_lang}-{target_lang}" == "python-typescript":
                return self._resolve_py_ts_path(dep_name, source_dir)
            elif f"{source_lang}-{target_lang}" == "typescript-python":
                return self._resolve_ts_py_path(dep_name, source_dir)
            else:
                return dep_name

        except Exception as e:
            logger.error(f"Cross-language path resolution error: {e}", exc_info=True)
            return dep_name

    def _calculate_confidence(self, dep_type: str, language: str) -> float:
        """Calculate confidence score for dependency"""
        try:
            # Base confidence by type
            base_scores = {
                "import": 0.9,
                "class": 0.8,
                "function": 0.7,
                "interface": 0.8,
            }

            # Language-specific adjustments
            lang_multipliers = {
                "python": 1.0,
                "typescript": 0.95,
                "javascript": 0.9,
                "java": 1.0,
                "kotlin": 0.95,
            }

            base = base_scores.get(dep_type, 0.5)
            multiplier = lang_multipliers.get(language, 0.8)

            return min(base * multiplier, 1.0)

        except Exception as e:
            logger.error(f"Confidence calculation error: {e}", exc_info=True)
            return self.config.min_confidence

    def _calculate_cross_language_confidence(
        self, binding_type: str, source_lang: str, target_lang: str
    ) -> float:
        """Calculate confidence score for cross-language binding"""
        try:
            # Base confidence by binding type
            base_scores = {"decorators": 0.9, "imports": 0.85, "markers": 0.7}

            # Language pair adjustments
            pair_multipliers = {
                "python-typescript": 0.95,
                "typescript-python": 0.95,
                "python-java": 0.9,
                "java-python": 0.9,
            }

            base = base_scores.get(binding_type, 0.6)
            pair = f"{source_lang}-{target_lang}"
            multiplier = pair_multipliers.get(pair, 0.8)

            return min(base * multiplier, 1.0)

        except Exception as e:
            logger.error(
                f"Cross-language confidence calculation error: {e}", exc_info=True
            )
            return self.config.min_confidence





class GitDependencyTracker:
    """
    A class that reads all the files of a git repo (cloned locally),
    extracts all the development and production dependencies from the relevant files,
    and lists all the corresponding git repo URLs for each build target or package.
    It also detects and parses .gitmodules files to track submodule dependencies.
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.dependency_files = {
            "python": ["requirements.txt", "pyproject.toml", "setup.py"],
            "javascript": ["package.json"],
            "cmake": ["CMakeLists.txt"],
            "ruby": ["Gemfile"],
            "php": ["composer.json"],
            "rust": ["Cargo.toml"],
            "go": ["go.mod"],
        }
        self.targets = {}
        self.submodules = {}

    def extract_dependencies(self):
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file in [
                    item
                    for sublist in self.dependency_files.values()
                    for item in sublist
                ]:
                    file_path = os.path.join(root, file)
                    self._parse_dependency_file(file_path)
            if ".gitmodules" in files:
                self._parse_gitmodules(os.path.join(root, ".gitmodules"))

    def _parse_dependency_file(self, file_path: str):
        file_name = os.path.basename(file_path)
        if file_name in self.dependency_files["python"]:
            self._parse_python_dependencies(file_path)
        elif file_name in self.dependency_files["javascript"]:
            self._parse_javascript_dependencies(file_path)
        elif file_name in self.dependency_files["cmake"]:
            self._parse_cmake_dependencies(file_path)
        # Add more parsing methods for other dependency file types as needed

    def _parse_python_dependencies(self, file_path: str):
        with open(file_path) as file:
            content = file.read()
            if file_path.endswith("requirements.txt"):
                deps = [
                    line.strip().split("==")[0]
                    for line in content.split("\n")
                    if line.strip() and not line.startswith("#")
                ]
                self.targets[os.path.relpath(file_path, self.repo_path)] = {
                    "language": "python",
                    "dependencies": deps,
                }
            elif file_path.endswith("pyproject.toml"):
                import toml

                parsed = toml.loads(content)
                project_name = (
                    parsed.get("tool", {})
                    .get("poetry", {})
                    .get("name", os.path.basename(os.path.dirname(file_path)))
                )
                deps = list(
                    parsed.get("tool", {})
                    .get("poetry", {})
                    .get("dependencies", {})
                    .keys()
                )
                self.targets[project_name] = {
                    "language": "python",
                    "dependencies": deps,
                }
            elif file_path.endswith("setup.py"):
                # This is a very basic parser and might miss some cases
                deps = re.findall(r"install_requires=\[(.*?)\]", content, re.DOTALL)
                if deps:
                    deps = re.findall(r"'(.*?)'", deps[0])
                setup_name = re.search(r"name=['\"](.+?)['\"]", content)
                target_name = (
                    setup_name.group(1)
                    if setup_name
                    else os.path.basename(os.path.dirname(file_path))
                )
                self.targets[target_name] = {"language": "python", "dependencies": deps}

    def _parse_javascript_dependencies(self, file_path: str):
        with open(file_path) as file:
            content = json.load(file)
            package_name = content.get(
                "name", os.path.basename(os.path.dirname(file_path))
            )
            deps = list(content.get("dependencies", {}).keys()) + list(
                content.get("devDependencies", {}).keys()
            )
            self.targets[package_name] = {
                "language": "javascript",
                "dependencies": deps,
            }

    def _parse_cmake_dependencies(self, file_path: str):
        with open(file_path) as file:
            content = file.read()
            project_name = re.search(r"project\((.*?)\)", content)
            target_name = (
                project_name.group(1)
                if project_name
                else os.path.basename(os.path.dirname(file_path))
            )
            deps = re.findall(r"find_package\((.*?)\)", content)
            self.targets[target_name] = {"language": "cmake", "dependencies": deps}

    def _parse_gitmodules(self, file_path: str):
        with open(file_path) as file:
            content = file.read()
            submodules = re.findall(
                r'\[submodule "(.*?)"\]\s*path = (.*?)\s*url = (.*?)(?=\[|\Z)',
                content,
                re.DOTALL,
            )
            for name, path, url in submodules:
                self.submodules[name] = {"path": path.strip(), "url": url.strip()}
                submodule_path = os.path.join(os.path.dirname(file_path), path.strip())
                if os.path.exists(submodule_path):
                    submodule_tracker = GitDependencyTracker(submodule_path)
                    submodule_analysis = submodule_tracker.analyze()
                    self.targets[f"submodule:{name}"] = submodule_analysis["targets"]
                    self.submodules[name]["dependencies"] = submodule_analysis[
                        "repo_urls"
                    ]

    def get_repo_urls(self):
        repo_urls = {}
        for target, info in self.targets.items():
            if isinstance(info, dict) and "language" in info:
                repo_urls[target] = []
                for dep in info["dependencies"]:
                    if info["language"] == "python":
                        repo_urls[target].append(f"https://github.com/pypi/{dep}")
                    elif info["language"] == "javascript":
                        repo_urls[target].append(f"https://github.com/npm/{dep}")
                    elif info["language"] == "cmake":
                        repo_urls[target].append(f"https://github.com/{dep}")
                    # Add more mappings for other languages as needed
            elif isinstance(info, dict):  # This is a submodule
                repo_urls[target] = info
        return repo_urls

    def analyze(self):
        self.extract_dependencies()
        return {
            "targets": self.targets,
            "repo_urls": self.get_repo_urls(),
            "submodules": self.submodules,
        }



class DependencyParser:

    def __init__(self):
        self.dependencies = []

    def parse(self, file_name: str, content: str):
        if file_name in ["pyproject.toml", "setup.py", "requirements.txt"]:
            self._parse_python_dependencies(content)
        elif file_name == "CMakeLists.txt":
            self._parse_cmake_dependencies(content)
        elif file_name == "package.json":
            self._parse_javascript_dependencies(content)

    def _parse_python_dependencies(self, content: str):
        import ast

        import toml
        from packaging import requirements

        # Parse pyproject.toml
        try:
            pyproject_data = toml.loads(content)
            if "tool" in pyproject_data and "poetry" in pyproject_data["tool"]:
                self.dependencies.extend(
                    pyproject_data["tool"]["poetry"].get("dependencies", {}).keys()
                )
        except toml.TomlDecodeError:
            pass

        # Parse setup.py
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and node.func.id == "setup":
                    for keyword in node.keywords:
                        if keyword.arg == "install_requires":
                            self.dependencies.extend([e.s for e in keyword.value.elts])
        except SyntaxError:
            pass

        # Parse requirements.txt
        for line in content.split("\n"):
            try:
                req = requirements.Requirement(line)
                self.dependencies.append(req.name)
            except requirements.InvalidRequirement:
                pass

    def _parse_cmake_dependencies(self, content: str):
        import re

        find_package_pattern = re.compile(r"find_package\s*\(\s*(\w+)", re.IGNORECASE)
        self.dependencies.extend(find_package_pattern.findall(content))

    def _parse_javascript_dependencies(self, content: str):
        import json

        try:
            package_data = json.loads(content)
            self.dependencies.extend(package_data.get("dependencies", {}).keys())
            self.dependencies.extend(package_data.get("devDependencies", {}).keys())
        except json.JSONDecodeError:
            pass

    def get_dependencies(self) -> list[str]:
        return self.dependencies



from .....utils.git.clients import GitHubClient, GitLabClient, GitClientBase


class GitDependencyCrawler:
    """
    A class that walks all repositories on GitLab and GitHub (starting from a given seed repository) and builds
    a dependency graph among those repositories based on the build or packaging files in each
    repository (e.g., pyproject.toml, setup.py, requirements.txt, CMakeLists.txt, package.json, etc.).
    """

    def __init__(self):
        self.dependency_graph: dict[str, list[str]] = {}
        self.github_client: GitHubClient | None = None
        self.gitlab_client: GitLabClient | None = None

    async def initialize(self):
        self.github_client = await GitHubClient()
        self.gitlab_client = await GitLabClient()
        await self.github_client.initialize()
        await self.gitlab_client.initialize()

    def _get_client(self, repo_url: str) -> GitClientBase:
        if "github.com" in repo_url:
            return self.github_client
        elif "gitlab.com" in repo_url:
            return self.gitlab_client
        else:
            raise ValueError(f"Unsupported repository: {repo_url}")

    def get_dependency_graph(self) -> dict[str, list[str]]:
        return self.dependency_graph

    def crawl(self, start_repo: str | None = None):
        if start_repo:
            self._crawl_repo(start_repo)
        else:
            client = self._get_client(start_repo)
            url = f"{self.base_url}repositories"
            while url:
                data = client._fetch_data_sync(url)
                if data:
                    for repo in data:
                        self._crawl_repo(repo["full_name"])
                    url = data.get("next")
                else:
                    break

    def crawl(self):
        url = f"{self.base_url}projects"
        while url:
            data = self._make_request(url)
            if data:
                for project in data:
                    self._crawl_project(project["id"], project["path_with_namespace"])
                url = requests.utils.parse_header_links(
                    data.headers.get("Link", "")
                ).get("next")
            else:
                break

    def _crawl_repo(self, repo: str):
        dependencies = self._parse_dependencies(repo)
        self.dependency_graph[repo] = dependencies

    def _parse_dependencies(self, repo: str) -> list[str]:
        dependencies = []
        files_to_check = [
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "CMakeLists.txt",
            "package.json",
        ]

        client = self._get_client(repo)
        for file in files_to_check:
            content = client.get_file_contents_sync(repo, file)
            if content:
                parser = DependencyParser()
                parser.parse(file, content)
                dependencies.extend(parser.get_dependencies())

        return list(set(dependencies))

    def _crawl_project(self, project_id: int, project_name: str):
        dependencies = self._parse_dependencies(project_id)
        self.dependency_graph[project_name] = dependencies


