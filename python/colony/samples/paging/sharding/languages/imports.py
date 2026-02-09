# Language-specific settings
from .extensions import get_registry


def get_language_configs() -> dict[str, dict]:
    """Get language configurations from centralized registry"""
    registry = get_registry()
    configs = {}

    # Get all registered languages
    for language_name in ["python", "typescript", "java", "kotlin", "rust", "go", "swift"]:
        lang_info = registry.get_language_info(language_name)
        if lang_info:
            configs[language_name] = {
                "import_patterns": lang_info.import_patterns,
                "type_patterns": lang_info.type_patterns,
                "extensions": list(lang_info.extensions),
            }

    return configs
