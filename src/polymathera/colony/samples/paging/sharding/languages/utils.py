


from pathlib import Path

import logging

logger = logging.getLogger(__name__)



def detect_language(file_path: str) -> str | None:
    """Detect programming language from file path"""
    try:
        ext = Path(file_path).suffix.lower()

        # Common language extensions
        EXTENSION_MAP = {
            '.py': 'python',
            '.pyi': 'python',
            '.pyx': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.cpp': 'cpp',
            '.hpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.scala': 'scala',
            '.swift': 'swift'
        }
        # TODO: Add more mappings as needed

        return EXTENSION_MAP.get(ext)

    except Exception as e:
        logger.error(f"Language detection error: {e}", exc_info=True)
        return None


def is_comment(line: str, language: str) -> bool:
    """Check if line is a comment"""
    comment_markers = {
        'python': '#',
        'typescript': '//',
        'javascript': '//',
        'java': '//',
        'kotlin': '//',
        'rust': '//',
        'go': '//',
        'swift': '//'
    }
    marker = comment_markers.get(language)
    return marker and line.lstrip().startswith(marker)
