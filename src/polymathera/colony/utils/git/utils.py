import shutil
import subprocess
from pathlib import Path
import logging
import git

logger = logging.getLogger(__name__)


def is_git_installed():
    return shutil.which("git") is not None


def is_git_repo(path: Path):
    return (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def init_git_repo(path: Path):
    subprocess.run(["git", "init"], cwd=path)


def has_uncommitted_changes(path: Path):
    return bool(
        subprocess.run(
            ["git", "diff", "--exit-code"],
            cwd=path,
            capture_output=True,
            text=True,
        ).returncode
    )


def filter_files_with_uncommitted_changes(
    basepath: Path, files_dict: dict[str, str]
) -> list[Path]:
    files_with_diff = (
        subprocess.run(
            ["git", "diff", "--name-only"], cwd=basepath, capture_output=True, text=True
        )
        .stdout.decode()
        .splitlines()
    )
    return [f for f in files_dict.keys() if f in files_with_diff]


def stage_files(path: Path, files: list[str]):
    subprocess.run(["git", "add", *files], cwd=path)


def filter_by_gitignore(path: Path, file_list: list[str]) -> list[str]:
    out = subprocess.run(
        ["git", "-C", ".", "check-ignore", "--no-index", "--stdin"],
        cwd=path,
        input="\n".join(file_list).encode(),
        capture_output=True,
        text=True,
    )
    paths = out.stdout.decode().splitlines()
    # return file_list but filter out the results from git check-ignore
    return [f for f in file_list if f not in paths]


def stage_uncommitted_to_git(path, files_dict, improve_mode):
    # Check if there's a git repo and verify that there aren't any uncommitted changes
    if is_git_installed() and not improve_mode:
        if not is_git_repo(path):
            print("\nInitializing an empty git repository")
            init_git_repo(path)

    if is_git_repo(path):
        modified_files = filter_files_with_uncommitted_changes(path, files_dict)
        if modified_files:
            print(
                "Staging the following uncommitted files before overwriting: ",
                ", ".join(modified_files),
            )
            stage_files(path, modified_files)


def validate_git_repository(repo: git.Repo) -> bool:
    """Validate that the Git repository is in a healthy state"""
    try:
        repo_path = Path(repo.working_dir)

        # Check if .git directory exists
        if not (repo_path / '.git').exists():
            logger.error(f"No .git directory found in {repo_path}")
            return False

        # Check if we can access the HEAD commit
        try:
            commit_hash = repo.head.commit.hexsha
            logger.debug(f"Repository HEAD commit: {commit_hash}")
        except Exception as e:
            logger.error(f"Cannot access HEAD commit: {e}")
            return False

        # Check if we can traverse the tree
        try:
            blob_count = sum(1 for obj in repo.head.commit.tree.traverse() if obj.type == "blob")
            logger.debug(f"Repository contains {blob_count} blobs")
        except Exception as e:
            logger.error(f"Cannot traverse repository tree: {e}")
            return False

        # Try to read a small sample of blobs to check for corruption
        try:
            sample_count = 0
            for blob in repo.head.commit.tree.traverse():
                if blob.type == "blob" and sample_count < 3:  # Test first 3 blobs
                    try:
                        data = blob.data_stream.read()
                        if not data:
                            logger.warning(f"Empty blob found: {blob.path}")
                        sample_count += 1
                    except Exception as blob_error:
                        logger.warning(f"Cannot read blob {blob.path}: {blob_error}")
                        # Don't fail validation for individual blob errors

            logger.debug(f"Validated {sample_count} sample blobs")
        except Exception as e:
            logger.warning(f"Blob validation failed: {e}")
            # Don't fail validation for blob issues - we'll handle them in processing

        return True

    except Exception as e:
        logger.error(f"Git repository validation failed: {e}")
        return False


def configure_git_safety(repo: git.Repo) -> None:
    """Configure Git safety settings to handle ownership issues"""
    try:
        repo_path = Path(repo.working_dir)

        # Add the repository to safe directories
        try:
            repo.git.config('--global', '--add', 'safe.directory', str(repo_path))
            logger.debug(f"Added {repo_path} to Git safe directories")
        except Exception as e:
            logger.warning(f"Failed to add {repo_path} to Git safe directories: {e}")

        # Set ownership to current user if possible
        try:
            import os
            if hasattr(os, 'chown') and repo_path.exists():
                uid = os.getuid()
                gid = os.getgid()
                for root, dirs, files in os.walk(repo_path):
                    for d in dirs:
                        try:
                            os.chown(os.path.join(root, d), uid, gid)
                        except (OSError, PermissionError):
                            pass
                    for f in files:
                        try:
                            os.chown(os.path.join(root, f), uid, gid)
                        except (OSError, PermissionError):
                            pass
        except Exception as e:
            logger.debug(f"Could not fix ownership for {repo_path}: {e}")

    except Exception as e:
        logger.warning(f"Git safety configuration failed: {e}")


