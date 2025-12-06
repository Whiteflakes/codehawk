"""Git commit lineage tracking."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional dependency - fail gracefully if not available
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logger.warning("GitPython not available - lineage tracking features will be disabled")


class LineageTracker:
    """Tracks commit lineage for files and code changes."""

    def __init__(self, repository_path: Path):
        """
        Initialize the lineage tracker.

        Args:
            repository_path: Path to git repository
        """
        if not GIT_AVAILABLE:
            raise ImportError("GitPython is required for lineage tracking. Install with: pip install GitPython")
        
        self.repository_path = repository_path
        self.repo: Optional[git.Repo] = None

    def open_repository(self):
        """Open the git repository."""
        try:
            self.repo = git.Repo(self.repository_path)
            logger.info(f"Opened repository: {self.repository_path}")
        except git.exc.InvalidGitRepositoryError:
            logger.error(f"Invalid git repository: {self.repository_path}")
            raise

    def get_file_commits(self, file_path: Path, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get commit history for a file.

        Args:
            file_path: Path to file relative to repository root
            limit: Maximum number of commits to retrieve

        Returns:
            List of commit information
        """
        if not self.repo:
            self.open_repository()

        commits = []
        
        try:
            relative_path = file_path.relative_to(self.repository_path)
            
            for commit in self.repo.iter_commits(paths=str(relative_path), max_count=limit):
                commits.append({
                    "hash": commit.hexsha,
                    "author": str(commit.author),
                    "author_email": commit.author.email,
                    "message": commit.message.strip(),
                    "timestamp": datetime.fromtimestamp(commit.committed_date),
                    "files_changed": len(commit.stats.files),
                })
        except Exception as e:
            logger.error(f"Error getting commits for {file_path}: {e}")

        return commits

    def get_recent_commits(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent commits in the repository.

        Args:
            limit: Maximum number of commits to retrieve

        Returns:
            List of commit information
        """
        if not self.repo:
            self.open_repository()

        commits = []
        
        try:
            for commit in self.repo.iter_commits(max_count=limit):
                files_changed = []
                try:
                    for diff in commit.diff(commit.parents[0] if commit.parents else None):
                        change_type = "modified"
                        if diff.new_file:
                            change_type = "added"
                        elif diff.deleted_file:
                            change_type = "deleted"
                        elif diff.renamed_file:
                            change_type = "renamed"
                        
                        files_changed.append({
                            "path": diff.b_path or diff.a_path,
                            "change_type": change_type,
                        })
                except Exception:
                    pass

                commits.append({
                    "hash": commit.hexsha,
                    "author": str(commit.author),
                    "author_email": commit.author.email,
                    "message": commit.message.strip(),
                    "timestamp": datetime.fromtimestamp(commit.committed_date),
                    "files_changed": files_changed,
                })
        except Exception as e:
            logger.error(f"Error getting recent commits: {e}")

        return commits

    def get_file_at_commit(self, file_path: Path, commit_hash: str) -> Optional[str]:
        """
        Get file content at a specific commit.

        Args:
            file_path: Path to file
            commit_hash: Commit hash

        Returns:
            File content or None if not found
        """
        if not self.repo:
            self.open_repository()

        try:
            relative_path = file_path.relative_to(self.repository_path)
            commit = self.repo.commit(commit_hash)
            blob = commit.tree / str(relative_path)
            return blob.data_stream.read().decode("utf-8")
        except Exception as e:
            logger.error(f"Error getting file at commit {commit_hash}: {e}")
            return None

    def get_blame_info(self, file_path: Path) -> Dict[int, Dict[str, Any]]:
        """
        Get blame information for a file (which commit last modified each line).

        Args:
            file_path: Path to file

        Returns:
            Dictionary mapping line numbers to commit info
        """
        if not self.repo:
            self.open_repository()

        blame_info = {}
        
        try:
            relative_path = file_path.relative_to(self.repository_path)
            blame = self.repo.blame("HEAD", str(relative_path))
            
            line_num = 1
            for commit, lines in blame:
                for line in lines:
                    blame_info[line_num] = {
                        "commit_hash": commit.hexsha,
                        "author": str(commit.author),
                        "timestamp": datetime.fromtimestamp(commit.committed_date),
                        "message": commit.message.strip(),
                    }
                    line_num += 1
        except Exception as e:
            logger.error(f"Error getting blame info for {file_path}: {e}")

        return blame_info

    def get_related_files(self, file_path: Path, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find files that are frequently modified together with the given file.

        Args:
            file_path: Path to file
            limit: Maximum number of related files to return

        Returns:
            List of related files with co-occurrence count
        """
        if not self.repo:
            self.open_repository()

        related_files: Dict[str, int] = {}
        
        try:
            relative_path = file_path.relative_to(self.repository_path)
            
            # Get commits that modified this file
            for commit in self.repo.iter_commits(paths=str(relative_path), max_count=100):
                # Find other files modified in the same commit
                try:
                    for diff in commit.diff(commit.parents[0] if commit.parents else None):
                        other_path = diff.b_path or diff.a_path
                        if other_path and other_path != str(relative_path):
                            related_files[other_path] = related_files.get(other_path, 0) + 1
                except Exception:
                    pass

            # Sort by co-occurrence count
            sorted_files = sorted(
                related_files.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:limit]

            return [
                {"path": path, "co_occurrence_count": count}
                for path, count in sorted_files
            ]
        except Exception as e:
            logger.error(f"Error finding related files for {file_path}: {e}")
            return []
