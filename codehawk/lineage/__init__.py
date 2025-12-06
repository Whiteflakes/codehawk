"""Git commit lineage tracking."""

import logging
from dataclasses import dataclass
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

    @dataclass
    class FileChange:
        path: str
        change_type: str
        fingerprint: Optional[str] = None
        previous_path: Optional[str] = None

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

    def get_blob_fingerprint(self, file_path: Path, commit_hash: str = "HEAD") -> Optional[str]:
        """Return the blob fingerprint for a file at a given commit."""
        if not self.repo:
            self.open_repository()

        try:
            relative_path = file_path.relative_to(self.repository_path)
            commit = self.repo.commit(commit_hash)
            blob = commit.tree / str(relative_path)
            return blob.hexsha
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(f"Unable to fingerprint {file_path} at {commit_hash}: {exc}")
            return None

    def get_commit_fingerprints(self, commit_hash: str = "HEAD") -> Dict[str, str]:
        """Walk a commit tree to map file paths to blob fingerprints."""
        if not self.repo:
            self.open_repository()

        commit = self.repo.commit(commit_hash)
        fingerprints: Dict[str, str] = {}

        def _walk_tree(tree, prefix: Path):
            for item in tree:
                if item.type == "blob":
                    fingerprints[str(prefix / item.name)] = item.hexsha
                elif item.type == "tree":
                    _walk_tree(item, prefix / item.name)

        _walk_tree(commit.tree, Path(""))
        return fingerprints

    def detect_changes(
        self, base_commit: str, target_commit: str = "HEAD"
    ) -> List["LineageTracker.FileChange"]:
        """Detect file changes between commits with rename awareness."""
        if not self.repo:
            self.open_repository()

        changes: List[LineageTracker.FileChange] = []
        seen: set = set()

        base_fingerprints = self.get_commit_fingerprints(base_commit)
        target_fingerprints = self.get_commit_fingerprints(target_commit)

        base_keys = set(base_fingerprints.keys())
        target_keys = set(target_fingerprints.keys())
        base_fingerprint_lookup = {v: k for k, v in base_fingerprints.items()}

        # Direct diff including rename metadata
        for diff in self.repo.commit(target_commit).diff(base_commit):
            change_type = "modified"
            path = diff.rename_to or diff.b_path or diff.a_path
            previous_path = diff.rename_from if diff.renamed_file else None

            if diff.new_file:
                change_type = "added"
            elif diff.deleted_file:
                change_type = "deleted"
            elif diff.renamed_file:
                change_type = "renamed"

            fingerprint = target_fingerprints.get(path or "")
            if change_type == "deleted":
                fingerprint = base_fingerprints.get(diff.a_path or "")
            elif change_type == "renamed":
                fingerprint = fingerprint or base_fingerprints.get(diff.a_path or "")
                previous_path = base_fingerprint_lookup.get(fingerprint) or previous_path
                if previous_path == path:
                    previous_path = base_fingerprint_lookup.get(fingerprint, previous_path)
                if fingerprint:
                    for candidate_path, candidate_fp in base_fingerprints.items():
                        if candidate_fp == fingerprint:
                            previous_path = candidate_path
                            break

            change = LineageTracker.FileChange(
                path=path or diff.a_path,
                change_type=change_type,
                fingerprint=fingerprint,
                previous_path=previous_path,
            )
            key = (change.path, change.change_type, change.previous_path)
            if key not in seen and (change_type != "renamed" or change.fingerprint):
                seen.add(key)
                changes.append(change)

        # Detect renames by matching fingerprints when diff metadata is unavailable
        base_only = base_keys - target_keys
        target_only = target_keys - base_keys

        fingerprint_lookup = {v: k for k, v in base_fingerprints.items()}
        for path in list(target_only):
            fingerprint = target_fingerprints[path]
            if fingerprint in fingerprint_lookup:
                changes.append(
                    LineageTracker.FileChange(
                        path=path,
                        change_type="renamed",
                        fingerprint=fingerprint,
                        previous_path=fingerprint_lookup[fingerprint],
                    )
                )
                base_only.discard(fingerprint_lookup[fingerprint])
                target_only.discard(path)

        # Remaining adds/deletes
        for path in target_only:
            changes.append(
                LineageTracker.FileChange(
                    path=path, change_type="added", fingerprint=target_fingerprints[path]
                )
            )

        for path in base_only:
            changes.append(
                LineageTracker.FileChange(
                    path=path, change_type="deleted", fingerprint=base_fingerprints[path]
                )
            )

        return changes

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
