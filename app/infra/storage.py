"""Audio file storage on a local disk (a Docker volume shared by api and worker).

In production this class would be swapped for S3/GCS with the same methods.
"""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

READ_BLOCK = 1024 * 1024  # 1 MB


class FileTooLargeError(Exception):
    pass


@dataclass
class StoredFile:
    path: Path
    size: int
    sha256: str


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.uploads = Path(root) / "uploads"
        self.work = Path(root) / "work"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)

    def save_upload(self, stream: BinaryIO, job_id: str, extension: str, max_bytes: int) -> StoredFile:
        """Copy the upload to disk block by block, hashing it and enforcing the size limit.

        The file is never held in memory as a whole, so large uploads are fine.
        """
        path = self.uploads / f"{job_id}{extension}"
        digest = hashlib.sha256()
        size = 0
        with path.open("wb") as out:
            while block := stream.read(READ_BLOCK):
                size += len(block)
                if size > max_bytes:
                    out.close()
                    path.unlink(missing_ok=True)
                    raise FileTooLargeError(f"File is larger than {max_bytes // (1024 * 1024)} MB")
                digest.update(block)
                out.write(block)
        return StoredFile(path=path, size=size, sha256=digest.hexdigest())

    def workdir(self, job_id: str) -> Path:
        """Scratch space for one job: normalized audio and chunk checkpoints."""
        return self.work / job_id

    def clean_workdir(self, job_id: str) -> None:
        shutil.rmtree(self.workdir(job_id), ignore_errors=True)

    def delete(self, job_id: str) -> None:
        for path in self.uploads.glob(f"{job_id}*"):
            path.unlink(missing_ok=True)
        self.clean_workdir(job_id)
