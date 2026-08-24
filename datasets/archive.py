"""Safe verification and extraction of the canonical benchmark archive.

The official dataset is a large password-protected ZIP64 archive.  Archive
member names are untrusted even after the outer archive digest has been
verified: this module validates the complete central directory and extracts
members manually instead of using :meth:`zipfile.ZipFile.extractall`.

The extracted-tree manifest is derived from the authenticated central
directory.  It records every regular file's uncompressed size and CRC32 plus
the complete directory set.  Verification rejects missing, additional, or
modified paths after extraction.
"""

from __future__ import annotations

from contextlib import contextmanager
import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
from typing import BinaryIO, Iterator
import unicodedata
import zipfile
import zlib


EXACT_ARCHIVE_FILENAME = "benchmarks.zip"
EXACT_ARCHIVE_SIZE_BYTES = 4_685_463_657
EXACT_ARCHIVE_SHA256 = (
    "ce0d153a1a78a3a752b29ec2e60167134b6b20c3249db2fe92f9fc1b8b8a9181"
)
EXACT_ARCHIVE_FILE_COUNT = 193
EXACT_ARCHIVE_DIRECTORY_ENTRY_COUNT = 113
EXACT_ARCHIVE_UNCOMPRESSED_SIZE_BYTES = 6_918_816_685
EXACT_ARCHIVE_PASSWORD = "LocalIntrinsicDimensionBenchmarks"

_HASH_CHUNK_SIZE = 8 * 1024 * 1024
_COPY_CHUNK_SIZE = 1024 * 1024
_ENCRYPTED_FLAG = 0x1


class ArchiveError(RuntimeError):
    """Base class for benchmark archive failures."""


class ArchiveValidationError(ArchiveError):
    """Raised when an archive violates its integrity or path contract."""


class ArchiveExtractionError(ArchiveError):
    """Raised when a validated archive cannot be extracted safely."""


@dataclass(frozen=True, slots=True)
class ArchiveMemberManifest:
    """Authenticated metadata for one regular file in an archive."""

    path: str
    size_bytes: int
    compressed_size_bytes: int
    crc32: str
    encrypted: bool


@dataclass(frozen=True, slots=True)
class ArchiveTreeManifest:
    """Manifest used to validate a complete extracted archive tree."""

    archive_size_bytes: int
    archive_sha256: str
    file_count: int
    directory_entry_count: int
    uncompressed_size_bytes: int
    directories: tuple[str, ...]
    files: tuple[ArchiveMemberManifest, ...]


def _validate_expected_digest(expected_sha256: str) -> str:
    digest = expected_sha256.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return digest


def _sha256_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while chunk := stream.read(_HASH_CHUNK_SIZE):
        digest.update(chunk)
        total += len(chunk)
    return total, digest.hexdigest()


def _safe_member_path(info: zipfile.ZipInfo) -> tuple[str, bool]:
    """Return a canonical POSIX member path and whether it is a directory."""

    original = info.orig_filename
    if not original:
        raise ArchiveValidationError("archive contains an empty member name")
    if "\x00" in original:
        raise ArchiveValidationError(
            f"archive member contains a NUL byte: {original!r}"
        )
    if "\\" in original:
        raise ArchiveValidationError(
            f"archive member uses a Windows path separator: {original!r}"
        )
    if any(ord(character) < 32 for character in original):
        raise ArchiveValidationError(
            f"archive member contains a control character: {original!r}"
        )

    is_directory = info.is_dir()
    without_trailing_slash = original[:-1] if is_directory else original
    if not without_trailing_slash or original.startswith("/"):
        raise ArchiveValidationError(
            f"archive member is empty or absolute: {original!r}"
        )
    if not is_directory and original.endswith("/"):
        raise ArchiveValidationError(
            f"archive member has inconsistent directory metadata: {original!r}"
        )

    parts = without_trailing_slash.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArchiveValidationError(
            f"archive member is not a canonical relative path: {original!r}"
        )
    if PureWindowsPath(without_trailing_slash).drive:
        raise ArchiveValidationError(
            f"archive member has a Windows drive prefix: {original!r}"
        )

    canonical = PurePosixPath(*parts).as_posix()
    if canonical != without_trailing_slash:
        raise ArchiveValidationError(
            f"archive member is not canonical: {original!r}"
        )
    return canonical, is_directory


def _validate_member_type(
    info: zipfile.ZipInfo,
    *,
    path: str,
    is_directory: bool,
) -> None:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ArchiveValidationError(
            f"archive member is a link or special file: {path!r}"
        )
    if is_directory and file_type == stat.S_IFREG:
        raise ArchiveValidationError(
            f"archive directory is marked as a regular file: {path!r}"
        )
    if not is_directory and file_type == stat.S_IFDIR:
        raise ArchiveValidationError(
            f"archive file is marked as a directory: {path!r}"
        )


def _collision_key(path: str) -> str:
    """Model common case-insensitive and Unicode-normalizing filesystems."""

    return unicodedata.normalize("NFC", path).casefold()


def _manifest_from_zip(
    archive: zipfile.ZipFile,
    *,
    archive_size_bytes: int,
    archive_sha256: str,
    expected_file_count: int,
    expected_uncompressed_size_bytes: int,
    expected_directory_entry_count: int | None,
    require_encrypted: bool,
) -> ArchiveTreeManifest:
    entries: dict[str, bool] = {}
    collision_keys: dict[str, str] = {}
    directories: set[str] = set()
    files: list[ArchiveMemberManifest] = []
    directory_entry_count = 0

    for info in archive.infolist():
        path, is_directory = _safe_member_path(info)
        _validate_member_type(info, path=path, is_directory=is_directory)

        if path in entries:
            raise ArchiveValidationError(f"archive repeats member path: {path!r}")
        key = _collision_key(path)
        if key in collision_keys:
            raise ArchiveValidationError(
                "archive member paths collide on a normalized filesystem: "
                f"{collision_keys[key]!r} and {path!r}"
            )
        entries[path] = is_directory
        collision_keys[key] = path

        parents = PurePosixPath(path).parents
        directories.update(
            parent.as_posix() for parent in parents if parent.as_posix() != "."
        )
        if is_directory:
            directory_entry_count += 1
            directories.add(path)
            continue

        encrypted = bool(info.flag_bits & _ENCRYPTED_FLAG)
        if require_encrypted and not encrypted:
            raise ArchiveValidationError(
                f"archive file is not password protected: {path!r}"
            )
        files.append(
            ArchiveMemberManifest(
                path=path,
                size_bytes=info.file_size,
                compressed_size_bytes=info.compress_size,
                crc32=f"{info.CRC:08x}",
                encrypted=encrypted,
            )
        )

    # Include implicit parent directories in collision checks.  Otherwise an
    # archive containing ``A/one`` and ``a/two`` could pass validation but
    # merge the directories on a case-insensitive filesystem.  The same table
    # detects a regular file used as another member's parent.
    filesystem_nodes: dict[str, bool] = {
        directory: True for directory in directories
    }
    for path, is_directory in entries.items():
        if path in filesystem_nodes and filesystem_nodes[path] != is_directory:
            raise ArchiveValidationError(
                f"archive file/directory path conflict: {path!r}"
            )
        filesystem_nodes[path] = is_directory

    normalized_nodes: dict[str, str] = {}
    for path in sorted(filesystem_nodes):
        key = _collision_key(path)
        if key in normalized_nodes and normalized_nodes[key] != path:
            raise ArchiveValidationError(
                "archive paths collide on a normalized filesystem: "
                f"{normalized_nodes[key]!r} and {path!r}"
            )
        normalized_nodes[key] = path

    files.sort(key=lambda member: member.path)
    if len(files) != expected_file_count:
        raise ArchiveValidationError(
            "archive file count mismatch: "
            f"expected {expected_file_count}, found {len(files)}"
        )
    uncompressed_size = sum(member.size_bytes for member in files)
    if uncompressed_size != expected_uncompressed_size_bytes:
        raise ArchiveValidationError(
            "archive uncompressed size mismatch: "
            f"expected {expected_uncompressed_size_bytes}, found {uncompressed_size}"
        )
    if (
        expected_directory_entry_count is not None
        and directory_entry_count != expected_directory_entry_count
    ):
        raise ArchiveValidationError(
            "archive directory entry count mismatch: "
            f"expected {expected_directory_entry_count}, found {directory_entry_count}"
        )

    return ArchiveTreeManifest(
        archive_size_bytes=archive_size_bytes,
        archive_sha256=archive_sha256,
        file_count=len(files),
        directory_entry_count=directory_entry_count,
        uncompressed_size_bytes=uncompressed_size,
        directories=tuple(sorted(directories)),
        files=tuple(files),
    )


@contextmanager
def _validated_archive(
    path: str | Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_file_count: int,
    expected_uncompressed_size_bytes: int,
    expected_directory_entry_count: int | None,
    require_encrypted: bool,
) -> Iterator[tuple[zipfile.ZipFile, ArchiveTreeManifest]]:
    if expected_size_bytes < 0:
        raise ValueError("expected_size_bytes cannot be negative")
    if expected_file_count < 0:
        raise ValueError("expected_file_count cannot be negative")
    if expected_uncompressed_size_bytes < 0:
        raise ValueError("expected_uncompressed_size_bytes cannot be negative")
    if expected_directory_entry_count is not None and expected_directory_entry_count < 0:
        raise ValueError("expected_directory_entry_count cannot be negative")
    expected_digest = _validate_expected_digest(expected_sha256)

    source = Path(path)
    try:
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ArchiveValidationError(
                    f"archive is not a regular file: {source}"
                )
            actual_size, actual_digest = _sha256_stream(stream)
            after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ArchiveValidationError(
                    f"archive changed while it was being hashed: {source}"
                )
            if actual_size != expected_size_bytes:
                raise ArchiveValidationError(
                    "archive size mismatch: "
                    f"expected {expected_size_bytes}, found {actual_size}"
                )
            if actual_digest != expected_digest:
                raise ArchiveValidationError(
                    "archive SHA-256 mismatch: "
                    f"expected {expected_digest}, found {actual_digest}"
                )

            stream.seek(0)
            try:
                with zipfile.ZipFile(stream, mode="r", allowZip64=True) as archive:
                    manifest = _manifest_from_zip(
                        archive,
                        archive_size_bytes=actual_size,
                        archive_sha256=actual_digest,
                        expected_file_count=expected_file_count,
                        expected_uncompressed_size_bytes=(
                            expected_uncompressed_size_bytes
                        ),
                        expected_directory_entry_count=(
                            expected_directory_entry_count
                        ),
                        require_encrypted=require_encrypted,
                    )
                    yield archive, manifest
            except zipfile.BadZipFile as exc:
                raise ArchiveValidationError(
                    f"archive is not a valid ZIP file: {source}: {exc}"
                ) from exc
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ArchiveValidationError(f"cannot read archive {source}: {exc}") from exc


def verify_archive(
    path: str | Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_file_count: int,
    expected_uncompressed_size_bytes: int,
    expected_directory_entry_count: int | None = None,
    require_encrypted: bool = False,
) -> ArchiveTreeManifest:
    """Verify a pinned ZIP and return its authenticated tree manifest."""

    with _validated_archive(
        path,
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        expected_file_count=expected_file_count,
        expected_uncompressed_size_bytes=expected_uncompressed_size_bytes,
        expected_directory_entry_count=expected_directory_entry_count,
        require_encrypted=require_encrypted,
    ) as (_, manifest):
        return manifest


def verify_exact_archive(path: str | Path) -> ArchiveTreeManifest:
    """Verify the byte identity and ZIP layout of the official archive."""

    return verify_archive(
        path,
        expected_size_bytes=EXACT_ARCHIVE_SIZE_BYTES,
        expected_sha256=EXACT_ARCHIVE_SHA256,
        expected_file_count=EXACT_ARCHIVE_FILE_COUNT,
        expected_uncompressed_size_bytes=EXACT_ARCHIVE_UNCOMPRESSED_SIZE_BYTES,
        expected_directory_entry_count=EXACT_ARCHIVE_DIRECTORY_ENTRY_COUNT,
        require_encrypted=True,
    )


def _crc32_file(path: Path) -> tuple[int, str]:
    checksum = 0
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_SIZE):
            checksum = zlib.crc32(chunk, checksum)
            size += len(chunk)
    return size, f"{checksum & 0xFFFFFFFF:08x}"


def _walk_extracted_tree(root: Path) -> tuple[set[str], dict[str, Path]]:
    directories: set[str] = set()
    files: dict[str, Path] = {}
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ArchiveValidationError(
                    f"extracted tree contains a symbolic link: {relative}"
                )
            if not path.is_dir():
                raise ArchiveValidationError(
                    f"extracted tree contains a special directory: {relative}"
                )
            directories.add(relative)
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise ArchiveValidationError(
                    f"extracted tree contains a link or special file: {relative}"
                )
            files[relative] = path
    return directories, files


def verify_extracted_tree(
    destination: str | Path,
    manifest: ArchiveTreeManifest,
) -> None:
    """Raise if an extracted tree differs from an archive-derived manifest."""

    root = Path(destination)
    if root.is_symlink() or not root.is_dir():
        raise ArchiveValidationError(
            f"extracted tree root is not a regular directory: {root}"
        )

    actual_directories, actual_files = _walk_extracted_tree(root)
    expected_directories = set(manifest.directories)
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        additional = sorted(actual_directories - expected_directories)
        raise ArchiveValidationError(
            "extracted directory manifest mismatch: "
            f"missing={missing}, additional={additional}"
        )

    expected_files = {member.path: member for member in manifest.files}
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        additional = sorted(set(actual_files) - set(expected_files))
        raise ArchiveValidationError(
            "extracted file manifest mismatch: "
            f"missing={missing}, additional={additional}"
        )

    for path in sorted(expected_files):
        member = expected_files[path]
        actual_size, actual_crc32 = _crc32_file(actual_files[path])
        if actual_size != member.size_bytes:
            raise ArchiveValidationError(
                f"extracted file size mismatch for {path!r}: "
                f"expected {member.size_bytes}, found {actual_size}"
            )
        if actual_crc32 != member.crc32:
            raise ArchiveValidationError(
                f"extracted file CRC32 mismatch for {path!r}: "
                f"expected {member.crc32}, found {actual_crc32}"
            )


def _password_bytes(password: str | bytes | None) -> bytes | None:
    if password is None:
        return None
    encoded = password.encode("utf-8") if isinstance(password, str) else password
    if not encoded:
        raise ValueError("password cannot be empty")
    return encoded


def _ensure_directory(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for component in relative.parts:
        current = current / component
        try:
            current.mkdir()
        except FileExistsError:
            if current.is_symlink() or not current.is_dir():
                raise ArchiveExtractionError(
                    f"cannot create archive directory over existing path: {current}"
                )
    return current


def _copy_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    *,
    password: bytes | None,
    expected: ArchiveMemberManifest,
) -> None:
    parent = _ensure_directory(destination, PurePosixPath(expected.path).parent)
    target = parent / PurePosixPath(expected.path).name
    checksum = 0
    size = 0
    try:
        with archive.open(info, mode="r", pwd=password) as source, target.open(
            "xb"
        ) as output:
            while chunk := source.read(_COPY_CHUNK_SIZE):
                output.write(chunk)
                checksum = zlib.crc32(chunk, checksum)
                size += len(chunk)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArchiveExtractionError(
            f"cannot extract archive member {expected.path!r}: {exc}"
        ) from exc

    actual_crc32 = f"{checksum & 0xFFFFFFFF:08x}"
    if size != expected.size_bytes or actual_crc32 != expected.crc32:
        raise ArchiveExtractionError(
            f"archive member failed post-write validation: {expected.path!r}"
        )


def extract_archive(
    path: str | Path,
    destination: str | Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_file_count: int,
    expected_uncompressed_size_bytes: int,
    password: str | bytes | None,
    expected_directory_entry_count: int | None = None,
    require_encrypted: bool = False,
) -> ArchiveTreeManifest:
    """Verify and safely extract a pinned ZIP into a new destination.

    ``destination`` is mandatory and must not name any existing filesystem
    object, including an empty directory or a broken symlink.  The directory
    is created atomically before extraction, so existing data is never
    overwritten.
    """

    target_root = Path(destination)
    if os.path.lexists(target_root):
        raise ArchiveExtractionError(
            f"refusing to overwrite existing extraction destination: {target_root}"
        )
    if not target_root.parent.is_dir():
        raise ArchiveExtractionError(
            f"extraction destination parent does not exist: {target_root.parent}"
        )
    encoded_password = _password_bytes(password)

    with _validated_archive(
        path,
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        expected_file_count=expected_file_count,
        expected_uncompressed_size_bytes=expected_uncompressed_size_bytes,
        expected_directory_entry_count=expected_directory_entry_count,
        require_encrypted=require_encrypted,
    ) as (archive, manifest):
        if any(member.encrypted for member in manifest.files) and encoded_password is None:
            raise ArchiveExtractionError(
                "archive contains encrypted files but no password was provided"
            )

        try:
            target_root.mkdir(mode=0o700, exist_ok=False)
        except OSError as exc:
            raise ArchiveExtractionError(
                f"cannot create extraction destination {target_root}: {exc}"
            ) from exc

        info_by_path: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            member_path, is_directory = _safe_member_path(info)
            if not is_directory:
                info_by_path[member_path] = info

        try:
            for relative in manifest.directories:
                _ensure_directory(target_root, PurePosixPath(relative))
            for member in manifest.files:
                _copy_member(
                    archive,
                    info_by_path[member.path],
                    target_root,
                    password=encoded_password,
                    expected=member,
                )
            verify_extracted_tree(target_root, manifest)
        except Exception:
            # ``target_root`` was created by this function after an exclusive
            # existence check, so cleanup cannot remove pre-existing user data.
            shutil.rmtree(target_root, ignore_errors=True)
            raise

    return manifest


def extract_exact_archive(
    path: str | Path,
    destination: str | Path,
    *,
    password: str | bytes,
) -> ArchiveTreeManifest:
    """Verify and extract the official encrypted benchmark archive."""

    return extract_archive(
        path,
        destination,
        expected_size_bytes=EXACT_ARCHIVE_SIZE_BYTES,
        expected_sha256=EXACT_ARCHIVE_SHA256,
        expected_file_count=EXACT_ARCHIVE_FILE_COUNT,
        expected_uncompressed_size_bytes=EXACT_ARCHIVE_UNCOMPRESSED_SIZE_BYTES,
        expected_directory_entry_count=EXACT_ARCHIVE_DIRECTORY_ENTRY_COUNT,
        password=password,
        require_encrypted=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Verify or safely extract the official archive outside Hydra runs."""

    parser = argparse.ArgumentParser(
        description=(
            "Verify or extract the pinned LID-Benchmarks exact archive. "
            "This is data preparation; experiment configuration remains Hydra/YAML-only."
        )
    )
    parser.add_argument("action", choices=("verify", "extract"))
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/benchmarks.zip"),
        help="path to the downloaded official benchmarks.zip",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/lid_benchmarks_exact"),
        help="new extraction root; it must not already exist",
    )
    parser.add_argument(
        "--password",
        default=EXACT_ARCHIVE_PASSWORD,
        help="public password published by the benchmark authors",
    )
    args = parser.parse_args(argv)
    if args.action == "verify":
        manifest = verify_exact_archive(args.archive)
    else:
        manifest = extract_exact_archive(
            args.archive, args.destination, password=args.password
        )
    print(
        f"verified {manifest.file_count} files, "
        f"{manifest.uncompressed_size_bytes} uncompressed bytes, "
        f"sha256={manifest.archive_sha256}"
    )


if __name__ == "__main__":
    main()
