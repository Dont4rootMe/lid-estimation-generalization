from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import stat
import warnings
import zipfile

import pytest
import yaml

from datasets.archive import (
    ArchiveExtractionError,
    ArchiveValidationError,
    EXACT_ARCHIVE_DIRECTORY_ENTRY_COUNT,
    EXACT_ARCHIVE_FILE_COUNT,
    EXACT_ARCHIVE_SHA256,
    EXACT_ARCHIVE_SIZE_BYTES,
    EXACT_ARCHIVE_UNCOMPRESSED_SIZE_BYTES,
    extract_archive,
    verify_archive,
    verify_extracted_tree,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# A tiny traditional ZipCrypto archive created with ``zip -P s3cret``.  Python's
# stdlib can decrypt ZipCrypto but cannot create encrypted fixtures itself.
ENCRYPTED_ZIP = base64.b64decode(
    "UEsDBAoACQAAAI+KF13Nd37VGwAAAA8AAAALABwAcGF5bG9hZC50eHRVVAkAA64B"
    "i2quAYtqdXgLAAEE9QEAAAQAAAAAPQOYy/xRBajR/qOf1ejtPWsIIyXN2fy2JcPf"
    "UEsHCM13ftUbAAAADwAAAFBLAQIeAwoACQAAAI+KF13Nd37VGwAAAA8AAAALABgA"
    "AAAAAAEAAACkgQAAAABwYXlsb2FkLnR4dFVUBQADrgGLanV4CwABBPUBAAAEAAAA"
    "AFBLBQYAAAAAAQABAFEAAABwAAAAAAA="
)


def _expectations(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        directories = [info for info in archive.infolist() if info.is_dir()]
    return {
        "expected_size_bytes": len(payload),
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_file_count": len(files),
        "expected_uncompressed_size_bytes": sum(info.file_size for info in files),
        "expected_directory_entry_count": len(directories),
    }


def _write_valid_zip(path: Path) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("benchmarks/", b"")
        archive.writestr("benchmarks/tiny/", b"")
        archive.writestr("benchmarks/tiny/dataset.npy", b"dataset")
        archive.writestr("benchmarks/tiny/lid.npy", b"lid")


def test_verifies_extracts_and_revalidates_complete_tree(tmp_path: Path) -> None:
    archive_path = tmp_path / "tiny.zip"
    _write_valid_zip(archive_path)
    expected = _expectations(archive_path)

    manifest = verify_archive(archive_path, **expected)

    assert manifest.file_count == 2
    assert manifest.directory_entry_count == 2
    assert manifest.directories == ("benchmarks", "benchmarks/tiny")
    assert [member.path for member in manifest.files] == [
        "benchmarks/tiny/dataset.npy",
        "benchmarks/tiny/lid.npy",
    ]

    destination = tmp_path / "extracted"
    extracted_manifest = extract_archive(
        archive_path,
        destination,
        password=None,
        **expected,
    )
    assert extracted_manifest == manifest
    assert (destination / "benchmarks/tiny/dataset.npy").read_bytes() == b"dataset"
    verify_extracted_tree(destination, manifest)

    with pytest.raises(ArchiveExtractionError, match="refusing to overwrite"):
        extract_archive(
            archive_path,
            destination,
            password=None,
            **expected,
        )

    (destination / "benchmarks/tiny/dataset.npy").write_bytes(b"changed")
    with pytest.raises(ArchiveValidationError, match="CRC32 mismatch"):
        verify_extracted_tree(destination, manifest)


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.txt",
        "nested/../../escape.txt",
        "/absolute.txt",
        "C:/windows-drive.txt",
        "nested\\windows-separator.txt",
        "nested//non-canonical.txt",
        "nested/./non-canonical.txt",
    ],
)
def test_rejects_unsafe_archive_paths(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(member_name, b"unsafe")

    with pytest.raises(ArchiveValidationError, match="archive member"):
        verify_archive(archive_path, **_expectations(archive_path))


def test_rejects_duplicate_and_normalized_collision_paths(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, mode="w") as archive:
            archive.writestr("same.txt", b"first")
            archive.writestr("same.txt", b"second")
    with pytest.raises(ArchiveValidationError, match="repeats member path"):
        verify_archive(duplicate, **_expectations(duplicate))

    collision = tmp_path / "collision.zip"
    with zipfile.ZipFile(collision, mode="w") as archive:
        archive.writestr("Result.txt", b"first")
        archive.writestr("result.txt", b"second")
    with pytest.raises(ArchiveValidationError, match="collide"):
        verify_archive(collision, **_expectations(collision))

    implicit_directory_collision = tmp_path / "implicit-directory-collision.zip"
    with zipfile.ZipFile(implicit_directory_collision, mode="w") as archive:
        archive.writestr("Output/one.txt", b"first")
        archive.writestr("output/two.txt", b"second")
    with pytest.raises(ArchiveValidationError, match="collide"):
        verify_archive(
            implicit_directory_collision,
            **_expectations(implicit_directory_collision),
        )

    ancestor_conflict = tmp_path / "ancestor-conflict.zip"
    with zipfile.ZipFile(ancestor_conflict, mode="w") as archive:
        archive.writestr("parent/child.txt", b"child")
        archive.writestr("parent", b"file")
    with pytest.raises(ArchiveValidationError, match="file/directory path conflict"):
        verify_archive(ancestor_conflict, **_expectations(ancestor_conflict))


def test_rejects_symlink_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(link, "target")

    with pytest.raises(ArchiveValidationError, match="link or special file"):
        verify_archive(archive_path, **_expectations(archive_path))


def test_rejects_size_hash_and_encryption_contract_mismatches(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "tiny.zip"
    _write_valid_zip(archive_path)
    expected = _expectations(archive_path)

    with pytest.raises(ArchiveValidationError, match="size mismatch"):
        verify_archive(
            archive_path,
            **{**expected, "expected_size_bytes": int(expected["expected_size_bytes"]) + 1},
        )
    with pytest.raises(ArchiveValidationError, match="SHA-256 mismatch"):
        verify_archive(
            archive_path,
            **{**expected, "expected_sha256": "0" * 64},
        )
    with pytest.raises(ArchiveValidationError, match="not password protected"):
        verify_archive(archive_path, require_encrypted=True, **expected)


def test_extracts_password_protected_synthetic_zip_without_overwrite(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "encrypted.zip"
    archive_path.write_bytes(ENCRYPTED_ZIP)
    expected = _expectations(archive_path)
    manifest = verify_archive(
        archive_path,
        require_encrypted=True,
        **expected,
    )
    assert manifest.files[0].encrypted

    missing_password_destination = tmp_path / "missing-password"
    with pytest.raises(ArchiveExtractionError, match="no password"):
        extract_archive(
            archive_path,
            missing_password_destination,
            password=None,
            require_encrypted=True,
            **expected,
        )
    assert not missing_password_destination.exists()

    wrong_password_destination = tmp_path / "wrong-password"
    with pytest.raises(ArchiveExtractionError, match="cannot extract"):
        extract_archive(
            archive_path,
            wrong_password_destination,
            password="wrong",
            require_encrypted=True,
            **expected,
        )
    assert not wrong_password_destination.exists()

    destination = tmp_path / "correct-password"
    extract_archive(
        archive_path,
        destination,
        password="s3cret",
        require_encrypted=True,
        **expected,
    )
    assert (destination / "payload.txt").read_bytes() == b"secret payload\n"


def test_registry_integrity_metadata_matches_archive_constants() -> None:
    registry_path = REPOSITORY_ROOT / "lid_benchmarks/DATA.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    upstream = registry["upstreams"]["lid_benchmarks"]
    integrity = upstream["official_archive_integrity"]

    assert integrity["size_bytes"] == EXACT_ARCHIVE_SIZE_BYTES
    assert integrity["sha256"] == EXACT_ARCHIVE_SHA256
    assert integrity["file_count"] == EXACT_ARCHIVE_FILE_COUNT
    assert (
        integrity["directory_entry_count"]
        == EXACT_ARCHIVE_DIRECTORY_ENTRY_COUNT
    )
    assert (
        integrity["uncompressed_size_bytes"]
        == EXACT_ARCHIVE_UNCOMPRESSED_SIZE_BYTES
    )
    assert integrity["crc32_verified"] is True
    assert upstream["official_archive_direct_url"].startswith(
        "https://drive.usercontent.google.com/download?"
    )
