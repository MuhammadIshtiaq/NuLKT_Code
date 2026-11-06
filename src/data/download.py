"""Official Harvard Dataverse downloader for the Duolingo SLAM corpus.

The downloader deliberately discovers files from the live dataset metadata rather
than encoding track abbreviations as language semantics.  The authoritative file
description identifies ``data_en_es.tar.gz`` as the English-learning track.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
DATASET_DOI = "10.7910/DVN/8SWHNO"
DATAVERSE_HOST = "https://dataverse.harvard.edu"
METADATA_URL = (
    f"{DATAVERSE_HOST}/api/datasets/:persistentId/"
    f"?persistentId=doi:{DATASET_DOI}"
)
PREPROCESSING_VERSION = "slam-preprocess-v1"


def project_root() -> Path:
    """Return the configured project root, defaulting to this repository."""

    configured = os.environ.get("NULKT_ROOT") or os.environ.get("EIP_KTFORMER_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[2]
    )


@dataclass(frozen=True)
class RemoteFile:
    """Minimal immutable description of one Dataverse file."""

    file_id: int
    filename: str
    description: str
    size: int
    checksum_type: str
    checksum: str
    persistent_id: str
    restricted: bool

    @property
    def download_url(self) -> str:
        return f"{DATAVERSE_HOST}/api/access/datafile/{self.file_id}"

    @property
    def is_english_learning_track(self) -> bool:
        text = self.description.casefold()
        return "english learners" in text and "speak spanish" in text

    @property
    def is_documentation(self) -> bool:
        text = f"{self.filename} {self.description}".casefold()
        return any(word in text for word in ("starter", "readme", "documentation"))


@dataclass(frozen=True)
class DatasetMetadata:
    """Verified fields needed for acquisition and provenance."""

    title: str
    doi: str
    persistent_url: str
    host: str
    version: str
    version_state: str
    publication_date: str
    release_time: str
    guestbook_id: int | None
    authors: tuple[str, ...]
    terms_of_use: str
    files: tuple[RemoteFile, ...]


def _request_bytes(url: str, *, retries: int = 4, timeout: int = 60) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "NuLKT/1.0"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Official Harvard Dataverse request failed after {retries} attempts: {url}. "
        "The host may be temporarily unavailable; rerun the download stage later."
    ) from last_error


def _field_values(citation: dict[str, Any], name: str) -> list[Any]:
    for field in citation.get("fields", []):
        if field.get("typeName") == name:
            value = field.get("value", [])
            return value if isinstance(value, list) else [value]
    return []


def parse_metadata(payload: dict[str, Any]) -> DatasetMetadata:
    """Parse and validate a Dataverse dataset API response."""

    if payload.get("status") != "OK" or "data" not in payload:
        raise ValueError("Dataverse response is not a successful dataset response")
    data = payload["data"]
    version = data.get("latestVersion") or {}
    if version.get("versionState") != "RELEASED":
        raise ValueError("The latest Dataverse dataset version is not released")
    citation = version.get("metadataBlocks", {}).get("citation", {})
    titles = _field_values(citation, "title")
    author_compounds = _field_values(citation, "author")
    authors = tuple(
        compound.get("authorName", {}).get("value", "")
        for compound in author_compounds
        if compound.get("authorName", {}).get("value")
    )
    files: list[RemoteFile] = []
    for entry in version.get("files", []):
        file_data = entry.get("dataFile", {})
        checksum = file_data.get("checksum", {})
        if not file_data.get("id") or not file_data.get("filename"):
            continue
        files.append(
            RemoteFile(
                file_id=int(file_data["id"]),
                filename=str(file_data["filename"]),
                description=str(entry.get("description") or file_data.get("description") or ""),
                size=int(file_data.get("filesize", 0)),
                checksum_type=str(checksum.get("type", "")),
                checksum=str(checksum.get("value", "")),
                persistent_id=str(file_data.get("persistentId", "")),
                restricted=bool(entry.get("restricted", file_data.get("restricted", False))),
            )
        )
    if not files:
        raise ValueError("No files were returned by the Dataverse metadata API")
    doi = f"{data.get('authority')}/{data.get('identifier')}"
    if doi.casefold() != DATASET_DOI.casefold():
        raise ValueError(f"Unexpected dataset DOI in metadata: {doi}")
    return DatasetMetadata(
        title=str(titles[0]) if titles else "",
        doi=doi,
        persistent_url=str(data.get("persistentUrl", f"https://doi.org/{doi}")),
        host=DATAVERSE_HOST,
        version=f"{version.get('versionNumber', 0)}.{version.get('versionMinorNumber', 0)}",
        version_state=str(version.get("versionState", "")),
        publication_date=str(version.get("publicationDate", "")),
        release_time=str(version.get("releaseTime", "")),
        guestbook_id=int(version["guestbookId"]) if version.get("guestbookId") else None,
        authors=authors,
        terms_of_use=html.unescape(str(version.get("termsOfUse", ""))),
        files=tuple(files),
    )


def fetch_metadata(*, root: Path | None = None) -> tuple[DatasetMetadata, dict[str, Any]]:
    """Fetch official metadata and save the unmodified API response."""

    root = (root or project_root()).resolve()
    metadata_dir = root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw = _request_bytes(METADATA_URL)
    payload = json.loads(raw.decode("utf-8"))
    metadata = parse_metadata(payload)
    (metadata_dir / "dataverse_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Stable integration name used by the quality-control stage.
    (metadata_dir / "dataset_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata, payload


def fetch_guestbook(guestbook_id: int, *, root: Path) -> dict[str, Any]:
    """Retrieve and preserve the official required-download guestbook schema."""

    url = f"{DATAVERSE_HOST}/api/guestbooks/{guestbook_id}"
    payload = json.loads(_request_bytes(url).decode("utf-8"))
    if payload.get("status") != "OK":
        raise RuntimeError(f"Could not retrieve Dataverse guestbook {guestbook_id}")
    path = root / "data" / "metadata" / "dataverse_guestbook.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["data"]


def guestbook_response(
    schema: dict[str, Any],
    *,
    name: str | None = None,
    email: str | None = None,
    institution: str | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    """Build a schema-aware response for the public-download guestbook.

    Harvard Dataverse 6.10+ requires this response even for unrestricted files.
    Values may be supplied through ``NULKT_GUESTBOOK_*`` environment variables;
    the former ``EIP_KTFORMER_GUESTBOOK_*`` names remain supported.
    Non-identifying project-level defaults keep unattended runs
    possible while accurately describing the automated research client.
    """

    values = {
        "name": name
        or os.environ.get("NULKT_GUESTBOOK_NAME")
        or os.environ.get("EIP_KTFORMER_GUESTBOOK_NAME")
        or "NuLKT automated research pipeline",
        "email": email
        or os.environ.get("NULKT_GUESTBOOK_EMAIL")
        or os.environ.get("EIP_KTFORMER_GUESTBOOK_EMAIL")
        or "nulkt@example.org",
        "institution": institution
        or os.environ.get("NULKT_GUESTBOOK_INSTITUTION")
        or os.environ.get("EIP_KTFORMER_GUESTBOOK_INSTITUTION")
        or "Independent research",
        "position": position
        or os.environ.get("NULKT_GUESTBOOK_POSITION")
        or os.environ.get("EIP_KTFORMER_GUESTBOOK_POSITION")
        or "Research software",
        "answers": [],
    }
    required = {
        "name": bool(schema.get("nameRequired")),
        "email": bool(schema.get("emailRequired")),
        "institution": bool(schema.get("institutionRequired")),
        "position": bool(schema.get("positionRequired")),
    }
    missing = [key for key, needed in required.items() if needed and not values[key]]
    if missing:
        raise ValueError(f"Required Dataverse guestbook fields are missing: {missing}")
    for question in schema.get("customQuestions", []):
        if question.get("required"):
            raise RuntimeError(
                "The official dataset guestbook now has an unanswered required custom "
                f"question (id={question.get('id')}). Update the acquisition configuration."
            )
    return {"guestbookResponse": values}


def signed_download_url(remote: RemoteFile, response_payload: dict[str, Any]) -> str:
    """Submit the required guestbook response and return a short-lived official URL."""

    request = Request(
        remote.download_url,
        data=json.dumps(response_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 NuLKT/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Harvard Dataverse rejected the required guestbook response for "
            f"{remote.filename}: {detail}"
        ) from exc
    if payload.get("status") != "OK" or not payload.get("data", {}).get("signedUrl"):
        raise RuntimeError(f"No signed download URL was returned for {remote.filename}")
    return str(payload["data"]["signedUrl"])


def select_required_files(
    metadata: DatasetMetadata, *, all_tracks: bool = False
) -> tuple[RemoteFile, ...]:
    """Select data using authoritative descriptions, plus supplied documentation."""

    if all_tracks:
        selected = [f for f in metadata.files if f.filename.endswith(".tar.gz")]
    else:
        selected = [
            f
            for f in metadata.files
            if f.is_english_learning_track or f.is_documentation
        ]
    english = [f for f in selected if f.is_english_learning_track]
    if not english:
        descriptions = "; ".join(f"{f.filename}: {f.description}" for f in metadata.files)
        raise RuntimeError(
            "Official metadata did not identify an English-learning track. "
            f"Available files were: {descriptions}"
        )
    if any(f.restricted for f in selected):
        raise PermissionError("A required Dataverse file is unexpectedly restricted")
    return tuple(selected)


def compute_checksum(path: Path, algorithm: str = "MD5") -> str:
    """Compute a streaming file checksum."""

    normalized = algorithm.casefold().replace("-", "")
    try:
        digest = hashlib.new(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}") from exc
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_is_valid(path: Path, remote: RemoteFile) -> bool:
    """Return true only when size and advertised checksum match."""

    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    if remote.checksum:
        return compute_checksum(path, remote.checksum_type) == remote.checksum.casefold()
    return True


def download_file(
    remote: RemoteFile,
    destination: Path,
    *,
    retries: int = 4,
    timeout: int = 120,
    force: bool = False,
    guestbook_payload: dict[str, Any] | None = None,
) -> Path:
    """Download with checksum verification and best-effort HTTP range resumption."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force and file_is_valid(destination, remote):
        LOGGER.info("Using verified existing file: %s", destination)
        return destination
    partial = destination.with_name(destination.name + ".part")
    if force:
        partial.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            access_url = (
                signed_download_url(remote, guestbook_payload)
                if guestbook_payload is not None
                else remote.download_url
            )
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "NuLKT/1.0"}
            if 0 < offset < remote.size:
                headers["Range"] = f"bytes={offset}-"
            elif offset >= remote.size:
                partial.unlink(missing_ok=True)
                offset = 0
            request = Request(access_url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                append = offset > 0 and status == 206
                if offset > 0 and status != 206:
                    offset = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if partial.stat().st_size != remote.size:
                raise IOError(
                    f"Incomplete download for {remote.filename}: "
                    f"{partial.stat().st_size} of {remote.size} bytes"
                )
            if remote.checksum and compute_checksum(partial, remote.checksum_type) != remote.checksum.casefold():
                partial.unlink(missing_ok=True)
                raise IOError(f"Checksum mismatch for {remote.filename}")
            os.replace(partial, destination)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Failed to acquire {remote.filename} from official Harvard Dataverse; "
        "a partial .part file was retained for a later retry."
    ) from last_error


def _safe_members(archive: tarfile.TarFile, destination: Path) -> Iterable[tarfile.TarInfo]:
    base = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe archive path: {member.name}") from exc
        if member.issym() or member.islnk():
            raise RuntimeError(f"Archive links are not extracted: {member.name}")
        yield member


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Safely and idempotently extract a tar archive."""

    marker = destination / ".extraction_complete.json"
    checksum = compute_checksum(archive_path, "sha256")
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("archive_sha256") == checksum:
            return destination
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(destination, members=_safe_members(archive, destination))
    marker.write_text(
        json.dumps(
            {
                "archive": archive_path.name,
                "archive_sha256": checksum,
                "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def write_provenance(
    metadata: DatasetMetadata,
    selected: Iterable[RemoteFile],
    downloaded: dict[str, Path],
    *,
    root: Path,
) -> Path:
    """Write machine-readable and human-readable provenance records."""

    access_date = datetime.now(timezone.utc).date().isoformat()
    selected = tuple(selected)
    english = [f.filename for f in selected if f.is_english_learning_track]
    records = []
    for remote in selected:
        path = downloaded.get(remote.filename)
        records.append(
            {
                **asdict(remote),
                "local_path": str(path.relative_to(root)) if path else "not downloaded",
                "verified": bool(path and file_is_valid(path, remote)),
            }
        )
    provenance = {
        "dataset": asdict(metadata) | {"files": records},
        "access_date_utc": access_date,
        "english_learning_tracks_selected": english,
        "track_selection_rule": (
            "Dataverse file description states 'English learners (who already speak Spanish)'"
        ),
        "preprocessing_version": PREPROCESSING_VERSION,
        "license_note": "CC BY-NC 4.0; raw data are acquired from, not redistributed by, this project.",
    }
    metadata_path = root / "data" / "metadata" / "dataset_provenance.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    output = root / "results" / "publication" / "dataset_provenance.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| `{r['filename']}` | {r['description']} | {r['size']} | "
        f"{r['checksum_type']} `{r['checksum']}` | {r['verified']} |"
        for r in records
    )
    terms = metadata.terms_of_use.replace("\r", " ").replace("\n", " ")
    output.write_text(
        f"""# Dataset provenance

- **Dataset:** {metadata.title}
- **Persistent DOI:** [{metadata.doi}]({metadata.persistent_url})
- **Official host:** {metadata.host}
- **Released dataset version:** {metadata.version} ({metadata.version_state})
- **Release timestamp:** {metadata.release_time}
- **Access date (UTC):** {access_date}
- **Original dataset author(s):** {', '.join(metadata.authors)}
- **Terms:** {terms}
- **English-learning track selected:** {', '.join(f'`{name}`' for name in english)}
- **Selection evidence:** the official Dataverse description explicitly says “TRAIN, DEV, and TEST sets for English learners (who already speak Spanish).”
- **Preprocessing version:** `{PREPROCESSING_VERSION}`

The corpus is observational interaction data, not a randomized evaluation of explicit instruction. It can support predictive modeling and a transparent recommendation layer, but it does not identify causal instructional effects.

## Files

| File | Official description | Bytes | Checksum | Verified locally |
|---|---|---:|---|---|
{rows}

Raw data are downloaded directly from Harvard Dataverse and are not redistributed by this repository. Use is subject to CC BY-NC 4.0 and the official dataset terms.
""",
        encoding="utf-8",
    )
    return output


def run_download(
    *,
    root: Path | None = None,
    all_tracks: bool = False,
    metadata_only: bool = False,
    force: bool = False,
    guestbook_name: str | None = None,
    guestbook_email: str | None = None,
    guestbook_institution: str | None = None,
) -> dict[str, Path]:
    """Execute metadata discovery, verified download, extraction, and provenance."""

    root = (root or project_root()).resolve()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata, _ = fetch_metadata(root=root)
    selected = select_required_files(metadata, all_tracks=all_tracks)
    response_payload: dict[str, Any] | None = None
    if metadata.guestbook_id is not None and not metadata_only:
        schema = fetch_guestbook(metadata.guestbook_id, root=root)
        response_payload = guestbook_response(
            schema,
            name=guestbook_name,
            email=guestbook_email,
            institution=guestbook_institution,
        )
    downloaded: dict[str, Path] = {}
    if not metadata_only:
        for remote in selected:
            destination = raw_dir / remote.filename
            downloaded[remote.filename] = download_file(
                remote,
                destination,
                force=force,
                guestbook_payload=response_payload,
            )
            if remote.filename.endswith(".tar.gz"):
                extract_archive(destination, raw_dir / "extracted" / remote.filename[:-7])
    write_provenance(metadata, selected, downloaded, root=root)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--all-tracks", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--guestbook-name", default=None)
    parser.add_argument("--guestbook-email", default=None)
    parser.add_argument("--guestbook-institution", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = run_download(
        root=args.root,
        all_tracks=args.all_tracks,
        metadata_only=args.metadata_only,
        force=args.force,
        guestbook_name=args.guestbook_name,
        guestbook_email=args.guestbook_email,
        guestbook_institution=args.guestbook_institution,
    )
    for name, path in paths.items():
        LOGGER.info("Verified %s at %s", name, path)


if __name__ == "__main__":
    main()
