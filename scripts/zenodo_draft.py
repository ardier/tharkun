#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CISPA Helmholtz Center for Information Security
# SPDX-License-Identifier: EUPL-1.2
"""Upload a validated release as a new Zenodo draft; publication stays manual."""

import argparse
import datetime
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


API = "https://zenodo.org/api"
RECORD = "23495266"
CONCEPT = "15844050"
REPOSITORY = "ardier/tharkun"


class ZenodoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.netloc != "zenodo.org":
            raise ValueError("Refusing to forward a Zenodo request to another origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request(method, url, *, token=None, payload=None, upload=None):
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https" or parsed.netloc != "zenodo.org"
        or not parsed.path.startswith("/api/")
    ):
        raise ValueError("Refusing an unexpected Zenodo API URL")
    if "/actions/" in parsed.path and (
        method != "POST" or not parsed.path.endswith("/actions/newversion")
    ):
        raise ValueError("Only the new-version draft action is allowed")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if upload is not None:
        data = upload
        headers["Content-Type"] = "application/zip"
        headers["Content-Length"] = str(os.fstat(upload.fileno()).st_size)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.build_opener(ZenodoRedirects).open(req, timeout=180) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1500]
        raise RuntimeError(f"Zenodo {method} {parsed.path}: HTTP {exc.code}: {detail}") from None
    return json.loads(body) if body else None


def check_archive(archive, expected_sha256, tag):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha256.update(chunk)
            md5.update(chunk)
    if sha256.hexdigest() != expected_sha256:
        raise ValueError("The downloaded ZIP does not match the validated SHA256")
    ref = f"refs/tags/{tag}"
    commit = subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"], text=True).strip()
    source_tar = subprocess.check_output(["git", "archive", ref, "src/tharkun"])
    with tarfile.open(fileobj=io.BytesIO(source_tar)) as source, zipfile.ZipFile(archive) as bundle:
        version = bundle.read("tharkun/VERSION").decode().split()[0]
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", version):
            raise ValueError("Unrecognized release VERSION")
        source_files = {
            member.name.removeprefix("src/tharkun/"): source.extractfile(member).read()
            for member in source.getmembers()
            if member.isfile() and "__pycache__" not in Path(member.name).parts
            and not member.name.endswith(".pyc")
        }
        package_files = {
            name.removeprefix("tharkun/lib/tharkun/"): bundle.read(name)
            for name in bundle.namelist()
            if name.startswith("tharkun/lib/tharkun/") and not name.endswith("/")
            and "__pycache__" not in Path(name).parts and not name.endswith(".pyc")
        }
        if not source_files or package_files != source_files:
            raise ValueError("The ZIP package files differ from the public release tag")
    return version, md5.hexdigest(), commit


def matches(files, md5, size):
    return len(files) == 1 and (
        files[0].get("filename", files[0].get("key")) == "tharkun.zip"
    ) and (
        files[0].get("checksum", "").removeprefix("md5:") == md5
        and int(files[0].get("filesize", files[0].get("size", -1))) == size
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.tag):
        parser.error("Invalid release tag")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", args.sha256):
        parser.error("SHA256 must contain 64 hexadecimal characters")
    expected_sha256 = args.sha256.lower()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        parser.error("Set the ZENODO_TOKEN repository secret before running this workflow")
    version, md5, commit = check_archive(args.archive, expected_sha256, args.tag)
    size = args.archive.stat().st_size
    release_url = f"https://github.com/{REPOSITORY}/releases/tag/{args.tag}"
    marker = f"GitHub release: {release_url}\nSHA256: {expected_sha256}"
    record = request("GET", f"{API}/records/{RECORD}/versions/latest")
    if str(record.get("conceptrecid")) != CONCEPT:
        raise ValueError("The record is outside the existing Tharkun DOI series")
    if record["metadata"].get("version") == version:
        if not matches(record.get("files", []), md5, size):
            raise ValueError("This version is already published with different bytes; bump the version")
        print(f"Already published: https://doi.org/{record['metadata']['doi']}")
        return
    previous_version = record["metadata"].get("version", "")
    previous_numbers = re.match(r"(\d+)\.(\d+)\.(\d+)", previous_version)
    new_numbers = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not previous_numbers or tuple(map(int, new_numbers.groups())) <= tuple(
        map(int, previous_numbers.groups())
    ):
        raise ValueError("A new release must increase major.minor.patch; older tags are not republished")
    deposit_url = f"{API}/deposit/depositions/{record['id']}"
    deposit = request("GET", deposit_url, token=token)
    draft_url = deposit.get("links", {}).get("latest_draft", deposit_url)
    if draft_url.rstrip("/") == deposit_url:
        deposit = request("POST", f"{deposit_url}/actions/newversion", token=token)
        draft_url = deposit["links"]["latest_draft"]
        draft = request("GET", draft_url, token=token)
    else:
        draft = request("GET", draft_url, token=token)
        if marker not in draft.get("metadata", {}).get("notes", ""):
            raise ValueError(f"Another Zenodo draft already exists; review https://zenodo.org/uploads/{draft['id']}")
    if (
        draft.get("submitted") is not False or draft.get("state") != "unsubmitted"
        or str(draft["id"]) == str(record["id"])
    ):
        raise ValueError("Refusing to modify a published record or an invalid draft")
    if str(draft.get("conceptrecid")) != CONCEPT:
        raise ValueError("The draft is outside the existing Tharkun DOI series")
    files = draft.get("files", [])
    if any(file.get("filename", file.get("key")) != "tharkun.zip" for file in files):
        raise ValueError("Unexpected draft attachment; review it before replacing any files")
    draft_link = f"https://zenodo.org/uploads/{draft['id']}"
    print(f"Draft for manual review: {draft_link}", flush=True)
    metadata = dict(draft["metadata"])
    metadata.pop("doi", None)
    metadata["version"] = version
    metadata["publication_date"] = datetime.date.today().isoformat()
    metadata["prereserve_doi"] = True
    metadata["description"] = re.sub(
        r"(This archive contains version )[^<\s]+",
        lambda match: match.group(1) + html.escape(version), metadata["description"],
    )
    if marker not in metadata.get("notes", ""):
        metadata["notes"] = metadata.get("notes", "").rstrip() + "\n\n" + marker
    draft = request("PUT", draft_url, token=token, payload={"metadata": metadata})
    files = draft.get("files", [])
    if not matches(files, md5, size):
        for file in files:
            request("DELETE", f"{draft_url}/files/{file['id']}", token=token)
        with args.archive.open("rb") as stream:
            uploaded = request(
                "PUT", f"{draft['links']['bucket']}/tharkun.zip", token=token, upload=stream
            )
        if (
            uploaded.get("checksum", "").removeprefix("md5:") != md5
            or uploaded.get("size") != size
        ):
            raise ValueError(f"Uploaded file checksum/size mismatch; review {draft_link}")
    draft = request("GET", draft_url, token=token)
    if not matches(draft.get("files", []), md5, size):
        raise ValueError(f"Draft attachment verification failed; review {draft_link}")
    doi = draft["metadata"].get("prereserve_doi", {}).get("doi")
    if not doi:
        raise ValueError(f"No reserved DOI was returned; review {draft_link}")
    summary = (
        f"## Tharkun {version}: Zenodo draft ready\n\n"
        f"- Review and publish manually: {draft_link}\n"
        f"- Reserved version DOI: `{doi}` (registered when you publish)\n"
        f"- Concept DOI: `10.5281/zenodo.{CONCEPT}`\n"
        f"- Public source: https://github.com/{REPOSITORY}/tree/{commit}\n"
        f"- SHA256: `{expected_sha256}`\n"
        f"- ZIP size: {size} bytes\n\n"
        "After publishing, use the version DOI in FM-Tools and rerun its required CI.\n"
    )
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(summary)


if __name__ == "__main__":
    main()
