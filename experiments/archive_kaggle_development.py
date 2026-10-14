"""Download only checksum-listed compact artifacts from a completed Kaggle run."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise RuntimeError("Archive destination must not already exist")
    api = KaggleApi()
    api.authenticate()
    owner, slug = args.kernel.split("/", 1)
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    if response.next_page_token:
        raise RuntimeError("Paginated output requires explicit pagination support")
    files = {item.file_name: item.url for item in response.files}

    def fetch(name):
        response = requests.get(files[f"results/{name}"], timeout=120)
        response.raise_for_status()
        return response.content

    inventory = fetch("artifact_checksums.sha256")
    names = {}
    for line in inventory.decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match:
            raise RuntimeError("Unsafe or malformed checksum inventory")
        digest, name = match.groups()
        if name in names or Path(name).suffix not in {".csv", ".json", ".png"}:
            raise RuntimeError("Unexpected compact artifact")
        if name == "error_analysis_per_profile.csv":
            raise RuntimeError("Profile-level output must not be archived")
        names[name] = digest
    manifest_bytes = fetch("run_manifest.json")
    manifest = json.loads(manifest_bytes)
    if manifest.get("git_commit") != args.commit:
        raise RuntimeError("Downloaded run does not match requested training commit")
    if manifest.get("run_mode") != "development" or manifest.get("reportable") is not False:
        raise RuntimeError("This downloader requires a non-reportable development run")
    destination.mkdir(parents=True)
    for name, expected in names.items():
        data = manifest_bytes if name == "run_manifest.json" else fetch(name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")
        (destination / name).write_bytes(data)
    (destination / "artifact_checksums.sha256").write_bytes(inventory)
    print(f"Archived {len(names)} verified compact files; no weights, logs or source data")


if __name__ == "__main__":
    main()
