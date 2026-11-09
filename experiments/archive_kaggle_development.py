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
    if api.kernels_status(args.kernel).to_dict().get("status") != "COMPLETE":
        raise RuntimeError("Only COMPLETE kernel outputs may be archived")
    owner, slug = args.kernel.split("/", 1)
    files = {}
    seen_tokens = set()
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.page_size = 20
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files:
                if item.file_name in files:
                    raise RuntimeError("Duplicate output file across pages")
                files[item.file_name] = item.url
            token = response.next_page_token
            if not token:
                break
            if token in seen_tokens:
                raise RuntimeError("Repeated output pagination token")
            seen_tokens.add(token)
            request.page_token = token

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
    if "run_manifest.json" not in names:
        raise RuntimeError("Checksum inventory must include the run manifest")
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
