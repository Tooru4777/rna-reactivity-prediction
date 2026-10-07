"""Submit an exact-commit smoke run with an explicit P100 machine shape."""

import argparse
import json
from pathlib import Path
import re
import tempfile

from kaggle.api import kaggle_api_extended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be an exact lowercase 40-character SHA")

    source = Path(__file__).resolve().parent
    metadata = json.loads((source / "kernel-metadata.json").read_text())
    runner = (source / metadata["code_file"]).read_text()
    placeholder = "__PIN_EXACT_COMMIT_BEFORE_SUBMISSION__"
    if runner.count(placeholder) != 1:
        raise RuntimeError("Runner must contain exactly one commit placeholder")

    # The installed CLI only maps enable_gpu, although its generated SDK
    # supports machine_shape. Set that documented field on every save request.
    original_request = kaggle_api_extended.ApiSaveKernelRequest

    def p100_request():
        request = original_request()
        request.machine_shape = "NvidiaTeslaP100"
        return request

    kaggle_api_extended.ApiSaveKernelRequest = p100_request
    try:
        api = kaggle_api_extended.KaggleApi()
        api.authenticate()
        with tempfile.TemporaryDirectory(prefix="rna-p100-submit-") as directory:
            staging = Path(directory)
            (staging / "kernel-metadata.json").write_text(json.dumps(metadata))
            (staging / metadata["code_file"]).write_text(
                runner.replace(placeholder, args.commit)
            )
            print(f"Submitting {args.commit} with NvidiaTeslaP100", flush=True)
            response = api.kernels_push(str(staging))
            print(response)
    finally:
        kaggle_api_extended.ApiSaveKernelRequest = original_request


if __name__ == "__main__":
    main()
