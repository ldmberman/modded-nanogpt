"""Fetch the small-model checkpoint for the upscaling benchmark.

    python -m records.track_4_upscaling.utils.download_weights small_L1.pt
"""
import os
import sys

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("UPSCALE_SMALL_REPO", "ldmberman/track4-upscaling-small-models")


def get(fname: str, repo_id: str = REPO_ID) -> str:
    local_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    target = os.path.join(local_dir, fname)
    if not os.path.exists(target):
        hf_hub_download(repo_id=repo_id, filename=fname, repo_type="model", local_dir=local_dir)
    return target


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "small_L1.pt"
    print(get(fname))
