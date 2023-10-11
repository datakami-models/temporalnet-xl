import os
from pathlib import Path
import requests
import subprocess
import time

# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"

def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(baseurl: str, basedest: Path, files: list[str]):
    start = time.time()
    print("downloading to: ", basedest)
    for f in files:
        dest = basedest / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix in (".json", ".txt"):
                download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)


def download_all_weights(weights, to="."):
    for weight in weights:
        download_weights(weight["src"], Path(to) / weight["dest"], weight["files"])
