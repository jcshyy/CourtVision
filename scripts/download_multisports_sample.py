"""Read an official uncompressed TAR by HTTP ranges; download registered clips only.

Authentication uses the local Hugging Face credential through its intended API.
Signed redirect URLs remain in memory and are never written to reports/logs.
Selected files get local hashes; this does NOT claim a full archive checksum.
"""
import argparse
import hashlib
import json
import math
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_multisports import REVISION
from scripts.prepare_multisports_validation import sha

ARCHIVE_SIZE = 15545866240
URL = f"https://huggingface.co/datasets/MCG-NJU/SportsAction/resolve/{REVISION}/data/trainval/basketball.tar?download=true"


class RangeReader:
    def __init__(self):
        self.session = requests.Session()
        self.redirect = None

    def get(self, start, size):
        end = start + size - 1
        for attempt in range(4):
            headers = {"Range": f"bytes={start}-{end}"}
            # A signed CDN redirect can expire during hundreds of header
            # probes. A cache-buster is used only when re-authorizing against
            # the public Hugging Face resolve endpoint; credentials never enter
            # the URL or reports.
            url = self.redirect or f"{URL}&range_start={start}&attempt={attempt}&at={int(time.time())}"
            if self.redirect is None:
                token = get_token()
                if not token:
                    raise RuntimeError("Local Hugging Face login is required")
                headers["Authorization"] = "Bearer " + token
            try:
                response = self.session.get(url, headers=headers, timeout=(15, 45))
                if response.status_code in (401, 403) and self.redirect:
                    self.redirect = None
                    continue
                if response.status_code != 206:
                    raise RuntimeError(f"Range download returned HTTP {response.status_code}")
                expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
                if response.headers.get("Content-Range") != expected or len(response.content) != size:
                    raise RuntimeError("Server returned a different byte range")
                self.redirect = response.url
                return response.content
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError("Official download connection failed after retries") from None
                time.sleep(1 + attempt)
        raise RuntimeError("Could not authorize requested byte range")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "runs/multisports-independent-v1")
    args = parser.parse_args()
    directory = args.directory.resolve()
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["revision"] != REVISION:
        raise ValueError("Unexpected dataset revision")
    wanted = {v + ".mp4" for v in protocol["selected_video_ids"]}
    index_path = directory / "archive_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "revision": REVISION, "archive_size": ARCHIVE_SIZE, "next_offset": 0, "members": {},
    }
    reader = RangeReader()
    offset = index["next_offset"]
    while offset + 512 <= ARCHIVE_SIZE and not wanted.issubset(index["members"]):
        header = reader.get(offset, 512)
        if header == bytes(512):
            break
        info = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        if info.size < 0 or offset + 512 + info.size > ARCHIVE_SIZE:
            raise ValueError("Invalid archive header size")
        name = info.name.removeprefix("./")
        if info.isfile():
            index["members"][name] = {"offset": offset + 512, "size": info.size}
        offset += 512 + math.ceil(info.size / 512) * 512
        index["next_offset"] = offset
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        if len(index["members"]) % 20 == 0 or name in wanted:
            print(f"Indexed {len(index['members'])} clips; selected {len(wanted.intersection(index['members']))}/{len(wanted)}; archive position {offset / ARCHIVE_SIZE:.1%}", flush=True)
    if not wanted.issubset(index["members"]):
        raise ValueError("Registered clips not found in official archive")
    total = sum(index["members"][name]["size"] for name in wanted)
    print(f"Downloading {len(wanted)} registered clips, {total / 1e6:.1f} MB instead of the full 15.5 GB", flush=True)

    def fetch(name):
        destination = (directory / "clips" / name).resolve()
        if not destination.is_relative_to((directory / "clips").resolve()):
            raise ValueError("Unsafe clip path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        member = index["members"][name]
        partial = destination.with_suffix(".mp4.partial")
        client = RangeReader()
        if not destination.exists():
            completed = partial.stat().st_size if partial.exists() else 0
            if completed > member["size"]:
                raise ValueError("Invalid partial clip length")
            with partial.open("ab") as output:
                while completed < member["size"]:
                    size = min(2 * 1024 * 1024, member["size"] - completed)
                    block = client.get(member["offset"] + completed, size)
                    output.write(block)
                    output.flush()
                    completed += len(block)
            partial.rename(destination)
        if destination.stat().st_size != member["size"]:
            raise ValueError("Incorrect clip size")
        result = {"video_id": name[:-4], "path": str(destination), "size": member["size"],
                  "archive_offset": member["offset"], "sha256": sha(destination)}
        print(f"Downloaded {name}: {member['size'] / 1e6:.1f} MB", flush=True)
        return result

    (directory / "clips").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch, sorted(wanted)))
    (directory / "range_downloads.json").write_text(json.dumps({
        "revision": REVISION, "official_archive_lfs_sha256": protocol["archive_sha256"],
        "full_archive_checksum_verified": False,
        "transport": "Authorized HTTPS byte ranges; TAR header checksums and Content-Range validated",
        "clips": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
