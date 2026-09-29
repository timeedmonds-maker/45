#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/153 Safari/537.36"
HEADERS = {"User-Agent": UA, "Referer": "https://clips.nba.com/", "Accept": "*/*"}
CAMERAS = (
    "Broadcast",
    "In Arena",
    "Left Slash",
    "Right Slash",
    "Left Above Rim",
    "Right Above Rim",
)

def safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")

def resolve_options(game_id: str, event_num: int) -> dict[str, str]:
    page = f"https://clips.nba.com/?gameNo={game_id}&eventNum={event_num}&source=grs"
    r = requests.get(page, headers=HEADERS, timeout=30)
    r.raise_for_status()
    options = {}
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', r.text, flags=re.I | re.S):
        url = html.unescape(m.group(1).strip())
        label = re.sub(r"<[^>]+>", "", html.unescape(m.group(3))).strip()
        if ".m3u8" in url.lower() and "lrmedia.nba.com" in url.lower():
            options[label] = url
    return options

def signed_join(base: str, child: str) -> str:
    full = urljoin(base, child)
    b = urlsplit(base)
    f = urlsplit(full)
    if b.query and not f.query:
        full = urlunsplit((f.scheme, f.netloc, f.path, b.query, f.fragment))
    return full

def choose_variant(master: str, target_width: int = 960) -> tuple[str, dict]:
    r = requests.get(master, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if "#EXT-X-STREAM-INF" not in r.text:
        return master, {}
    lines = [x.strip() for x in r.text.splitlines() if x.strip()]
    variants = []
    for i, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF"):
            continue
        rs = re.search(r"RESOLUTION=(\d+)x(\d+)", line)
        bw = re.search(r"BANDWIDTH=(\d+)", line)
        child = None
        for q in lines[i + 1:i + 5]:
            if not q.startswith("#"):
                child = q
                break
        if child is None:
            continue
        w = int(rs.group(1)) if rs else 10**9
        h = int(rs.group(2)) if rs else 0
        variants.append((w, h, int(bw.group(1)) if bw else 0, signed_join(master, child)))
    if not variants:
        return master, {}
    under = [x for x in variants if x[0] <= target_width]
    chosen = max(under, key=lambda x: (x[0], x[2])) if under else min(variants, key=lambda x: x[0])
    return chosen[3], {"width": chosen[0], "height": chosen[1], "bandwidth": chosen[2]}

def probe(path: Path) -> dict:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(raw)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", default="0022500301")
    ap.add_argument("--event-num", type=int, default=489)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    options = resolve_options(args.game_id, args.event_num)
    missing = [c for c in CAMERAS if c not in options]
    if missing:
        raise RuntimeError(f"official NBA page missing required camera labels: {missing}")

    rows = []
    header = f"User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n"
    for i, label in enumerate(CAMERAS, start=1):
        url, variant = choose_variant(options[label], 960)
        out = args.out / f"{i:02d}_{safe(label)}_SOURCE.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-xerror", "-v", "error",
            "-rw_timeout", "30000000", "-headers", header,
            "-i", url, "-map", "0:v:0", "-map", "0:a:0?",
            "-c", "copy", "-movflags", "+faststart", str(out),
        ], check=True, timeout=240)
        q = probe(out)
        s = q["streams"][0]
        if int(s["width"]) != 960 or int(s["height"]) != 540:
            raise RuntimeError(f"{label}: expected 960x540 source, got {s['width']}x{s['height']}")
        rows.append({"label": label, "file": out.name, "variant": variant, "probe": q})
        print("SOURCE", label, variant, flush=True)

    (args.out / "manifest.json").write_text(json.dumps({
        "game_id": args.game_id,
        "event_num": args.event_num,
        "cameras": rows,
    }, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
