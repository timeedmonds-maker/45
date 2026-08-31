#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import html as htmlmod
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PBP_CSV_URL = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.csv"
UA = "adams-ft-miss-oreb-native-video/2026-08-31"
ADAMS_ID = "203500"
TEAM = "HOU"
MAX_PART_BYTES = 95_000_000
OUT = Path("outputs/adams_ft_miss_oreb_all_angles_2025_26")

PAGE_HEADERS = {
    "User-Agent": UA,
    "Referer": "https://clips.nba.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
ACTOR_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")

# Fingerprints of the known NBA "video not available" placeholder.
PLACEHOLDER_AHASH = [
    "0000180039fc39fc38c839fc39fc10000000",
    "0000180019fc39fc38c819fc39fc18000000",
    "0000100011fc11fc18ec11fc19fc10000000",
]


def scalar(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def integer(v):
    try:
        if pd.isna(v) or v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def actor(v) -> tuple[str, str]:
    s = scalar(v)
    if not s:
        return "", ""
    m = ACTOR_RE.match(s)
    return (m.group(1), m.group(2).strip()) if m else ("", s)


def norm_gid(v) -> str:
    s = scalar(v)
    if not s:
        return ""
    try:
        s = str(int(float(s)))
    except Exception:
        s = re.sub(r"\D", "", s)
    return s.zfill(10)


def event_col(columns: list[str]) -> str:
    for c in ("event_num", "event_id", "eventnum", "event_no", "event_number", "number_event"):
        if c in columns:
            return c
    raise RuntimeError(f"No event identifier column found. Columns={columns}")


def load_pbp() -> pd.DataFrame:
    # Deliberately use the upstream CSV here. This avoids any binary/gzip/RDS transport layer.
    # The exact same 2025-26 PBP source is published by ramirobentes as data.csv.
    r = requests.get(PBP_CSV_URL, headers={"User-Agent": UA}, timeout=180)
    r.raise_for_status()
    print(f"PBP_HTTP_BYTES={len(r.content)}", flush=True)
    d = pd.read_csv(StringIO(r.text), low_memory=False)
    print(f"PBP_ROWS={len(d)} COLS={len(d.columns)}", flush=True)
    return d


def find_events(pbp: pd.DataFrame) -> list[dict]:
    evc = event_col(list(pbp.columns))
    required = {"game_id", "period", "msg_type", "team_abb", "player1_name", "description", evc}
    missing = sorted(required - set(pbp.columns))
    if missing:
        raise RuntimeError(f"Missing required PBP columns: {missing}")

    q = pbp.copy().reset_index(drop=False).rename(columns={"index": "source_row"})
    q["_gid"] = q.game_id.map(norm_gid)
    q["_event"] = pd.to_numeric(q[evc], errors="coerce")
    q["_period"] = pd.to_numeric(q.period, errors="coerce")
    q["_msg"] = pd.to_numeric(q.msg_type, errors="coerce")
    q["_team"] = q.team_abb.fillna("").astype(str).str.strip().str.upper()
    q["_desc"] = q.description.fillna("").astype(str)

    # NBA regular-season game IDs begin 002.
    q = q[q._gid.str.startswith("002")].copy()
    q = q.sort_values(["_gid", "_event", "source_row"], kind="stable").reset_index(drop=True)

    rows: list[dict] = []
    for i in range(len(q) - 1):
        ft = q.iloc[i]
        rb = q.iloc[i + 1]

        if integer(ft._msg) != 3 or "MISS" not in ft._desc.upper() or ft._team != TEAM:
            continue
        shooter_id, shooter_name = actor(ft.player1_name)
        if shooter_id == ADAMS_ID:
            continue

        # Strict definition of "immediately after": the very next ordered PBP row.
        if rb._gid != ft._gid or integer(rb._period) != integer(ft._period):
            continue
        if integer(rb._msg) != 4 or rb._team != TEAM:
            continue
        rebounder_id, rebounder_name = actor(rb.player1_name)
        if rebounder_id != ADAMS_ID:
            continue

        rec = {
            "rank": len(rows) + 1,
            "season": "2025-26",
            "game_id": rb._gid,
            "period": integer(rb._period),
            "clock": scalar(rb.get("clock", "")),
            "ft_event_id": integer(ft._event),
            "rebound_event_id": integer(rb._event),
            "ft_shooter_id": shooter_id,
            "ft_shooter": shooter_name,
            "rebounder_id": rebounder_id,
            "rebounder": rebounder_name,
            "team": TEAM,
            "ft_description": scalar(ft.description),
            "rebound_description": scalar(rb.description),
            "source_row_ft": integer(ft.source_row),
            "source_row_rebound": integer(rb.source_row),
            "definition": "HOU teammate missed FT and the immediately next ordered PBP row is an HOU rebound credited to Steven Adams",
            "pbp_source": PBP_CSV_URL,
        }
        for c in ("team_home", "team_away", "date", "game_date"):
            if c in q.columns:
                rec[c] = scalar(rb.get(c, ""))
        rows.append(rec)
    return rows


def parse_clips_page(game_id: str, event_id: int) -> dict:
    url = f"https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs"
    r = requests.get(url, headers=PAGE_HEADERS, timeout=45)
    r.raise_for_status()
    text = r.text
    title_m = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
    title = htmlmod.unescape(title_m.group(1).strip()) if title_m else ""

    angles = []
    seen = set()
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', text, flags=re.I | re.S):
        hls = htmlmod.unescape(m.group(1).strip())
        if ".m3u8" not in hls.lower() or "lrmedia.nba.com" not in hls.lower() or hls in seen:
            continue
        seen.add(hls)
        label = re.sub(r"<[^>]+>", "", htmlmod.unescape(m.group(3))).strip() or "angle"
        angles.append({"hls": hls, "label": label, "selected": "selected" in m.group(2).lower()})

    return {"page_url": url, "title": title, "angles": angles}


def safe_name(s: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()).strip("_")
    return x[:72] or "angle"


def download_hls_native(hls: str, dest: Path) -> None:
    headers = f"User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n"
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-rw_timeout", "30000000",
        "-headers", headers, "-i", hls,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c", "copy", "-movflags", "+faststart", str(dest),
    ]
    print("+", " ".join(cmd[:8] + ["<SIGNED_HLS>"] + cmd[9:]), flush=True)
    subprocess.run(cmd, check=True)


def ahash_frame(path: Path, t: float) -> str:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path),
         "-vf", "scale=16:9,format=gray", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    if p.returncode or len(p.stdout) != 144:
        return ""
    vals = list(p.stdout)
    avg = sum(vals) / len(vals)
    bits = "".join("1" if x >= avg else "0" for x in vals)
    return f"{int(bits, 2):036x}"


def hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    return (int(a, 16) ^ int(b, 16)).bit_count()


def probe_video(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True,
    )
    if p.returncode:
        return {"ok": False, "reason": "ffprobe_failed"}
    try:
        j = json.loads(p.stdout)
        s = j["streams"][0]
        duration = float((j.get("format") or {}).get("duration") or 0)
    except Exception:
        return {"ok": False, "reason": "ffprobe_parse"}

    times = [min(max(duration * f, 0.25), max(duration - 0.25, 0.25)) for f in (0.25, 0.5, 0.75)]
    hashes = [ahash_frame(path, t) for t in times]
    distances = [hamming_hex(a, b) for a, b in zip(hashes, PLACEHOLDER_AHASH)]
    placeholder_like = all(d <= 12 for d in distances)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ok = duration >= 2.0 and all(hashes) and not placeholder_like
    return {
        "ok": ok,
        "reason": None if ok else ("nba_video_not_available_placeholder" if placeholder_like else "invalid_media"),
        "duration": duration,
        "codec": s.get("codec_name"),
        "width": s.get("width"),
        "height": s.get("height"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "sha256": sha,
        "bytes": path.stat().st_size,
        "visual_hashes": hashes,
        "placeholder_hamming": distances,
    }


def build_videos(events: list[dict]) -> tuple[list[dict], list[Path]]:
    clipdir = OUT / "native_angles"
    clipdir.mkdir(parents=True, exist_ok=True)
    qa: list[dict] = []
    kept: list[Path] = []

    # Within each sequence, query BOTH the missed-FT event and rebound event pages,
    # because the NBA sometimes associates the usable replay package with one or the other.
    for ev in events:
        gid = ev["game_id"]
        sequence_seen_sha: set[str] = set()
        candidate_pages = [("ft", int(ev["ft_event_id"])), ("rebound", int(ev["rebound_event_id"]))]
        pages_with_angles = 0

        for anchor_kind, anchor_event in candidate_pages:
            try:
                page = parse_clips_page(gid, anchor_event)
            except Exception as exc:
                qa.append({
                    "rank": ev["rank"], "game_id": gid, "anchor_kind": anchor_kind,
                    "anchor_event_id": anchor_event, "status": "page_failed", "error": repr(exc),
                })
                continue

            if not page["angles"]:
                qa.append({
                    "rank": ev["rank"], "game_id": gid, "anchor_kind": anchor_kind,
                    "anchor_event_id": anchor_event, "clips_page_url": page["page_url"],
                    "status": "no_angles_on_page",
                })
                continue

            pages_with_angles += 1
            for ai, angle in enumerate(page["angles"], 1):
                rec = {
                    "rank": ev["rank"], "game_id": gid, "period": ev["period"], "clock": ev["clock"],
                    "ft_shooter": ev["ft_shooter"], "anchor_kind": anchor_kind,
                    "anchor_event_id": anchor_event, "clips_page_url": page["page_url"],
                    "clips_page_title": page["title"], "angle_index": ai,
                    "angle_count_on_page": len(page["angles"]), "angle_label": angle["label"],
                    "selected_on_page": angle["selected"], "status": "failed",
                }
                path = clipdir / (
                    f"{int(ev['rank']):02d}_{gid}_ft{int(ev['ft_event_id'])}_rb{int(ev['rebound_event_id'])}_"
                    f"{anchor_kind}_a{ai:02d}_{safe_name(angle['label'])}.mp4"
                )
                try:
                    download_hls_native(angle["hls"], path)
                    probe = probe_video(path)
                    rec["probe"] = probe
                    if not probe["ok"]:
                        raise RuntimeError(probe["reason"])
                    sha = probe["sha256"]
                    if sha in sequence_seen_sha:
                        rec["status"] = "duplicate_excluded"
                        rec["duplicate_reason"] = "identical native media already captured for this FT/rebound sequence"
                        path.unlink(missing_ok=True)
                    else:
                        sequence_seen_sha.add(sha)
                        rec["status"] = "ok"
                        rec["native_path"] = str(path)
                        kept.append(path)
                except Exception as exc:
                    rec["error"] = repr(exc)
                    path.unlink(missing_ok=True)
                qa.append(rec)

        if pages_with_angles == 0:
            print(f"WARNING no clip angles found for qualifying event rank={ev['rank']} gid={gid}", flush=True)

    return qa, kept


def write_event_manifest(events: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "event_manifest.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    if events:
        fields = list(dict.fromkeys(k for r in events for k in r))
        with (OUT / "event_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(events)


def package(files: list[Path]) -> list[dict]:
    dl = OUT / "downloads"
    dl.mkdir(parents=True, exist_ok=True)

    inventory = [{"file": p.name, "bytes": p.stat().st_size} for p in files]
    with (OUT / "video_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "bytes"])
        w.writeheader(); w.writerows(inventory)

    groups: list[list[Path]] = []
    cur: list[Path] = []
    cur_bytes = 0
    for p in files:
        size = p.stat().st_size
        if cur and cur_bytes + size > MAX_PART_BYTES:
            groups.append(cur); cur = []; cur_bytes = 0
        cur.append(p); cur_bytes += size
    if cur:
        groups.append(cur)

    info = []
    for idx, group in enumerate(groups, 1):
        zpath = dl / f"steven_adams_ft_miss_oreb_all_angles_2025_26_part_{idx:02d}.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
            for p in group:
                z.write(p, arcname=f"videos/{p.name}")
            for meta in ("event_manifest.csv", "event_manifest.json", "video_inventory.csv", "video_qa.json"):
                mp = OUT / meta
                if mp.exists():
                    z.write(mp, arcname=meta)
        info.append({
            "part": idx, "filename": zpath.name, "bytes": zpath.stat().st_size,
            "mb_decimal": round(zpath.stat().st_size / 1_000_000, 2), "video_files": len(group),
        })
    (OUT / "download_parts.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    pbp = load_pbp()
    events = find_events(pbp)
    del pbp

    print(f"QUALIFYING_EVENTS={len(events)}", flush=True)
    for ev in events:
        print("EVENT=" + json.dumps({k: ev.get(k) for k in (
            "rank", "game_id", "period", "clock", "ft_event_id", "rebound_event_id",
            "ft_shooter", "ft_description", "rebound_description"
        )}), flush=True)

    if not events:
        raise SystemExit("No qualifying events found")
    write_event_manifest(events)

    qa, files = build_videos(events)
    ok_by_rank = defaultdict(int)
    for row in qa:
        if row.get("status") == "ok":
            ok_by_rank[int(row["rank"])] += 1

    qa_payload = {
        "query": "Steven Adams offensive rebounds immediately after a Houston teammate missed free throw, 2025-26 regular season",
        "definition": "strict PBP adjacency: missed HOU FT by non-Adams shooter, immediately followed by HOU rebound credited to Steven Adams",
        "pbp_source": PBP_CSV_URL,
        "video_source": "clips.nba.com game/event pages -> signed lrmedia.nba.com HLS",
        "video_processing": "native source stream copy only; no upscaling, no denoise, no sharpening, no interpolation, no transcoding, no concatenation",
        "qualifying_events": len(events),
        "valid_distinct_native_angle_files": len(files),
        "ok_files_by_event_rank": dict(ok_by_rank),
        "angles": qa,
    }
    (OUT / "video_qa.json").write_text(json.dumps(qa_payload, indent=2), encoding="utf-8")

    missing = [ev for ev in events if ok_by_rank[int(ev["rank"])] == 0]
    if missing:
        print("MISSING_VIDEO_EVENTS=" + json.dumps(missing, indent=2), flush=True)
        sys.exit(2)

    parts = package(files)
    print("DOWNLOAD_PARTS=" + json.dumps(parts), flush=True)
    print(f"VALID_DISTINCT_NATIVE_ANGLE_FILES={len(files)}", flush=True)


if __name__ == "__main__":
    main()
