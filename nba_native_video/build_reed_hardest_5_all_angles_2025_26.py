#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import html as htmlmod
import json
import math
import re
import shutil
import subprocess
import time
import zipfile
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

REED_ID = 1642263
PBP_URL = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.csv'
XFG_URL = 'https://stats.gleague.nba.com/stats/shotqualityvideologs'
OUT = Path('outputs/reed_hardest_5_all_angles_2025_26')
MAX_PART_BYTES = 95_000_000
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36'
NBA_HEADERS = {
    'User-Agent': UA,
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}
CLIP_HEADERS = {
    'User-Agent': UA,
    'Referer': 'https://clips.nba.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}
PLACEHOLDER_AHASH = [
    '0000180039fc39fc38c839fc39fc10000000',
    '0000180019fc39fc38c819fc39fc18000000',
    '0000100011fc11fc18ec11fc19fc10000000',
]


def norm_gid(v) -> str:
    s = str(v).strip()
    try:
        s = str(int(float(s)))
    except Exception:
        s = re.sub(r'\D', '', s)
    return s.zfill(10)


def fetch_pbp() -> pd.DataFrame:
    r = requests.get(PBP_URL, headers={'User-Agent': UA}, timeout=180)
    r.raise_for_status()
    print(f'PBP_BYTES={len(r.content)}', flush=True)
    return pd.read_csv(StringIO(r.text), low_memory=False)


def reed_games_from_pbp(pbp: pd.DataFrame) -> list[str]:
    gid = pbp['game_id'].map(norm_gid)
    reg = gid.str.startswith('002')
    fg = pd.to_numeric(pbp['is_field_goal'], errors='coerce').fillna(0).eq(1)
    p1 = pbp['player1_name'].fillna('').astype(str).str.strip()
    reed = p1.str.startswith(f'{REED_ID} ')
    games = sorted(gid[reg & fg & reed].dropna().unique().tolist())
    print(f'REED_GAMES_WITH_FGA={len(games)}', flush=True)
    return games


def fetch_xfg_game(game_id: str) -> dict:
    last = None
    for attempt in range(1, 7):
        try:
            r = requests.get(
                XFG_URL,
                params={'GameID': game_id, 'PlayerID': REED_ID},
                headers=NBA_HEADERS,
                timeout=(8, 45),
            )
            if r.status_code == 200:
                j = r.json()
                if str(j.get('gameId') or '').zfill(10) == game_id and int(j.get('playerId') or 0) == REED_ID:
                    return j
            last = f'status={r.status_code} prefix={r.text[:160]!r}'
        except Exception as exc:
            last = repr(exc)
        if attempt < 6:
            time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
    raise RuntimeError(f'xFG fetch failed {game_id}: {last}')


def shot_rows(games: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    errors = []
    for i, gid in enumerate(games, 1):
        try:
            j = fetch_xfg_game(gid)
            for s in (j.get('shotList') or []):
                xfg = s.get('shotQuality')
                try:
                    xfg = float(xfg)
                except Exception:
                    xfg = math.nan
                rows.append({
                    'season': '2025-26',
                    'season_type': 'Regular Season',
                    'game_id': str(s.get('gameId') or gid).zfill(10),
                    'game_date': j.get('gameDate'),
                    'matchup': j.get('matchup'),
                    'team_abbr': j.get('teamAbbreviation'),
                    'player_id': int(s.get('playerId') or REED_ID),
                    'player_name': j.get('playerName') or 'Reed Sheppard',
                    'event_num': int(s.get('eventNum')) if s.get('eventNum') is not None else None,
                    'event_type': str(s.get('eventType') or '').strip(),
                    'action_type': s.get('actionType'),
                    'shot_type': s.get('shotType'),
                    'made': int(s.get('success') or 0),
                    'period': s.get('period'),
                    'game_clock': s.get('gameClock'),
                    'xfg': xfg,
                    'xfg_pct': xfg * 100 if not math.isnan(xfg) else math.nan,
                    'loc_x': s.get('locX'),
                    'loc_y': s.get('locY'),
                    'guid': s.get('guid'),
                    'source_host': 'stats.gleague.nba.com',
                    'source_resource': 'shotqualityvideologs',
                })
        except Exception as exc:
            errors.append({'game_id': gid, 'error': repr(exc)})
        if i % 10 == 0 or i == len(games):
            print(f'XFG_GAMES={i}/{len(games)} SHOTS={len(rows)} ERRORS={len(errors)}', flush=True)
    return pd.DataFrame(rows), errors


def select_hardest_five(df: pd.DataFrame) -> pd.DataFrame:
    made = df[(df['player_id'] == REED_ID) & (df['made'] == 1) & df['xfg'].notna()].copy()
    ranked = made.sort_values(['xfg', 'game_date', 'game_id', 'event_num'], kind='stable').head(5).reset_index(drop=True)
    ranked.insert(0, 'rank_hardest', range(1, len(ranked) + 1))
    ranked['difficulty_definition'] = 'lowest official NBA shot-level xFG among Reed Sheppard made field goals'
    ranked['video_anchor'] = 'exact made field goal event_num'
    ranked['source_provenance'] = 'official NBA shotqualityvideologs endpoint'
    if len(ranked) != 5:
        raise RuntimeError(f'Expected 5 ranked makes, got {len(ranked)}')
    if ranked['event_num'].isna().any():
        raise RuntimeError('Selected row missing event_num')
    return ranked


def parse_clips_page(game_id: str, event_id: int) -> dict:
    page_url = f'https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs'
    r = requests.get(page_url, headers=CLIP_HEADERS, timeout=45)
    r.raise_for_status()
    text = r.text
    title_m = re.search(r'<title>(.*?)</title>', text, flags=re.I | re.S)
    title = htmlmod.unescape(title_m.group(1).strip()) if title_m else ''
    angles = []
    seen = set()
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', text, flags=re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        if '.m3u8' not in url.lower() or 'lrmedia.nba.com' not in url.lower() or url in seen:
            continue
        seen.add(url)
        label = re.sub(r'<[^>]+>', '', htmlmod.unescape(m.group(3))).strip() or 'angle'
        angles.append({'url': url, 'label': label, 'selected': 'selected' in m.group(2).lower()})
    if not angles:
        raise RuntimeError(f'No signed lrmedia HLS angles on {page_url}')
    return {'page_url': page_url, 'title': title, 'angles': angles}


def safe(s: str) -> str:
    x = re.sub(r'[^A-Za-z0-9._-]+', '_', s.strip()).strip('_')
    return x[:80] or 'angle'


def download_native_hls(url: str, out: Path) -> None:
    headers = f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-rw_timeout', '30000000',
        '-headers', headers, '-i', url,
        '-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', '-movflags', '+faststart', str(out),
    ]
    subprocess.run(cmd, check=True)


def ahash_frame(path: Path, t: float) -> str:
    p = subprocess.run([
        'ffmpeg', '-v', 'error', '-ss', str(t), '-i', str(path),
        '-vf', 'scale=16:9,format=gray', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'
    ], capture_output=True)
    if p.returncode or len(p.stdout) != 144:
        return ''
    vals = list(p.stdout)
    avg = sum(vals) / len(vals)
    bits = ''.join('1' if v >= avg else '0' for v in vals)
    return f'{int(bits, 2):036x}'


def hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    return (int(a, 16) ^ int(b, 16)).bit_count()


def probe(path: Path) -> dict:
    p = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate:format=duration', '-of', 'json', str(path)
    ], capture_output=True, text=True)
    if p.returncode:
        return {'ok': False, 'reason': 'ffprobe_failed'}
    try:
        j = json.loads(p.stdout)
        s = j['streams'][0]
        duration = float((j.get('format') or {}).get('duration') or 0)
    except Exception:
        return {'ok': False, 'reason': 'ffprobe_parse'}
    times = [min(max(duration * f, 0.25), max(duration - 0.25, 0.25)) for f in (0.25, 0.5, 0.75)]
    hashes = [ahash_frame(path, t) for t in times]
    distances = [hamming(a, b) for a, b in zip(hashes, PLACEHOLDER_AHASH)]
    placeholder = all(d <= 12 for d in distances)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ok = duration >= 2.0 and all(hashes) and not placeholder
    return {
        'ok': ok,
        'reason': None if ok else ('known_nba_video_not_available_placeholder' if placeholder else 'invalid_media'),
        'duration': duration,
        'codec': s.get('codec_name'),
        'width': s.get('width'),
        'height': s.get('height'),
        'avg_frame_rate': s.get('avg_frame_rate'),
        'bytes': path.stat().st_size,
        'sha256': sha,
        'visual_fingerprints': hashes,
        'placeholder_hamming': distances,
    }


def download_all_angles(selected: pd.DataFrame) -> tuple[list[dict], list[Path]]:
    native = OUT / 'native_all_angles'
    native.mkdir(parents=True, exist_ok=True)
    qa = []
    kept = []
    global_sha = defaultdict(list)

    for row in selected.itertuples(index=False):
        rank = int(row.rank_hardest)
        gid = str(row.game_id)
        eid = int(row.event_num)
        page = parse_clips_page(gid, eid)
        seen_event_sha = set()
        print(f'EVENT rank={rank} game={gid} event={eid} xFG={row.xfg_pct:.2f}% angles={len(page["angles"])}', flush=True)
        for ai, angle in enumerate(page['angles'], 1):
            rec = {
                'rank_hardest': rank,
                'game_id': gid,
                'event_num': eid,
                'xfg': float(row.xfg),
                'xfg_pct': float(row.xfg_pct),
                'action_type': row.action_type,
                'shot_type': row.shot_type,
                'period': row.period,
                'game_clock': row.game_clock,
                'clips_page_url': page['page_url'],
                'clips_page_title': page['title'],
                'angle_index': ai,
                'angle_count_on_page': len(page['angles']),
                'angle_label': angle['label'],
                'selected_on_page': angle['selected'],
                'status': 'failed',
            }
            path = native / f'{rank:02d}_{gid}_{eid}_a{ai:02d}_{safe(angle["label"])}.mp4'
            try:
                download_native_hls(angle['url'], path)
                q = probe(path)
                rec['probe'] = q
                if not q['ok']:
                    raise RuntimeError(q['reason'])
                sha = q['sha256']
                if sha in seen_event_sha:
                    rec['status'] = 'duplicate_angle_excluded'
                    path.unlink(missing_ok=True)
                else:
                    seen_event_sha.add(sha)
                    global_sha[sha].append((gid, eid, angle['label']))
                    rec['status'] = 'ok'
                    rec['native_file'] = path.name
                    kept.append(path)
            except Exception as exc:
                rec['error'] = repr(exc)
                path.unlink(missing_ok=True)
            qa.append(rec)

    # Reject identical media reused across distinct shot events.
    bad_shas = {sha for sha, locs in global_sha.items() if len({(g, e) for g, e, _ in locs}) > 1}
    if bad_shas:
        for rec in qa:
            q = rec.get('probe') or {}
            if rec.get('status') == 'ok' and q.get('sha256') in bad_shas:
                rec['status'] = 'failed_duplicate_across_events'
                p = native / rec['native_file']
                p.unlink(missing_ok=True)
                if p in kept:
                    kept.remove(p)
    return qa, kept


def write_manifest(df: pd.DataFrame) -> None:
    df.to_csv(OUT / 'reed_hardest_5_manifest.csv', index=False)
    (OUT / 'reed_hardest_5_manifest.json').write_text(json.dumps(df.to_dict('records'), indent=2, default=str), encoding='utf-8')


def package(files: list[Path]) -> list[dict]:
    dl = OUT / 'downloads'
    dl.mkdir(parents=True, exist_ok=True)
    meta_names = ['reed_hardest_5_manifest.csv', 'reed_hardest_5_manifest.json', 'video_qa.json', 'video_inventory.csv']
    inv = [{'file': p.name, 'bytes': p.stat().st_size} for p in files]
    pd.DataFrame(inv).to_csv(OUT / 'video_inventory.csv', index=False)

    groups, cur, cur_bytes = [], [], 0
    for p in files:
        size = p.stat().st_size
        if cur and cur_bytes + size > MAX_PART_BYTES:
            groups.append(cur); cur = []; cur_bytes = 0
        cur.append(p); cur_bytes += size
    if cur:
        groups.append(cur)

    parts = []
    for i, group in enumerate(groups, 1):
        zpath = dl / f'reed_sheppard_5_hardest_makes_all_angles_native_2025_26_part_{i:02d}.zip'
        with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_STORED) as z:
            for p in group:
                z.write(p, arcname=f'videos/{p.name}')
            for name in meta_names:
                mp = OUT / name
                if mp.exists():
                    z.write(mp, arcname=name)
        parts.append({'part': i, 'filename': zpath.name, 'bytes': zpath.stat().st_size, 'mb': round(zpath.stat().st_size / 1e6, 2), 'video_files': len(group)})
    (OUT / 'download_parts.json').write_text(json.dumps(parts, indent=2), encoding='utf-8')
    return parts


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    pbp = fetch_pbp()
    games = reed_games_from_pbp(pbp)
    if not games:
        raise SystemExit('No Reed Sheppard FGA games found')

    shots, errors = shot_rows(games)
    shots.to_csv(OUT / 'reed_all_official_xfg_shots_2025_26.csv', index=False)
    pd.DataFrame(errors).to_csv(OUT / 'xfg_fetch_errors.csv', index=False)
    if errors:
        raise RuntimeError(f'xFG fetch errors for {len(errors)} games; refuse incomplete ranking')

    selected = select_hardest_five(shots)
    write_manifest(selected)
    print('TOP5=' + json.dumps(selected[['rank_hardest','game_id','game_date','matchup','event_num','action_type','shot_type','period','game_clock','xfg_pct','loc_x','loc_y']].to_dict('records'), default=str), flush=True)

    qa, files = download_all_angles(selected)
    ok_by_rank = defaultdict(int)
    for r in qa:
        if r.get('status') == 'ok':
            ok_by_rank[int(r['rank_hardest'])] += 1
    payload = {
        'player': 'Reed Sheppard',
        'player_id': REED_ID,
        'season': '2025-26 regular season',
        'selection': 'five made field goals with lowest official NBA shot-level xFG',
        'xfg_source': XFG_URL,
        'video_source': 'clips.nba.com exact game/event page -> all distinct signed lrmedia.nba.com HLS angle options',
        'video_processing': 'native stream copy only (-c copy); no upscale, no resize, no denoise, no sharpening, no interpolation',
        'selected_events': 5,
        'valid_distinct_native_angle_files': len(files),
        'ok_angles_by_rank': dict(ok_by_rank),
        'angle_records': qa,
    }
    (OUT / 'video_qa.json').write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')

    missing = [rank for rank in range(1, 6) if ok_by_rank[rank] == 0]
    failed = [r for r in qa if str(r.get('status', '')).startswith('failed')]
    if missing or failed:
        print(f'MISSING_RANKS={missing} FAILED_ANGLE_RECORDS={len(failed)}', flush=True)
        raise SystemExit(2)

    parts = package(files)
    print('DOWNLOAD_PARTS=' + json.dumps(parts), flush=True)
    print(f'VALID_DISTINCT_NATIVE_ANGLE_FILES={len(files)}', flush=True)


if __name__ == '__main__':
    main()
