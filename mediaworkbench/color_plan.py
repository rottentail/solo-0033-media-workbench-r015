"""Freeze occurrence colour interpretation and display conditions.

Every occurrence keeps its own trim, source identity, colour encoding and
frame count.  The snapshot is immutable, so later edits, reordering, trim
changes or a process restart do not alter a saved plan.  Whole-shot exposure
statistics are derived by the renderer from the complete materialized shot
(a single sequential decode), which keeps a middle preview identical to the
matching frames of a full delivery without analysing only the previewed range.
"""
import hashlib
import json
import math
from . import db, timeline, planner, media_sources

VERSION = 2


def create(conn, project_id, encodings, white_nits=100):
    if (type(white_nits) not in (int, float) or not math.isfinite(white_nits)
            or not 80 <= white_nits <= 300):
        raise ValueError('display white must be between 80 and 300 nits')
    snapshot = media_sources.bind(conn, timeline.snapshot(conn, project_id))
    frozen = planner.prepare(snapshot)
    if (not isinstance(encodings, list) or len(encodings) != len(frozen['clips'])
            or any(e not in ('pq2020', 'linear2020', 'srgb709') for e in encodings)):
        raise ValueError('one supported colour encoding per occurrence required')
    duration = sum(c['end_ms'] - c['start_ms'] for c in frozen['clips'])
    if not 40 <= duration <= 30000:
        raise ValueError('colour programme must be between 40 ms and 30 seconds')
    if any(c['start_ms'] % 40 or c['end_ms'] % 40 for c in frozen['clips']):
        raise ValueError('colour review cuts must align to the 25 fps picture grid')
    for clip in frozen['clips']:
        video = next(stream for stream in json.loads(clip['metadata'])['streams'] if stream.get('codec_type') == 'video')
        if not 1 <= video['width'] <= 1280 or not 1 <= video['height'] <= 720:
            raise ValueError('colour review supports sources up to 1280 by 720')
    for clip, encoding in zip(frozen['clips'], encodings):
        clip['encoding'] = encoding
        clip['frames'] = max(1, round((clip['end_ms'] - clip['start_ms']) / 40))
    payload = dict(version=VERSION, snapshot=frozen, white_nits=white_nits,
                   width=160, height=90, fps=25,
                   frames=sum(c['frames'] for c in frozen['clips']))
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    checksum = hashlib.sha256(encoded.encode()).hexdigest()
    with db.write(conn):
        cursor = conn.execute('INSERT INTO color_plans(payload,payload_sha256) VALUES (?,?)',
                              (encoded, checksum))
    return get(conn, cursor.lastrowid)


def get(conn, identity):
    row = conn.execute('SELECT * FROM color_plans WHERE id=?', (identity,)).fetchone()
    if row is None:
        raise ValueError('colour plan not found')
    if hashlib.sha256(row['payload'].encode()).hexdigest() != row['payload_sha256']:
        raise ValueError('colour plan checksum mismatch')
    plan = json.loads(row['payload'])
    if plan['version'] not in (1, VERSION):
        raise ValueError('unsupported colour plan')
    plan['id'] = identity
    return plan


def encoding_of(plan, index):
    """Frozen colour interpretation of one occurrence.

    Version 1 plans grouped occurrences by asset and stored one encoding per
    group; recover the occurrence's encoding from that group, but never reuse
    another occurrence's trim or exposure.
    """
    clips = plan['snapshot']['clips']
    if not 0 <= index < len(clips):
        raise ValueError('colour occurrence out of range')
    clip = clips[index]
    if 'encoding' in clip:
        return clip['encoding']
    presets = plan.get('display_presets') or {}
    preset = presets.get(clip.get('display_group', str(index)))
    if preset is not None:
        positions = preset.get('positions') or []
        if index in positions or not positions:
            return preset['encoding']
    return 'srgb709'
