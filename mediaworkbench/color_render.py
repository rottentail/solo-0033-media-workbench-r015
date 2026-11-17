"""Decode each occurrence once, analyse its whole shot and stream display frames.

A render materializes each trimmed occurrence to a private high-precision RGB
spool exactly once.  That spool is then read one bounded row band at a time:
first to build the complete shot's log-luminance histogram, then to emit the
requested pictures.  A middle preview therefore applies the same whole-shot
exposure as the full delivery while the original is decoded only once, and no
picture or band set is kept in memory.
"""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from . import color_plan, color_signal, media_sources, planner
from .assets import fingerprint
from .tools import run

# Rows processed per band.  Bounds both the analysis and the mapping workspace
# independently of the source resolution.
BAND_ROWS = 16

MATRICES = {'bt709': 'bt709', 'bt470bg': 'bt601', 'smpte170m': 'bt601',
            'bt2020nc': 'bt2020', 'bt2020c': 'bt2020', 'bt2020': 'bt2020',
            'fcc': 'fcc', 'smpte240m': 'smpte240m', 'bt470m': 'bt470'}
RGB_PIXEL_FORMATS = ('gbrp', 'rgb', 'bgr', 'argb', 'abgr', 'rgba', 'bgra',
                     'gbrap', 'ya8')


def video_stream(clip):
    return next(stream for stream in json.loads(clip['metadata'])['streams']
                if stream.get('codec_type') == 'video')


def conversion(clip):
    """Matrix and range used to turn the source into high-precision RGB."""
    stream = video_stream(clip)
    pixel_format = (stream.get('pix_fmt') or '').lower()
    tagged_range = (stream.get('color_range') or '').lower()
    if pixel_format.startswith(RGB_PIXEL_FORMATS):
        # RGB code values are full range unless the file says otherwise.
        return {'rgb': True,
                'range': 'tv' if tagged_range in ('tv', 'limited') else 'pc'}
    # Untagged YUV is conventionally limited range; only an explicit full-range
    # tag switches the conversion.
    source_range = 'pc' if tagged_range in ('pc', 'jpeg') else 'tv'
    matrix = MATRICES.get((stream.get('color_space') or '').lower())
    if matrix is None:
        matrix = 'bt709' if clip['encoding'] == 'srgb709' else 'bt2020'
    return {'rgb': False, 'matrix': matrix, 'range': source_range}


def materialize(clip, root, index, width, height):
    """One sequential decode of the trimmed occurrence to 16-bit RGB on disk."""
    path = root / f'source-{index}.rgb48'
    settings = conversion(clip)
    if settings['rgb']:
        scale = (f'scale={width}:{height}:in_range={settings["range"]}:'
                 'out_range=full:flags=bicubic')
    else:
        scale = (f'scale={width}:{height}:in_color_matrix={settings["matrix"]}:'
                 f'in_range={settings["range"]}:out_range=full:flags=bicubic')
    run('ffmpeg', ['-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
        '-ss', str(clip['start_ms'] / 1000), '-i', clip['path'],
        '-t', str((clip['end_ms'] - clip['start_ms']) / 1000), '-an',
        '-vf', f'{scale},fps=25', '-frames:v', str(clip['frames']),
        '-pix_fmt', 'rgb48le', '-f', 'rawvideo', str(path)])
    expected = width * height * 6 * clip['frames']
    if path.stat().st_size != expected:
        raise ValueError('colour source did not provide every planned picture')
    return path


def frame_bands(path, clip, width, height, first, count):
    """Yield (frame index, linear-light rows) for a bounded frame range.

    Only ``BAND_ROWS`` picture rows are converted at a time.  ``first`` and
    ``count`` address frames of the materialized spool; analysis passes the
    whole shot while delivery passes the requested range.
    """
    frame_bytes = width * height * 6
    with path.open('rb') as handle:
        for local in range(first, first + count):
            handle.seek(local * frame_bytes)
            remaining = height
            while remaining:
                rows = min(BAND_ROWS, remaining)
                raw = handle.read(rows * width * 6)
                if len(raw) != rows * width * 6:
                    raise ValueError('truncated high precision picture')
                rgb = np.frombuffer(raw, dtype='<u2').reshape(rows, width, 3) / np.float32(65535)
                yield local, color_signal.decode(rgb, clip['encoding'])
                remaining -= rows


def shot_exposure(path, clip, width, height, white_nits):
    """One exposure decision from the complete trimmed shot."""
    histogram = np.zeros(1024, dtype=np.int64)
    for _, band in frame_bands(path, clip, width, height, 0, clip['frames']):
        color_signal.accumulate(histogram, band)
    gain = color_signal.exposure(histogram, white_nits)
    return gain


def display_frames(path, clip, width, height, first, count, gain, white_nits, output):
    """Write the requested pictures as 8-bit sRGB bytes."""
    bands = 0
    for _, band in frame_bands(path, clip, width, height, first, count):
        mapped = color_signal.map_frame(band, gain, white_nits)
        output.write(np.rint(color_signal.encode(mapped) * 255).astype('uint8').tobytes())
        bands += 1
    if bands != count * bands_per_frame(height):
        raise ValueError('colour shot did not provide every requested picture')


def bands_per_frame(height):
    return (height + BAND_ROWS - 1) // BAND_ROWS


def render(conn, identity, destination, at_frame=0, frame_count=None):
    plan = color_plan.get(conn, identity)
    frame_count = plan['frames'] - at_frame if frame_count is None else frame_count
    if (type(at_frame) is not int or type(frame_count) is not int
            or at_frame < 0 or frame_count < 1 or at_frame + frame_count > plan['frames']):
        raise ValueError('invalid colour preview range')
    snapshot = media_sources.bind(conn, plan['snapshot'])
    planner.verify_sources(snapshot)
    target = Path(destination).expanduser().absolute()
    target = target.parent.resolve() / target.name
    if os.path.lexists(target):
        raise ValueError('colour destination exists')
    if not target.parent.is_dir():
        raise ValueError('colour destination directory does not exist')
    width, height = plan['width'], plan['height']
    with tempfile.TemporaryDirectory(prefix='.color-', dir=target.parent) as directory:
        root = Path(directory)
        raw_output = root / 'display.rgb'
        cursor = 0
        gains = []
        with raw_output.open('wb') as output:
            for index, clip in enumerate(snapshot['clips']):
                clip['encoding'] = color_plan.encoding_of(plan, index)
                local_start = max(0, at_frame - cursor)
                local_end = min(clip['frames'], at_frame + frame_count - cursor)
                if local_start >= local_end:
                    gains.append(None)
                    cursor += clip['frames']
                    continue
                source = materialize(clip, root, index, width, height)
                try:
                    # The whole shot decides this occurrence's exposure, even
                    # when only a middle part is being previewed.
                    gain = shot_exposure(source, clip, width, height, plan['white_nits'])
                    display_frames(source, clip, width, height, local_start,
                                   local_end - local_start, gain,
                                   plan['white_nits'], output)
                finally:
                    source.unlink()
                gains.append(gain)
                cursor += clip['frames']
        if raw_output.stat().st_size != frame_count * width * height * 3:
            raise ValueError('colour delivery is incomplete')
        result = root / 'delivery.mkv'
        run('ffmpeg', ['-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
            '-r', '25', '-i', str(raw_output), '-an', '-c:v', 'ffv1', '-level', '3',
            '-pix_fmt', 'gbrp', '-colorspace', 'bt709', '-color_primaries', 'bt709',
            '-color_trc', 'iec61966-2-1', '-color_range', 'pc', str(result)])
        planner.verify_sources(snapshot)
        checksum = fingerprint(result)
        with result.open('rb') as handle:
            os.fsync(handle.fileno())
        os.link(result, target)
    return dict(plan_id=identity, path=str(target), frames=frame_count,
                at_frame=at_frame, gains=gains, sha256=checksum)
