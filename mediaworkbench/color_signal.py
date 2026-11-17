"""Absolute-light decoding, scene exposure and luminance-preserving SDR mapping."""
import numpy as np

LUMA = np.array([.2126, .7152, .0722])
TO_709 = np.array([[1.660491, -.587641, -.072850],
                   [-.124550, 1.132900, -.008349],
                   [-.018151, -.100579, 1.118730]])


def decode(rgb, encoding):
    values = np.asarray(rgb, dtype=float)
    if values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError('finite three-channel colour required')
    if encoding == 'pq2020':
        linear = np.maximum(values, 0) ** 2.4 * 1000
    elif encoding == 'linear2020':
        linear = values * 1000
    elif encoding == 'srgb709':
        return np.maximum(values, 0) ** 2.2 * 100
    else:
        raise ValueError('unsupported source colour encoding')
    return np.maximum(linear @ TO_709.T, 0)


def exposure(histogram, white_nits):
    counts = np.asarray(histogram, dtype=float)
    occupied = np.flatnonzero(counts)
    if not len(occupied):
        return 1.
    total = counts.sum()
    centres = -12 + (np.arange(1024) + .5) * 28 / 1024
    levels = 2 ** centres
    mean = float(np.dot(levels, counts) / total)
    cumulative = np.cumsum(counts)
    tail = min(1023, int(np.searchsorted(cumulative, total * .9999)))
    peak = float(levels[tail])
    body_gain = .18 * white_nits / max(mean, 1e-8)
    headroom_gain = white_nits / max(peak, 1e-8)
    return min(64., max(.001, min(body_gain, headroom_gain)))


def accumulate(histogram, linear):
    values = np.maximum(linear, 0)
    luminance = values @ LUMA
    positive = luminance[luminance > 1e-8]
    if positive.size:
        coordinates = (np.log2(positive) + 12) * 1024 / 28
        bins = np.clip(coordinates.astype(int), 0, 1023)
        counts = np.bincount(bins, minlength=1024)
        histogram += counts


def map_frame(linear, gain, white_nits):
    scaled = np.maximum(np.asarray(linear, dtype=float), 0) * gain / white_nits
    peak = scaled.max(axis=-1)
    highlight = peak > .6
    shoulder = .6 + .4 * np.maximum(peak - .6, 0) / (.4 + np.maximum(peak - .6, 0))
    scale = np.ones_like(peak)
    scale[highlight] = shoulder[highlight] / np.maximum(peak[highlight], 1e-12)
    compressed = scaled * scale[..., None]
    display = np.minimum(compressed, 1)
    return np.maximum(display, 0)



def encode(linear):
    return np.where(linear <= .0031308, linear * 12.92,
                    1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055)
