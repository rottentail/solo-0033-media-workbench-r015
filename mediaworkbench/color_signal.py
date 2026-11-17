"""Absolute-light decoding, whole-shot exposure and hue-preserving SDR mapping.

All processing happens on linear-light BT.709 RGB.  Source code values are
turned into absolute nits first (ST 2084 for PQ, 1000 nits full scale for
linear BT.2020 and 100 nits full scale for sRGB/BT.709), so the exposure and
highlight shoulder below work on the scene's real luminance relationships.

The functions operate on whole pictures or on bounded row bands with the same
result; the renderer feeds them small bands so memory does not grow with the
input resolution.
"""
import numpy as np

LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)
TO_709 = np.array([[1.660491, -.587641, -.072850],
                   [-.124550, 1.132900, -.008349],
                   [-.018151, -.100579, 1.118730]], dtype=np.float32)

# ST 2084 constants.
PQ_M1 = 2610 / 16384
PQ_M2 = 2523 / 32
PQ_C1 = 3424 / 4096
PQ_C2 = 2413 / 128
PQ_C3 = 2392 / 128

# Luminance above this fraction of display white rolls off continuously.  It
# starts with unit slope and leaves everything below untouched, so midtones keep
# their original ratios and the picture is not darkened to make room for peaks.
KNEE = .8


def _pq_eotf(values):
    """ST 2084 signal to absolute luminance in nits (0..10000)."""
    signal = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    power = signal ** np.float32(1 / PQ_M2)
    numerator = np.maximum(power - np.float32(PQ_C1), 0)
    denominator = PQ_C2 - PQ_C3 * power
    return np.float32(10000) * (numerator / np.maximum(denominator, 1e-12)) ** np.float32(1 / PQ_M1)


def _srgb_eotf(values):
    """IEC 61966-2-1 signal to relative luminance (1.0 == 100 nits)."""
    signal = np.maximum(np.asarray(values, dtype=np.float32), 0)
    return np.where(signal <= np.float32(.04045), signal / np.float32(12.92),
                    ((signal + np.float32(.055)) / np.float32(1.055)) ** np.float32(2.4)) * np.float32(100)


def decode(rgb, encoding):
    """Convert source code values to signed linear-light BT.709 RGB in nits.

    Out-of-gamut BT.2020 colours intentionally keep their negative BT.709
    components: hue-preserving gamut compression happens after tone mapping
    instead of clipping individual channels here.
    """
    values = np.asarray(rgb, dtype=np.float32)
    if values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError('finite three-channel colour required')
    if encoding == 'pq2020':
        return (_pq_eotf(values) @ TO_709.T).astype(np.float32, copy=False)
    if encoding == 'linear2020':
        return ((values * np.float32(1000)) @ TO_709.T).astype(np.float32, copy=False)
    if encoding == 'srgb709':
        return _srgb_eotf(values)
    raise ValueError('unsupported source colour encoding')


def accumulate(histogram, linear):
    """Add one picture's positive-luminance pixels to a log2 histogram."""
    values = np.asarray(linear, dtype=np.float32)
    luminance = values @ LUMA
    positive = luminance[luminance > 1e-8]
    if positive.size:
        coordinates = (np.log2(positive) + 12) * 1024 / 28
        bins = np.clip(coordinates.astype(int), 0, 1023)
        histogram += np.bincount(bins, minlength=1024)


def exposure(histogram, white_nits):
    """One exposure for a complete shot.

    The middle of the scene is the geometric mean of the log-luminance
    distribution after dropping the darkest and brightest 5% of pixels.  It is
    placed at 18% of the chosen display white.  A tiny specular object is
    therefore discarded by the percentile trim and cannot pull the whole shot
    down; no peak-headroom term is used, because highlight shape is handled by
    the continuous shoulder in :func:`map_frame`.
    """
    counts = np.asarray(histogram, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 1.
    centres = -12 + (np.arange(1024) + .5) * 28 / 1024
    cumulative = np.cumsum(counts)
    before = cumulative - counts
    start, stop = total * .05, total * .95
    # Pixels of every bin that fall inside the middle 90%; boundary bins are
    # counted partially instead of being dropped or included whole.
    admitted = np.clip(np.minimum(cumulative, stop) - np.maximum(before, start), 0, None)
    weight = float(admitted.sum())
    if weight <= 0:
        admitted, weight = counts, total
    middle = 2 ** (float(np.dot(centres, admitted)) / weight)
    gain = .18 * float(white_nits) / max(middle, 1e-12)
    return min(1e6, max(1e-6, gain))


def shoulder(luminance):
    """Continuous unit-slope highlight roll-off in display-relative units."""
    level = np.maximum(np.asarray(luminance, dtype=np.float32), 0)
    excess = np.maximum(level - np.float32(KNEE), 0) / np.float32(1 - KNEE)
    rolled = KNEE + (1 - KNEE) * excess / (1 + excess)
    return np.where(level <= KNEE, level, rolled).astype(np.float32, copy=False)


def map_frame(linear, gain, white_nits):
    """Tone map and gamut map one linear-light picture to display linear RGB.

    * Exposure scales every pixel by the same whole-shot gain.
    * The continuous shoulder is applied to the pixel's largest component and
      the same factor scales all three.  Hue and saturation ratios therefore
      survive, no channel is clipped on its own, and a brighter colour is never
      driven below a darker one of the same hue.  The shoulder asymptotes just
      below 1.0 but keeps a distinct level for every brighter level, so a sky
      gradient or a lit sign does not collapse into one flat white.
    * Out-of-gamut BT.2020 colours (a negative BT.709 component) are then moved
      along their constant-luminance hue ray towards the neutral axis only as
      far as needed to fit [0, 1].  Lightness and hue are preserved, the
      colour is not darkened as a whole and no channel is clipped separately.
    """
    values = np.asarray(linear, dtype=np.float32)
    relative = values * np.float32(gain) / np.float32(white_nits)
    peak = relative.max(axis=-1)
    positive = peak > 0
    scaled = shoulder(np.where(positive, peak, np.float32(0)))
    factor = np.where(positive, scaled / np.where(positive, peak, np.float32(1)),
                      np.float32(0))
    toned = relative * factor[..., None]
    return np.clip(_gamut_map(toned), 0, np.float32(1)).astype(np.float32, copy=False)


def _gamut_map(toned):
    """Reduce only saturation until the colour fits the BT.709 cube.

    The colour is moved towards the neutral axis of equal luminance.  Because
    RGB luminance is linear, the moved colour keeps its luminance exactly and
    keeps its hue direction; only chroma shrinks, and only by the amount needed
    to reach the cube boundary.  In-gamut colours are returned untouched.
    """
    luminance = np.clip(toned @ LUMA, 0, 1)
    delta = toned - luminance[..., None]
    lower = np.full_like(delta, np.inf)
    upper = np.full_like(delta, np.inf)
    np.divide(luminance[..., None], -delta, out=lower, where=delta < 0)
    np.divide((1 - luminance)[..., None], delta, out=upper, where=delta > 0)
    chroma = np.minimum(np.float32(1), np.minimum(lower, upper).min(axis=-1))
    return luminance[..., None] + chroma[..., None] * delta


def encode(linear):
    """sRGB display encoding of relative linear light."""
    values = np.clip(np.asarray(linear, dtype=np.float32), 0, 1)
    return np.where(values <= np.float32(.0031308), values * np.float32(12.92),
                    np.float32(1.055) * values ** np.float32(1 / 2.4) - np.float32(.055))
