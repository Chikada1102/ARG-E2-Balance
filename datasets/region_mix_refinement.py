"""Training-only target feedback and image-only ROI seam treatment for V4.1.

No test measurements, external CSVs, image exports, or model predictions enter
either operation. Geometry, donor eligibility and label interpolation stay V4.
"""
import numpy as np
from scipy import ndimage

REFINEMENTS = ('none', 'balance', 'seam', 'both')
REFINE_METRICS = ('match_checked', 'match_pixels', 'matched_pixels_changed',
                  'intensity_gain', 'intensity_offset', 'feather_pixels',
                  'seam_pairs', 'seam_jump_before', 'seam_jump_after', 'identity_fallback')
BALANCE_RULE = dict(version=1, prior_exposures=50., prior_applications=5.,
                    exponent=.5, min_multiplier=.5, max_multiplier=2.)
SEAM_RULE = dict(version=1, width_output_pixels=2., organ_band_output_pixels=1.,
                 min_gain=.9, max_gain=1.1, max_offset_scale=.25)


class TargetBalance:
    """One update per complete training pass, in the consumer/main process.

The multiplier equalizes successful applications per visible positive slice;
it does not target equal raw counts or promise that rare organs become feasible.
The first epoch uses the original V4 weights. Feedback starts at epoch two.
"""
    def __init__(self, visible_frequency, fingerprint):
        self.frequency = np.asarray(visible_frequency, dtype=np.int64).copy()
        if self.frequency.shape != (9,) or (self.frequency < 0).any() or self.frequency[0]:
            raise ValueError('Expected nine nonnegative visible frequencies with background zero')
        self.fingerprint = str(fingerprint)
        self.completed_epochs = 0
        self.selected = np.zeros(9, np.int64)
        self.applied = np.zeros(9, np.int64)

    def rates(self):
        return ((self.applied + BALANCE_RULE['prior_applications']) /
                (self.frequency * self.completed_epochs + BALANCE_RULE['prior_exposures']))

    def weights(self):
        values = np.ones(9, np.float64)
        present = self.frequency > 0
        if self.completed_epochs and present.any():
            rates = self.rates()
            reference = float(np.exp(np.log(rates[present]).mean()))
            values[present] = np.clip((reference / rates[present]) ** BALANCE_RULE['exponent'],
                                      BALANCE_RULE['min_multiplier'], BALANCE_RULE['max_multiplier'])
        return values

    def update(self, epoch, selected, applied):
        a, b = np.asarray(selected), np.asarray(applied)
        if int(epoch) != self.completed_epochs + 1:
            raise RuntimeError('Balance feedback must consume consecutive complete training epochs')
        for array in (a, b):
            if (array.shape != (9,) or not np.isfinite(array).all() or
                    (array < 0).any() or (array != np.floor(array)).any() or array[0]):
                raise ValueError('Invalid training target counts')
        if (b > a).any() or (a > self.frequency).any():
            raise ValueError('Target counts exceed visible training exposures')
        self.selected += a.astype(np.int64)
        self.applied += b.astype(np.int64)
        self.completed_epochs = int(epoch)

    def state_dict(self):
        return dict(version=1, rule=BALANCE_RULE, fingerprint=self.fingerprint,
                    visible_frequency=self.frequency.tolist(), completed_epochs=self.completed_epochs,
                    selected=self.selected.tolist(), applied=self.applied.tolist())

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get('version') != 1 or
                state.get('rule') != BALANCE_RULE or state.get('fingerprint') != self.fingerprint or
                state.get('visible_frequency') != self.frequency.tolist()):
            raise RuntimeError('Balance state belongs to a different training pool or rule')
        epochs = state.get('completed_epochs')
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 0:
            raise RuntimeError('Invalid completed balance epochs')
        try:
            selected = np.asarray(state['selected'], dtype=np.float64)
            applied = np.asarray(state['applied'], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError('Invalid saved balance counts') from error
        if (selected.shape != (9,) or applied.shape != (9,) or
                not np.isfinite(selected).all() or not np.isfinite(applied).all() or
                (selected < 0).any() or (applied < 0).any() or
                (selected != np.floor(selected)).any() or (applied != np.floor(applied)).any() or
                selected[0] or applied[0] or (applied > selected).any() or
                (selected > self.frequency * epochs).any()):
            raise RuntimeError('Invalid saved balance counts')
        self.selected = selected.astype(np.int64)
        self.applied = applied.astype(np.int64)
        self.completed_epochs = epochs


def refine_patch(anchor, donor, anchor_label, donor_label, mask, sampling, intensity_scale):
    """Fit a bounded local gray transform, then feather common background only.

Foreground pixels use the donor transform with a hard image/label assignment.
The feather never crosses either image's foreground or its one-output-pixel
band, and never writes outside the existing V4 mixing mask.
"""
    result = anchor.copy()
    common = mask & (anchor_label == 0) & (donor_label == 0)
    if not common.any():
        raise RuntimeError('Refinement requires the context already required by V4')
    donor_values, anchor_values = donor[common], anchor[common]
    dq = np.percentile(donor_values, [25, 50, 75])
    aq = np.percentile(anchor_values, [25, 50, 75])
    spread = float(dq[2] - dq[0])
    gain = (float(np.clip((aq[2] - aq[0]) / spread, SEAM_RULE['min_gain'], SEAM_RULE['max_gain']))
            if spread > 1e-6 else 1.)
    limit = SEAM_RULE['max_offset_scale'] * max(float(intensity_scale), 1e-6)
    offset = float(np.clip(aq[1] - dq[1], -limit, limit))
    matched = ((donor - float(dq[1])) * gain + float(dq[1]) + offset).astype(np.float32)
    if not np.isfinite(matched).all():
        raise RuntimeError('Non-finite matched donor image')
    result[mask] = matched[mask]
    # Explicit padding makes the outer ROI edge a real seam even for an all-1 mask.
    edge_distance = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=sampling)[1:-1, 1:-1]
    anchor_distance = (ndimage.distance_transform_edt(anchor_label == 0, sampling=sampling)
                       if np.any(anchor_label) else np.full(mask.shape, np.inf))
    donor_distance = (ndimage.distance_transform_edt(donor_label == 0, sampling=sampling)
                      if np.any(donor_label) else np.full(mask.shape, np.inf))
    feather = (common & (edge_distance < SEAM_RULE['width_output_pixels']) &
               (anchor_distance > SEAM_RULE['organ_band_output_pixels']) &
               (donor_distance > SEAM_RULE['organ_band_output_pixels']))
    alpha = np.clip(edge_distance[feather] / SEAM_RULE['width_output_pixels'], 0., 1.)
    result[feather] = alpha * matched[feather] + (1. - alpha) * anchor[feather]
    metrics = np.zeros(len(REFINE_METRICS), np.float64)
    metrics[:6] = (1., int(common.sum()), int(np.count_nonzero(matched[mask] != donor[mask])),
                   gain, offset, int(feather.sum()))
    if not np.array_equal(result[~mask], anchor[~mask]):
        raise RuntimeError('Seam treatment changed protected pixels')
    return result, metrics


def background_seam(before, after, mask, anchor_label, new_label):
    """Mean absolute adjacent-pixel jump across the mask, in common background.

This is an image diagnostic in native intensity units, not segmentation HD95.
"""
    common = (anchor_label == 0) & (new_label == 0)
    count, total_before, total_after = 0, 0., 0.
    for axis in (0, 1):
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        edges = (mask[left] != mask[right]) & common[left] & common[right]
        count += int(edges.sum())
        total_before += float(np.abs(before[left] - before[right])[edges].sum(dtype=np.float64))
        total_after += float(np.abs(after[left] - after[right])[edges].sum(dtype=np.float64))
    return count, total_before / max(1, count), total_after / max(1, count)
