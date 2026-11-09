"""VM-UNet training-only augmentation: exact V1 control and targeted V3.

Only NumPy/SciPy are needed by process(); PyTorch is imported in __call__.
The dataset lists, sampling, training objective and model are not changed here.
"""
import hashlib
import numpy as np
from scipy import ndimage
from .multi_organ_aug_v1 import MultiOrganRandomGenerator as V1
from .region_mix import blank_report, array_hash

OPS = ("crop", "rotate", "intensity", "adaptive_crop", "local_intensity", "region_mix")
STATES = ("checked", "triggered", "applied", "rejected", "ineligible")
PRESETS = {
    "baseline": dict.fromkeys(OPS, 0.),
    "v1": dict(zip(OPS, (.35, .5, .5, 0., 0., 0.))),
    "v3": dict(zip(OPS, (0., .5, .5, .35, .25, 0.))),
    "v4": dict(zip(OPS, (.35, .5, .5, 0., 0., .2))),
}


class MultiOrganRandomGenerator(V1):
    def __init__(self, output_size, preset="v3", disable=(), min_retention=.8,
                 preserve_organs=True, probabilities=None):
        super().__init__(output_size, min_retention=min_retention)
        if preset not in PRESETS:
            raise ValueError("Expected baseline, v1, v3 or v4")
        self.preset = preset
        self.probabilities = dict(PRESETS[preset])
        self.preserve_organs = bool(preserve_organs)
        self.num_classes = 9
        disable = tuple(disable)
        if set(disable) - set(OPS):
            raise ValueError("Unknown augmentation in disable")
        if preset == "v3" and "adaptive_crop" in disable:
            # Ablating the replacement restores the original V1 crop.
            self.probabilities["crop"] = .35
        for name in disable:
            self.probabilities[name] = 0.
        if probabilities:
            if set(probabilities) - set(OPS):
                raise ValueError("Unknown probability name")
            self.probabilities.update(probabilities)
        if any(not np.isfinite(p) or not 0 <= p <= 1
               for p in self.probabilities.values()):
            raise ValueError("Probabilities must be finite and in [0,1]")
        if self.probabilities["crop"] and self.probabilities["adaptive_crop"]:
            raise ValueError("V1 crop and adaptive crop are alternative operations")

    def _retains_organs(self, before, after, area_scale=1.):
        return (not self.preserve_organs or
                super()._retains_organs(before, after, area_scale))

    @staticmethod
    def _target_retained(before, after, target, area_scale=1., minimum=.95):
        old = np.count_nonzero(before == target)
        new = np.count_nonzero(after == target)
        return new > 0 and new >= minimum * old * area_scale

    def _crop(self, image, label, adaptive=False):
        ids = np.unique(label)
        ids = ids[ids != 0]
        detail = dict(target=-1, trials=0, zoom=1.)
        if not ids.size:
            return image, label, "ineligible", detail
        if adaptive:
            counts = np.array([np.count_nonzero(label == c) for c in ids], float)
            # A bounded preference for small visible organs; no test statistics.
            weights = np.minimum(np.sqrt(counts.max() / counts), 3.)
            target = int(np.random.choice(ids, p=weights / weights.sum()))
        else:
            # The exact draw used in the original V1 implementation.
            target = int(np.random.choice(ids))
        detail["target"] = target
        coords = np.argwhere(label == target)
        h, w = label.shape
        small = len(coords) / float(h * w) <= .01
        attempts = 8 if adaptive else 5
        for _ in range(attempts):
            detail["trials"] += 1
            cy, cx = coords[np.random.randint(len(coords))]
            bounds = (.75, .95) if adaptive and small else (.85, 1.)
            fraction = np.random.uniform(*bounds)
            ch = max(1, int(round(h * fraction)))
            cw = max(1, int(round(w * fraction)))
            top = int(np.clip(cy - ch // 2, 0, h - ch))
            left = int(np.clip(cx - cw // 2, 0, w - cw))
            cropped = label[top:top + ch, left:left + cw]
            if not self._retains_organs(label, cropped):
                continue
            if adaptive and self.preserve_organs and not self._target_retained(label, cropped, target):
                continue
            ci, cl = self._resize(image[top:top + ch, left:left + cw], cropped, (h, w))
            scale = h * w / float(ch * cw)
            if not self._retains_organs(label, cl, scale):
                continue
            if adaptive and self.preserve_organs and not self._target_retained(label, cl, target, scale):
                continue
            detail["zoom"] = np.sqrt(scale)
            # A rounded full-image V1 crop is returned, without extra retries.
            # This preserves both V1 output and its random state.
            state = "ineligible" if np.array_equal(image, ci) and np.array_equal(label, cl) else "applied"
            return ci, cl, state, detail
        return image, label, "rejected", detail

    def _local_intensity(self, image, label):
        ids, counts = np.unique(label, return_counts=True)
        counts, ids = counts[ids != 0], ids[ids != 0]
        lo, hi, spread = float(image.min()), float(image.max()), float(image.std())
        if not ids.size or hi <= lo or spread <= 1e-8:
            return image, "ineligible", -1
        weights = np.minimum(np.sqrt(counts.max() / counts.astype(float)), 3.)
        target = int(np.random.choice(ids, p=weights / weights.sum()))
        yy, xx = np.where(label == target)
        h, w = label.shape
        height, width = float(yy.max() - yy.min() + 1), float(xx.max() - xx.min() + 1)
        cy = .5 * (yy.min() + yy.max()) + np.random.uniform(-.15, .15) * height
        cx = .5 * (xx.min() + xx.max()) + np.random.uniform(-.15, .15) * width
        sy, sx = max(.75 * height, .08 * h), max(.75 * width, .08 * w)
        grid_y, grid_x = np.ogrid[:h, :w]
        # A smooth ellipse extends into surrounding tissue. It does not follow
        # the exact label contour, which would stamp an artificial sharp edge.
        field = np.exp(-.5 * (((grid_y - cy) / sy) ** 2 + ((grid_x - cx) / sx) ** 2))
        mean = float((image * field).sum() / field.sum())
        contrast = np.random.uniform(.9, 1.1)
        shift = np.random.uniform(-.08, .08) * spread
        delta = field * ((contrast - 1.) * (image - mean) + shift)
        delta = np.clip(delta, -.15 * spread, .15 * spread)
        output = np.clip(image + delta, lo, hi).astype(np.float32)
        return output, "ineligible" if np.array_equal(output, image) else "applied", target

    def process(self, sample):
        image = np.asarray(sample["image"], dtype=np.float32)
        label = np.asarray(sample["label"])
        if image.ndim != 2 or image.shape != label.shape:
            raise ValueError("Expected image and label with matching [H,W] shapes")
        if not np.isfinite(image).all() or not np.isfinite(label).all():
            raise ValueError("Non-finite input")
        if (label < 0).any() or (label >= self.num_classes).any() or (label != np.floor(label)).any():
            raise ValueError("Invalid integer class ID")
        label = label.astype(np.int64, copy=True)
        mix_report = sample.get('region_mix_report')
        if self.preset == 'v4':
            if mix_report is None or set(mix_report) != set(blank_report()):
                raise RuntimeError('V4 requires the training-only RegionMixDataset path')
            if (array_hash(image, np.float32) != mix_report['mix_image_sha256'] or
                    array_hash(label, np.int64) != mix_report['mix_label_sha256']):
                raise RuntimeError('Mixed image/label changed before the V1 transform')
            event = np.asarray(mix_report['mix_event'])
            if event.shape != (5,) or event[0] != 1 or event[1] != event[2:].sum():
                raise RuntimeError('Invalid region-mix event accounting')
        else:
            mix_report = blank_report()
        original_counts = np.bincount(label.ravel(), minlength=self.num_classes)
        image, label = self._resize(image, label, self.output_size, image_order=3)
        reference = label.copy()
        before_counts = np.bincount(label.ravel(), minlength=self.num_classes)
        events = np.zeros((len(OPS), len(STATES)), dtype=np.int64)
        events[:, 0] = 1
        events[OPS.index('region_mix')] = mix_report['mix_event']
        retention_sum = np.zeros(self.num_classes, dtype=float)
        retention_count = np.zeros(self.num_classes, dtype=np.int64)
        detail = dict(target=-1, trials=0, zoom=1.)
        local_target = -1

        def record(name, state):
            row = events[OPS.index(name)]
            row[1] = 1
            row[STATES.index(state)] = 1

        def spatial_retention(old, new, scale):
            a = np.bincount(old.ravel(), minlength=self.num_classes)
            b = np.bincount(new.ravel(), minlength=self.num_classes)
            present = a > 0
            present[0] = False
            retention_sum[present] += b[present] / (a[present] * scale)
            retention_count[present] += 1

        crop_name = "adaptive_crop" if self.probabilities["adaptive_crop"] else "crop"
        # Exactly the three original V1 probability draws, in their old order.
        if np.random.random() < self.probabilities[crop_name]:
            old = label
            image, label, state, detail = self._crop(image, label, crop_name == "adaptive_crop")
            record(crop_name, state)
            if state == "applied":
                spatial_retention(old, label, detail["zoom"] ** 2)

        if np.random.random() < self.probabilities["rotate"]:
            angle = np.random.uniform(-12., 12.)
            cl = ndimage.rotate(label, angle, reshape=False, order=0,
                                mode="constant", cval=0, prefilter=False)
            acceptable = self._retains_organs(label, cl)
            if crop_name == "adaptive_crop":
                # Also protect cumulative retention after crop + rotation.
                acceptable = acceptable and self._retains_organs(reference, cl, detail["zoom"] ** 2)
            if acceptable:
                ci = ndimage.rotate(image, angle, reshape=False, order=1,
                                    mode="constant", cval=float(image.min()), prefilter=False)
                state = "ineligible" if np.array_equal(image, ci) and np.array_equal(label, cl) else "applied"
                if state == "applied":
                    spatial_retention(label, cl, 1.)
                image, label = ci, cl
                record("rotate", state)
            else:
                record("rotate", "rejected")

        if np.random.random() < self.probabilities["intensity"]:
            lo, hi, spread = float(image.min()), float(image.max()), float(image.std())
            if hi > lo and spread > 1e-8:
                previous = image
                mean = float(image.mean())
                image = (image - mean) * np.random.uniform(.9, 1.1) + mean
                image += np.random.uniform(-.05, .05) * spread
                if np.random.random() < .5:
                    sigma = np.random.uniform(.01, .03) * spread
                    image += np.random.normal(0, sigma, image.shape).astype(np.float32)
                image = np.clip(image, lo, hi)
                record("intensity", "ineligible" if np.array_equal(image, previous) else "applied")
            else:
                record("intensity", "ineligible")

        # A disabled new branch must NOT consume a random number in V1 mode.
        p_local = self.probabilities["local_intensity"]
        if p_local > 0 and np.random.random() < p_local:
            image, state, local_target = self._local_intensity(image, label)
            record("local_intensity", state)

        image = np.ascontiguousarray(image, dtype=np.float32)
        label = np.ascontiguousarray(label, dtype=np.int64)
        after_counts = np.bincount(label.ravel(), minlength=self.num_classes)
        metadata = dict(
            image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            label_sha256=hashlib.sha256(label.tobytes()).hexdigest(),
            pipeline=np.array(1, dtype=np.int64), events=events,
            target=np.array(detail["target"], dtype=np.int64),
            local_target=np.array(local_target, dtype=np.int64),
            crop_zoom=np.array(detail["zoom"], dtype=float),
            crop_trials=np.array(detail["trials"], dtype=np.int64),
            before_counts=before_counts, after_counts=after_counts,
            resize_lost=((original_counts > 0) & (before_counts == 0)).astype(np.int64),
            retention_sum=retention_sum, retention_count=retention_count,
        )
        metadata.update(mix_report)
        return image, label, metadata

    def __call__(self, sample):
        import torch
        if self.preset == "baseline":
            from .dataset import RandomGenerator
            return RandomGenerator(self.output_size)(sample)
        image, label, metadata = self.process(sample)
        result = dict(sample)
        result.pop('region_mix_report', None)
        result.update(image=torch.from_numpy(image).unsqueeze(0),
                      label=torch.from_numpy(label).long(), aug_audit=metadata)
        return result
