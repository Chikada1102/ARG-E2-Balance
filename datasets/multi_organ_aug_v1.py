"""Online 2D multi-organ augmentation for JCruan519/VM-UNet Synapse.

Save as datasets/multi_organ_aug.py. Input: image/label NumPy arrays [H,W].
Only attach to the training dataset; retain the official validation pipeline.
"""

import numpy as np
from scipy import ndimage


class MultiOrganRandomGenerator:
    def __init__(self, output_size, p_crop=0.35, p_rotate=0.5,
                 p_intensity=0.5, min_retention=0.8):
        self.output_size = tuple(int(v) for v in output_size)
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError("output_size must contain two positive integers")
        for value in (p_crop, p_rotate, p_intensity):
            if not 0 <= value <= 1:
                raise ValueError("probabilities must be in [0, 1]")
        if not 0 < min_retention <= 1:
            raise ValueError("min_retention must be in (0, 1]")
        self.p_crop = p_crop
        self.p_rotate = p_rotate
        self.p_intensity = p_intensity
        self.min_retention = min_retention

    @staticmethod
    def _resize(image, label, size, image_order=1):
        if image.shape == tuple(size):
            return image.copy(), label.copy()
        factors = (size[0] / image.shape[0], size[1] / image.shape[1])
        image = ndimage.zoom(image, factors, order=image_order)
        label = ndimage.zoom(label, factors, order=0, prefilter=False)
        return image, label

    def _retains_organs(self, before, after, area_scale=1.0):
        # Check each organ separately so a large organ cannot hide a lost small one.
        ids, counts = np.unique(before, return_counts=True)
        for organ_id, count in zip(ids, counts):
            if organ_id == 0:
                continue
            remaining = np.count_nonzero(after == organ_id)
            required = self.min_retention * count * area_scale
            if remaining == 0 or remaining < required:
                return False
        return True

    def _organ_crop(self, image, label):
        ids = np.unique(label)
        ids = ids[ids != 0]
        if ids.size == 0:
            return image, label
        h, w = label.shape
        # Uniform over classes PRESENT IN THIS SLICE, rather than foreground pixels.
        organ_id = np.random.choice(ids)
        coordinates = np.argwhere(label == organ_id)
        for _ in range(5):
            cy, cx = coordinates[np.random.randint(len(coordinates))]
            fraction = np.random.uniform(0.85, 1.0)
            ch = max(1, int(round(h * fraction)))
            cw = max(1, int(round(w * fraction)))
            top = int(np.clip(cy - ch // 2, 0, h - ch))
            left = int(np.clip(cx - cw // 2, 0, w - cw))
            crop_label = label[top:top + ch, left:left + cw]
            if not self._retains_organs(label, crop_label):
                continue
            crop_image = image[top:top + ch, left:left + cw]
            new_image, new_label = self._resize(crop_image, crop_label, (h, w))
            # Account for increased pixel area after enlarging the crop.
            if self._retains_organs(label, new_label, h * w / (ch * cw)):
                return new_image, new_label
        return image, label

    def augment_numpy(self, image, label):
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label)
        if image.ndim != 2 or image.shape != label.shape:
            raise ValueError("Expected image and label with the same [H,W] shape")
        if not np.isfinite(image).all() or not np.isfinite(label).all():
            raise ValueError("image/label contains NaN or Inf")
        if (label < 0).any() or not np.equal(label, np.floor(label)).all():
            raise ValueError("label must contain nonnegative integer class IDs")
        label = label.astype(np.int64, copy=True)
        # Same initial resizing interpolation as the official RandomGenerator.
        image, label = self._resize(image, label, self.output_size, image_order=3)

        if np.random.random() < self.p_crop:
            image, label = self._organ_crop(image, label)

        if np.random.random() < self.p_rotate:
            angle = np.random.uniform(-12.0, 12.0)
            new_label = ndimage.rotate(
                label, angle, reshape=False, order=0,
                mode="constant", cval=0, prefilter=False)
            if self._retains_organs(label, new_label):
                image = ndimage.rotate(
                    image, angle, reshape=False, order=1,
                    mode="constant", cval=float(image.min()), prefilter=False)
                label = new_label

        # Intensity changes affect only the image; preserve its original scale.
        if np.random.random() < self.p_intensity:
            lo, hi = float(image.min()), float(image.max())
            spread = float(image.std())
            if hi > lo and spread > 1e-8:
                mean = float(image.mean())
                image = (image - mean) * np.random.uniform(0.9, 1.1) + mean
                image += np.random.uniform(-0.05, 0.05) * spread
                if np.random.random() < 0.5:
                    sigma = np.random.uniform(0.01, 0.03) * spread
                    image += np.random.normal(0, sigma, image.shape).astype(np.float32)
                image = np.clip(image, lo, hi)

        return (np.ascontiguousarray(image, dtype=np.float32),
                np.ascontiguousarray(label, dtype=np.int64))

    def __call__(self, sample):
        import torch
        image, label = self.augment_numpy(sample["image"], sample["label"])
        result = dict(sample)
        result["image"] = torch.from_numpy(image).unsqueeze(0)
        result["label"] = torch.from_numpy(label).long()
        return result
