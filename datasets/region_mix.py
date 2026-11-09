"""Online cross-case ROI mixing, restricted to the existing training list.

Default: replace a visible organ with a matched donor organ and its context,
keeping all other recipient organs and a small surrounding band unchanged.
No PyTorch dependency is required until the final dataset transform is called.
"""
import hashlib
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
import numpy as np
from scipy import ndimage
from .multi_organ_aug_v1 import MultiOrganRandomGenerator as OriginalV1
from .region_mix_refinement import (REFINEMENTS, REFINE_METRICS, TargetBalance,
                                    refine_patch, background_seam)

MIX_METRICS = ("mask_fraction", "target_area_ratio", "donor_target_retention",
               "recipient_target_replaced", "other_foreground_overwritten",
               "other_foreground_total", "donor_other_imported", "context_pixels")
REASONS = ("disabled", "not_triggered", "applied", "no_visible_target",
           "no_compatible_donor", "target_clipped", "area_ratio",
           "target_too_small", "no_context", "unchanged", "max_attempts")
TRAIN_NAME = re.compile(r"^(case\d+)_slice\d+(?:\.npz)?$")


def array_hash(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def blank_report(image=None, label=None):
    return dict(mix_version=np.int64(1), mix_event=np.array([1, 0, 0, 0, 0], np.int64),
                mix_target=np.int64(-1), mix_attempts=np.int64(0), mix_reason=np.int64(0),
                mix_anchor_case="", mix_donor_case="", mix_anchor_name="", mix_donor_name="",
                mix_metrics=np.zeros(len(MIX_METRICS), np.float64),
                mix_refinement=np.int64(0), mix_selection_epoch=np.int64(0),
                mix_target_multiplier=np.float64(1.),
                mix_balance_weights=np.ones(9, np.float64),
                mix_refine_metrics=np.zeros(len(REFINE_METRICS), np.float64),
                mix_original_counts=(np.bincount(np.asarray(label, np.int64).ravel(), minlength=9)
                                     if label is not None else np.zeros(9, np.int64)),
                mix_image_sha256=array_hash(image, np.float32) if image is not None else "",
                mix_label_sha256=array_hash(label, np.int64) if label is not None else "")


def validated_arrays(image, label, classes=9):
    image, label = np.asarray(image, np.float32), np.asarray(label)
    if image.ndim != 2 or image.shape != label.shape:
        raise ValueError("Region mix expects matching 2D image/label arrays")
    if not np.isfinite(image).all() or not np.isfinite(label).all():
        raise ValueError("Non-finite training NPZ")
    if (label < 0).any() or (label >= classes).any() or (label != np.floor(label)).any():
        raise ValueError("Expected integer labels 0..%d" % (classes - 1))
    return image, label.astype(np.int64, copy=False)


def label_descriptor(label, output_size):
    h, w = label.shape
    scaled = ndimage.zoom(label, (output_size[0] / h, output_size[1] / w),
                          order=0, prefilter=False) if label.shape != tuple(output_size) else label
    output_counts = np.bincount(scaled.ravel(), minlength=9)
    classes = {}
    for c in np.unique(label):
        if c == 0:
            continue
        yy, xx = np.where(label == c)
        y0, y1, x0, x1 = int(yy.min()), int(yy.max() + 1), int(xx.min()), int(xx.max() + 1)
        classes[str(int(c))] = dict(pixels=int(len(yy)), output_pixels=int(output_counts[c]),
                                   bbox=[y0, x0, y1, x1],
                                   area=float(len(yy) / (h * w)),
                                   cy=float(yy.mean() / h), cx=float(xx.mean() / w),
                                   aspect=float(((y1 - y0) / h) / ((x1 - x0) / w)))
    return dict(shape=[h, w], classes=classes)


class TrainDonorPool:
    def __init__(self, base_dir, list_dir, allowed_cases, output_size=(224, 224),
                 cache_path=None, logger=None, cache_images=2):
        self.base_dir = str(Path(base_dir).resolve())
        self.output_size = tuple(output_size)
        self.allowed_cases = frozenset(allowed_cases)
        self.cache_images = max(0, int(cache_images))
        self._images = OrderedDict()
        list_path = Path(list_dir) / 'train.txt'
        content = list_path.read_bytes()
        self.names = [x.strip() for x in content.decode('utf-8-sig').splitlines() if x.strip()]
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Training list is empty or contains duplicate slice entries")
        self.cases, file_stats = [], []
        for name in self.names:
            match = TRAIN_NAME.fullmatch(name)
            if not match or match.group(1) not in self.allowed_cases:
                raise ValueError("Donor slice is not an allowed training case: " + name)
            self.cases.append(match.group(1))
            stat = self._path(name).stat()
            file_stats.append([int(stat.st_size), int(stat.st_mtime_ns)])
        if set(self.cases) != set(self.allowed_cases):
            raise ValueError("Donor pool does not match the existing training-case list")
        self.file_stats = file_stats
        fingerprint = dict(version=1, base_dir=self.base_dir, output_size=list(output_size),
                           list_sha256=hashlib.sha256(content).hexdigest(),
                           names=self.names, file_stats=file_stats)
        self.fingerprint = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        records = None
        cache = Path(cache_path) if cache_path else None
        if cache and cache.is_file():
            try:
                saved = json.loads(cache.read_text())
                if saved.get('fingerprint') == self.fingerprint and len(saved['records']) == len(self.names):
                    records = saved['records']
            except (ValueError, KeyError, TypeError):
                records = None
        if records is None:
            records = []
            for index, name in enumerate(self.names):
                _, label = self._load_file(index)
                records.append(label_descriptor(label, self.output_size))
                if logger and ((index + 1) % 200 == 0 or index + 1 == len(self.names)):
                    logger.info('REGION_MIX_INDEX %d/%d training slices', index + 1, len(self.names))
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_name(cache.name + '.tmp.' + str(os.getpid()))
                temporary.write_text(json.dumps(dict(fingerprint=self.fingerprint, records=records)))
                os.replace(str(temporary), str(cache))
        self.records = records
        self.class_frequency = np.array([sum(str(c) in r['classes'] for r in records)
                                         for c in range(9)], dtype=np.int64)
        self.case_array = np.asarray(self.cases)
        self.tables = {}
        for c in range(1, 9):
            indices = [i for i, r in enumerate(records)
                       if str(c) in r['classes'] and r['classes'][str(c)]['output_pixels'] >= 8]
            self.tables[c] = dict(indices=np.array(indices, np.int64),
                                 features=np.array([[records[i]['classes'][str(c)][k]
                                                     for k in ('area', 'cy', 'cx', 'aspect')]
                                                    for i in indices], float).reshape(-1, 4))
        if logger:
            logger.info('REGION_MIX_POOL training_cases=%d slices=%d class_positive_slices=%s fingerprint=%s',
                        len(self.allowed_cases), len(self.names), self.class_frequency.tolist(), self.fingerprint)

    def _path(self, name):
        return Path(self.base_dir) / (name if name.endswith('.npz') else name + '.npz')

    def _load_file(self, index):
        path = self._path(self.names[index])
        stat = path.stat()
        if [int(stat.st_size), int(stat.st_mtime_ns)] != self.file_stats[index]:
            raise RuntimeError("Training file changed after donor index creation: " + str(path))
        with np.load(path, allow_pickle=False) as data:
            if 'image' not in data or 'label' not in data:
                raise ValueError("Expected image/label keys in " + str(path))
            image, label = validated_arrays(data['image'], data['label'])
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def load(self, index):
        if not 0 <= index < len(self.names) or self.cases[index] not in self.allowed_cases:
            raise RuntimeError("Donor is outside the training pool")
        if index in self._images:
            value = self._images.pop(index)
            self._images[index] = value
            return value
        value = self._load_file(index)
        for array in value:
            array.setflags(write=False)
        if self.cache_images:
            self._images[index] = value
            while len(self._images) > self.cache_images:
                self._images.popitem(last=False)
        return value

    def candidates(self, anchor_index, target):
        table = self.tables[target]
        indices, features = table['indices'], table['features']
        if not len(indices):
            return indices
        a = self.records[anchor_index]['classes'][str(target)]
        area_ratio = features[:, 0] / a['area']
        aspect_ratio = features[:, 3] / a['aspect']
        position = np.hypot(features[:, 1] - a['cy'], features[:, 2] - a['cx'])
        valid = ((self.case_array[indices] != self.cases[anchor_index]) &
                 (area_ratio >= 2 / 3) & (area_ratio <= 1.5) &
                 (aspect_ratio >= 2 / 3) & (aspect_ratio <= 1.5) & (position <= .15))
        candidates = indices[valid]
        anchor_context = set(self.records[anchor_index]['classes'])
        return np.array([i for i in candidates
                         if len(anchor_context & set(self.records[i]['classes'])) /
                         len(anchor_context | set(self.records[i]['classes'])) >= .5], np.int64)


class OrganRegionMixer:
    def __init__(self, pool, probability=.2, strategy='protected', max_attempts=8,
                 refinement='none'):
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Mix probability must be in [0,1]")
        if strategy not in ('protected', 'organ', 'cutmix'):
            raise ValueError("Expected protected, organ or cutmix strategy")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if refinement not in REFINEMENTS or (refinement != 'none' and strategy != 'protected'):
            raise ValueError('V4.1 refinements require the protected organ strategy')
        self.pool, self.probability, self.strategy = pool, float(probability), strategy
        self.max_attempts = int(max_attempts)
        self.refinement = refinement
        self.balance_enabled = refinement in ('balance', 'both')
        self.seam_enabled = refinement in ('seam', 'both')
        frequency = (np.array([0] + [len(pool.tables[c]['indices']) for c in range(1, 9)])
                     if self.balance_enabled else None)
        self.balance = TargetBalance(frequency, pool.fingerprint) if self.balance_enabled else None
        self.selection_epoch = 0
        self.active_weights = np.ones(9, np.float64)
        self._last_refine_metrics = np.zeros(len(REFINE_METRICS), np.float64)

    def set_epoch(self, epoch):
        if int(epoch) < 1 or (self.balance and self.balance.completed_epochs != int(epoch) - 1):
            raise RuntimeError('Training epoch and balance feedback state differ')
        self.selection_epoch = int(epoch)
        self.active_weights = self.balance.weights() if self.balance else np.ones(9, np.float64)

    def state_dict(self):
        return dict(version=1, refinement=self.refinement,
                    balance=self.balance.state_dict() if self.balance else None)

    def load_state_dict(self, state, saved_epoch):
        if (not isinstance(state, dict) or state.get('version') != 1 or
                state.get('refinement') != self.refinement):
            raise RuntimeError('Missing or mismatched region-mix refinement state')
        if self.balance:
            self.balance.load_state_dict(state.get('balance'))
            if self.balance.completed_epochs != int(saved_epoch):
                raise RuntimeError('Saved balance state and checkpoint epoch differ')
        elif state.get('balance') is not None:
            raise RuntimeError('Unexpected balance state in a non-balanced experiment')

    def target_probabilities(self, visible):
        frequencies = self.pool.class_frequency[visible].astype(float)
        weights = np.minimum(np.sqrt(frequencies.max() / frequencies), 3.)
        if self.balance_enabled:
            weights = weights * self.active_weights[visible]
        return weights / weights.sum()

    def _roi(self, bbox, shape):
        y0, x0, y1, x1 = bbox
        h, w = shape
        # Include real context, with a minimum margin of four output pixels.
        py = max(int(round(.25 * (y1 - y0))), int(np.ceil(4 * h / self.pool.output_size[0])))
        px = max(int(round(.25 * (x1 - x0))), int(np.ceil(4 * w / self.pool.output_size[1])))
        return max(0, y0 - py), max(0, x0 - px), min(h, y1 + py), min(w, x1 + px)

    @staticmethod
    def _protection(label, target, radius):
        other = (label != 0) & (label != target)
        band = ndimage.binary_dilation(other, iterations=radius)
        # Keep the target itself eligible even when it touches another organ.
        return other | (band & (label == 0))

    def _organ_candidate(self, image, label, anchor_index, donor_index, target):
        donor_image, donor_label = self.pool.load(donor_index)
        ad = self.pool.records[anchor_index]['classes'][str(target)]
        dd = self.pool.records[donor_index]['classes'][str(target)]
        ay0, ax0, ay1, ax1 = self._roi(ad['bbox'], label.shape)
        dy0, dx0, dy1, dx1 = self._roi(dd['bbox'], donor_label.shape)
        shape = (ay1 - ay0, ax1 - ax0)
        donor_patch, donor_mask = OriginalV1._resize(
            donor_image[dy0:dy1, dx0:dx1], donor_label[dy0:dy1, dx0:dx1], shape, image_order=3)
        anchor_patch = label[ay0:ay1, ax0:ax1]
        mask = np.ones(shape, bool)
        if self.strategy == 'protected':
            radius = max(1, int(round(2 * min(label.shape) / min(self.pool.output_size))))
            protect = self._protection(label, target, radius)[ay0:ay1, ax0:ax1]
            donor_protect = self._protection(donor_mask, target, radius)
            mask &= ~(protect | donor_protect)
        donor_target = donor_mask == target
        target_total = int(donor_target.sum())
        recipient_total = int(np.count_nonzero(label == target))
        donor_retained = float((donor_target & mask).sum() / max(1, target_total))
        recipient_replaced = float(((anchor_patch == target) & mask).sum() / max(1, recipient_total))
        if donor_retained < .95 or recipient_replaced < .95:
            return None, 'target_clipped'
        context_pixels = int((mask & (donor_mask == 0) & (anchor_patch == 0)).sum())
        min_context = max(4, int(np.ceil(16 * label.size / np.prod(self.pool.output_size))))
        if context_pixels < min_context:
            return None, 'no_context'
        new_label = label.copy()
        target_patch = new_label[ay0:ay1, ax0:ax1]
        target_patch[mask] = donor_mask[mask]
        target_area_ratio = np.count_nonzero(new_label == target) / float(max(1, recipient_total))
        if not .75 <= target_area_ratio <= 4 / 3:
            return None, 'area_ratio'
        scaled = ndimage.zoom(new_label, (self.pool.output_size[0] / label.shape[0],
                                         self.pool.output_size[1] / label.shape[1]),
                              order=0, prefilter=False)
        if np.count_nonzero(scaled == target) < 8:
            return None, 'target_too_small'
        new_image = image.copy()
        view = new_image[ay0:ay1, ax0:ax1]
        view[mask] = donor_patch[mask]
        if self.seam_enabled:
            before = new_image.copy()
            spread = max(float(image.std()), float(np.ptp(np.percentile(image, [5, 95]))))
            refined, refined_metrics = refine_patch(
                image[ay0:ay1, ax0:ax1], donor_patch, anchor_patch, donor_mask, mask,
                (self.pool.output_size[0] / label.shape[0],
                 self.pool.output_size[1] / label.shape[1]), spread)
            view[:] = refined
            if (np.array_equal(new_image, image) and np.array_equal(new_label, label) and
                    not np.array_equal(before, image)):
                # Preserve V4's donor acceptance/RNG if perfect gray matching
                # cancels the entire augmentation. Audit this exceptional fallback.
                new_image = before
                refined_metrics[2:6] = (0., 1., 0., 0.)
                refined_metrics[9] = 1.
            full_mask = np.zeros(label.shape, bool)
            full_mask[ay0:ay1, ax0:ax1] = mask
            refined_metrics[6:9] = background_seam(before, new_image, full_mask, label, new_label)
            self._last_refine_metrics = refined_metrics
        other = (anchor_patch != 0) & (anchor_patch != target)
        metrics = np.array([mask.sum() / label.size, target_area_ratio, donor_retained,
                            recipient_replaced, (other & mask).sum(),
                            ((label != 0) & (label != target)).sum(),
                            (mask & (donor_mask != 0) & (donor_mask != target)).sum(),
                            context_pixels], float)
        if self.strategy == 'protected':
            unchanged = (label != 0) & (label != target)
            if not np.array_equal(new_label[unchanged], label[unchanged]) or not np.array_equal(new_image[unchanged], image[unchanged]):
                raise RuntimeError("Protected recipient foreground changed")
            if metrics[4] != 0 or metrics[6] != 0:
                raise RuntimeError("Unexpected non-target organ import/overwrite")
        return (new_image, new_label, metrics), 'applied'

    def _cutmix(self, image, label, donor_index):
        donor_image, donor_label = self.pool.load(donor_index)
        donor_image, donor_label = OriginalV1._resize(donor_image, donor_label, label.shape, image_order=3)
        h, w = label.shape
        length = np.sqrt(1 - np.random.beta(.5, .5))
        cy, cx = np.random.randint(h), np.random.randint(w)
        ch, cw = max(1, round(h * length)), max(1, round(w * length))
        y0, x0 = max(0, cy - ch // 2), max(0, cx - cw // 2)
        y1, x1 = min(h, cy + (ch + 1) // 2), min(w, cx + (cw + 1) // 2)
        new_image, new_label = image.copy(), label.copy()
        new_image[y0:y1, x0:x1] = donor_image[y0:y1, x0:x1]
        new_label[y0:y1, x0:x1] = donor_label[y0:y1, x0:x1]
        metrics = np.zeros(len(MIX_METRICS), float)
        metrics[0] = (y1 - y0) * (x1 - x0) / label.size
        metrics[4] = np.count_nonzero(label[y0:y1, x0:x1])
        metrics[5] = np.count_nonzero(label)
        metrics[6] = np.count_nonzero(donor_label[y0:y1, x0:x1])
        metrics[7] = np.count_nonzero(donor_label[y0:y1, x0:x1] == 0)
        return new_image, new_label, metrics

    def process(self, sample, anchor_index):
        image, label = validated_arrays(sample['image'], sample['label'])
        if not 0 <= anchor_index < len(self.pool.names):
            raise IndexError("Anchor is outside the training list")
        expected = self.pool.records[anchor_index]
        if list(label.shape) != expected['shape']:
            raise RuntimeError("Anchor shape differs from indexed training file")
        counts = np.bincount(label.ravel(), minlength=9)
        if any(int(counts[c]) != int(expected['classes'].get(str(c), {}).get('pixels', 0)) for c in range(1, 9)):
            raise RuntimeError("Anchor labels differ from indexed training file")
        report = blank_report(image, label)
        report.update(mix_refinement=np.int64(REFINEMENTS.index(self.refinement)),
                      mix_selection_epoch=np.int64(self.selection_epoch),
                      mix_balance_weights=self.active_weights.copy())
        report['mix_anchor_case'] = self.pool.cases[anchor_index]
        report['mix_anchor_name'] = self.pool.names[anchor_index]
        if self.probability == 0:
            return dict(sample, region_mix_report=report)
        if np.random.random() >= self.probability:
            report['mix_reason'] = np.int64(REASONS.index('not_triggered'))
            return dict(sample, region_mix_report=report)
        report['mix_event'][1] = 1
        target = -1
        if self.strategy == 'cutmix':
            candidates = np.flatnonzero(self.pool.case_array != self.pool.cases[anchor_index])
        else:
            visible = [int(c) for c, d in expected['classes'].items() if d['output_pixels'] >= 8]
            if not visible:
                report['mix_event'][4] = 1
                report['mix_reason'] = np.int64(REASONS.index('no_visible_target'))
                return dict(sample, region_mix_report=report)
            target = int(np.random.choice(visible, p=self.target_probabilities(visible)))
            report['mix_target'] = np.int64(target)
            report['mix_target_multiplier'] = np.float64(self.active_weights[target])
            candidates = self.pool.candidates(anchor_index, target)
        if not len(candidates):
            report['mix_event'][4] = 1
            report['mix_reason'] = np.int64(REASONS.index('no_compatible_donor'))
            return dict(sample, region_mix_report=report)
        candidates = np.random.choice(candidates, size=min(self.max_attempts, len(candidates)), replace=False)
        last_reason = 'max_attempts'
        for donor_index in candidates:
            self._last_refine_metrics = np.zeros(len(REFINE_METRICS), np.float64)
            donor_index = int(donor_index)
            report['mix_attempts'] += 1
            if self.pool.cases[donor_index] == self.pool.cases[anchor_index]:
                raise RuntimeError("Region mixing used the same patient")
            if self.strategy == 'cutmix':
                result, reason = self._cutmix(image, label, donor_index), 'applied'
            else:
                result, reason = self._organ_candidate(image, label, anchor_index, donor_index, target)
            if result is None:
                last_reason = reason
                continue
            new_image, new_label, metrics = result
            if np.array_equal(new_image, image) and np.array_equal(new_label, label):
                last_reason = 'unchanged'
                continue
            report.update(mix_donor_case=self.pool.cases[donor_index],
                          mix_donor_name=self.pool.names[donor_index], mix_metrics=metrics,
                          mix_refine_metrics=self._last_refine_metrics.copy(),
                          mix_image_sha256=array_hash(new_image, np.float32),
                          mix_label_sha256=array_hash(new_label, np.int64),
                          mix_reason=np.int64(REASONS.index('applied')))
            report['mix_event'][2] = 1
            return dict(sample, image=new_image, label=new_label, region_mix_report=report)
        report['mix_event'][3] = 1
        report['mix_reason'] = np.int64(REASONS.index(last_reason))
        return dict(sample, region_mix_report=report)


class RegionMixDataset:
    def __init__(self, raw_dataset, transform, mixer):
        if getattr(raw_dataset, 'transform', None) is not None or getattr(raw_dataset, 'split', 'train') != 'train':
            raise ValueError("Wrap an untransformed training dataset only")
        if len(raw_dataset) != len(mixer.pool.names):
            raise ValueError("Dataset length differs from the existing training list")
        sample_list = getattr(raw_dataset, 'sample_list', mixer.pool.names)
        if [str(x).strip() for x in sample_list] != mixer.pool.names:
            raise ValueError("Dataset order differs from donor-index train.txt")
        if transform.preset != 'v4' or transform.probabilities['region_mix'] != mixer.probability:
            raise ValueError("Mixer probability and audited transform differ")
        self.raw_dataset, self.transform, self.mixer = raw_dataset, transform, mixer
        transform.mix_allowed_cases = mixer.pool.allowed_cases
        transform.mix_strategy = mixer.strategy
        self._sync_refinement()

    def _sync_refinement(self):
        self.transform.mix_refinement = self.mixer.refinement
        self.transform.mix_selection_epoch = self.mixer.selection_epoch
        self.transform.mix_balance_weights = self.mixer.active_weights.copy()

    def set_mix_epoch(self, epoch):
        self.mixer.set_epoch(epoch)
        self._sync_refinement()

    def finish_mix_epoch(self, audit, epoch, logger, directory):
        if not self.mixer.balance:
            return
        if audit.samples != len(self):
            raise RuntimeError('Balance feedback requires one complete training-list pass')
        used = self.mixer.active_weights.copy()
        self.mixer.balance.update(epoch, audit.selected, audit.applied)
        next_weights = self.mixer.balance.weights()
        # Imported at call time to avoid the augmentation/audit module cycle.
        from innovation_audit import append_csv
        balance = self.mixer.balance
        rows = [dict(epoch=int(epoch), class_id=c,
                     visible_positive_slices=int(balance.frequency[c]),
                     selected=int(audit.selected[c]), applied=int(audit.applied[c]),
                     cumulative_selected=int(balance.selected[c]), cumulative_applied=int(balance.applied[c]),
                     smoothed_applications_per_positive=float(balance.rates()[c]),
                     multiplier_used=float(used[c]), multiplier_next=float(next_weights[c]))
                for c in range(1, 9)]
        append_csv(os.path.join(directory, 'region_mix_balance.csv'), rows)
        logger.info('REGION_MIX_BALANCE epoch=%d used=%s next=%s', epoch, used.tolist(), next_weights.tolist())

    def __len__(self):
        return len(self.raw_dataset)

    def mixed_sample(self, index):
        sample = self.raw_dataset[index]
        if sample.get('case_name', self.mixer.pool.names[index]) != self.mixer.pool.names[index]:
            raise RuntimeError("Training sample identity differs from train.txt order")
        return self.mixer.process(sample, index)

    def process_numpy(self, index):
        return self.transform.process(self.mixed_sample(index))

    def __getitem__(self, index):
        return self.transform(self.mixed_sample(index))
