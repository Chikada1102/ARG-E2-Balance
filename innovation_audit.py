"""Consumer-side augmentation accounting and requested module path checks."""
import csv
import hashlib
import os
import numpy as np
from collections import Counter
from datasets.multi_organ_aug import OPS, STATES
from datasets.region_mix import MIX_METRICS, REASONS
from datasets.region_mix_refinement import REFINEMENTS, REFINE_METRICS


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def append_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class AugmentationAudit:
    def __init__(self, transform, strict=True):
        self.transform, self.strict = transform, strict
        self.events = np.zeros((len(OPS), len(STATES)), dtype=np.int64)
        self.samples, self.trials = 0, 0
        self.target_counts = np.zeros(transform.num_classes, dtype=np.int64)
        self.applied_target_counts = self.target_counts.copy()
        self.before = self.target_counts.copy()
        self.after = self.target_counts.copy()
        self.lost = self.target_counts.copy()
        self.resize_lost = self.target_counts.copy()
        self.retention_sum = np.zeros(transform.num_classes)
        self.retention_count = self.target_counts.copy()
        self.positive_samples = self.target_counts.copy()
        self.local_target_counts = self.target_counts.copy()
        self.crop_zoom_sum = np.zeros(transform.num_classes)
        self.region = RegionMixAudit(transform)

    def consume(self, batch):
        if self.transform.preset == "baseline":
            return
        if "aug_audit" not in batch:
            raise RuntimeError("Missing aug_audit: augmentation or metadata path was bypassed")
        meta = {k: _numpy(v) for k, v in batch["aug_audit"].items()
                if k not in ("image_sha256", "label_sha256")}
        count = len(meta["pipeline"])
        if not np.all(meta["pipeline"] == 1):
            raise RuntimeError("Invalid augmentation pipeline marker")
        events = meta["events"]
        if events.shape != (count, len(OPS), len(STATES)) or not np.all(events[:, :, 0] == 1):
            raise RuntimeError("An augmentation branch was not checked for each sample")
        if not np.array_equal(events[:, :, 1], events[:, :, 2:].sum(axis=2)):
            raise RuntimeError("Inconsistent augmentation event accounting")
        labels = _numpy(batch["label"])
        if labels.shape[0] != count:
            raise RuntimeError("Audit batch length does not match labels")
        actual = np.stack([np.bincount(y.astype(np.int64).ravel(),
                                      minlength=self.transform.num_classes) for y in labels])
        if not np.array_equal(actual, meta["after_counts"]):
            raise RuntimeError("Labels changed AFTER augmentation auditing")
        images = _numpy(batch["image"])
        checksums = batch["aug_audit"]["image_sha256"]
        if images.shape != (count, 1, *labels.shape[1:]) or len(checksums) != count:
            raise RuntimeError("Augmentation image audit shape mismatch")
        for image, checksum in zip(images, checksums):
            actual_checksum = hashlib.sha256(np.ascontiguousarray(image[0], dtype=np.float32).tobytes()).hexdigest()
            if actual_checksum != str(checksum):
                raise RuntimeError("Image changed AFTER augmentation auditing")
        for label, checksum in zip(labels, batch["aug_audit"]["label_sha256"]):
            actual_checksum = hashlib.sha256(np.ascontiguousarray(label, dtype=np.int64).tobytes()).hexdigest()
            if actual_checksum != str(checksum):
                raise RuntimeError("Label spatial arrangement changed AFTER augmentation auditing")
        self.region.consume(meta)
        self.samples += count
        self.events += events.sum(axis=0)
        self.trials += int(meta["crop_trials"].sum())
        targets = meta["target"].ravel()
        self.target_counts += np.bincount(targets[targets >= 0], minlength=len(self.target_counts))
        crop_indices = [OPS.index(name) for name in ("crop", "adaptive_crop")]
        crop_applied = events[:, crop_indices, 2].sum(axis=1) > 0
        applied_targets = targets[(targets >= 0) & crop_applied]
        self.applied_target_counts += np.bincount(applied_targets, minlength=len(self.target_counts))
        zooms = meta["crop_zoom"].ravel()
        for target, zoom, applied in zip(targets, zooms, crop_applied):
            if target >= 0 and applied:
                self.crop_zoom_sum[int(target)] += float(zoom)
        local_targets = meta["local_target"].ravel()
        local_applied = events[:, OPS.index("local_intensity"), 2] == 1
        selected = local_targets[(local_targets >= 0) & local_applied]
        self.local_target_counts += np.bincount(selected, minlength=len(self.target_counts))
        self.positive_samples += (meta["before_counts"] > 0).sum(axis=0)
        self.before += meta["before_counts"].sum(axis=0)
        self.after += actual.sum(axis=0)
        self.lost += ((meta["before_counts"] > 0) & (actual == 0)).sum(axis=0)
        self.resize_lost += meta["resize_lost"].sum(axis=0)
        self.retention_sum += meta["retention_sum"].sum(axis=0)
        self.retention_count += meta["retention_count"].sum(axis=0)

    def finish(self, epoch, logger, directory):
        if self.transform.preset == "baseline":
            return
        if self.samples == 0:
            raise RuntimeError("No consumed augmented training samples")
        rows, failures = [], []
        for index, name in enumerate(OPS):
            checked, triggered, applied, rejected, ineligible = map(int, self.events[index])
            probability = self.transform.probabilities[name]
            row = dict(epoch=epoch, op=name, probability=probability, samples=self.samples,
                       checked=checked, triggered=triggered, applied=applied,
                       rejected=rejected, ineligible=ineligible, not_triggered=checked-triggered,
                       applied_rate=applied/self.samples)
            rows.append(row)
            logger.info("AUG_AUDIT " + str(row))
            if probability > 0 and applied == 0:
                failures.append(name)
        append_csv(os.path.join(directory, "augmentation_epoch.csv"), rows)
        organs = []
        for c in range(1, len(self.before)):
            organs.append(dict(epoch=epoch, class_id=c, crop_target_count=int(self.target_counts[c]),
                               crop_applied_target_count=int(self.applied_target_counts[c]),
                               before_pixels=int(self.before[c]), after_pixels=int(self.after[c]),
                               lost_samples=int(self.lost[c]), initial_resize_lost_samples=int(self.resize_lost[c]),
                               accepted_spatial_steps=int(self.retention_count[c]),
                               mean_area_retention=(self.retention_sum[c]/self.retention_count[c]
                                                    if self.retention_count[c] else ""),
                               positive_samples=int(self.positive_samples[c]),
                               local_intensity_applied_target_count=int(self.local_target_counts[c]),
                               mean_crop_linear_zoom=(self.crop_zoom_sum[c]/self.applied_target_counts[c]
                                                      if self.applied_target_counts[c] else ""),
                               region_mix_target_count=int(self.region.selected[c]),
                               region_mix_applied_target_count=int(self.region.applied[c]),
                               original_positive_samples=(int(self.region.original_positive[c])
                                                          if self.transform.preset == 'v4'
                                                          else int(self.positive_samples[c]))))
        append_csv(os.path.join(directory, "augmentation_organs.csv"), organs)
        self.region.finish(epoch, logger, directory)
        logger.info("AUG_AUDIT consumed=%d crop_candidate_trials=%d", self.samples, self.trials)
        if failures and self.strict:
            raise RuntimeError("Enabled augmentation had ZERO actual applications this epoch: " +
                               ", ".join(failures) + ". Inspect audit CSV; do not silently bypass.")


class RegionMixAudit:
    def __init__(self, transform):
        self.transform = transform
        self.selected = np.zeros(transform.num_classes, np.int64)
        self.applied = self.selected.copy()
        self.original_positive = self.selected.copy()
        self.reasons = np.zeros(len(REASONS), np.int64)
        self.metric_sum = np.zeros(len(MIX_METRICS), float)
        self.pairs, self.pair_metrics = Counter(), {}
        self.samples = self.applied_samples = self.attempts = 0
        self.target_reasons = np.zeros((transform.num_classes, len(REASONS)), np.int64)
        self.refine_sum = np.zeros(len(REFINE_METRICS), np.float64)
        self.refine_changed_samples = self.feathered_samples = 0

    def consume(self, meta):
        if self.transform.preset != 'v4':
            return
        allowed = getattr(self.transform, 'mix_allowed_cases', None)
        if not allowed or not np.all(meta['mix_version'] == 1):
            raise RuntimeError('Missing training-only region-mix provenance')
        event = meta['mix_event']
        if not np.array_equal(event, meta['events'][:, OPS.index('region_mix')]):
            raise RuntimeError('Region-mix report differs from operation events')
        size = len(event)
        expected_mode = REFINEMENTS.index(getattr(self.transform, 'mix_refinement', 'none'))
        if (not np.all(meta['mix_refinement'] == expected_mode) or
                not np.all(meta['mix_selection_epoch'] == getattr(self.transform, 'mix_selection_epoch', 0))):
            raise RuntimeError('Worker refinement mode or epoch differs from the consumer')
        expected_weights = np.asarray(getattr(self.transform, 'mix_balance_weights', np.ones(9)))
        if (meta['mix_balance_weights'].shape != (size, 9) or
                not np.array_equal(meta['mix_balance_weights'], np.broadcast_to(expected_weights, (size, 9)))):
            raise RuntimeError('Worker balance weights are stale or mismatched')
        refinement = meta['mix_refine_metrics']
        if refinement.shape != (size, len(REFINE_METRICS)) or not np.isfinite(refinement).all():
            raise RuntimeError('Invalid refinement metrics')
        if meta['mix_metrics'].shape != (size, len(MIX_METRICS)) or not np.isfinite(meta['mix_metrics']).all():
            raise RuntimeError('Invalid region-mix metrics')
        reason = meta['mix_reason'].astype(int).ravel()
        if (reason < 0).any() or (reason >= len(REASONS)).any():
            raise RuntimeError('Unknown region-mix rejection reason')
        self.reasons += np.bincount(reason, minlength=len(REASONS))
        self.samples += size
        self.attempts += int(meta['mix_attempts'].sum())
        self.original_positive += (meta['mix_original_counts'] > 0).sum(axis=0)
        targets = meta['mix_target'].astype(int).ravel()
        if (targets < -1).any() or (targets >= self.transform.num_classes).any():
            raise RuntimeError('Invalid region-mix target class')
        self.selected += np.bincount(targets[targets >= 1], minlength=len(self.selected))
        for i in range(size):
            target = int(targets[i])
            if target >= 1:
                self.target_reasons[target, reason[i]] += 1
                if float(meta['mix_target_multiplier'][i]) != float(expected_weights[target]):
                    raise RuntimeError('Selected target did not use the recorded balance multiplier')
        for i in range(size):
            anchor = str(meta['mix_anchor_case'][i])
            if anchor not in allowed:
                raise RuntimeError('Region-mix anchor is outside the training cases')
            if event[i, 2] != 1:
                continue
            donor = str(meta['mix_donor_case'][i])
            if donor not in allowed or donor == anchor:
                raise RuntimeError('Region mixing used a non-training or same-case donor')
            metrics = meta['mix_metrics'][i]
            refined = refinement[i]
            if expected_mode in (2, 3):
                if (refined[0] != 1 or refined[1] < 1 or not .9 <= refined[3] <= 1.1 or
                        (refined[[1, 2, 5, 6, 7, 8]] < 0).any()):
                    raise RuntimeError('Applied region mix bypassed seam treatment')
            elif refined.any():
                raise RuntimeError('Seam treatment ran in a disabled experiment')
            if not 0 < metrics[0] <= 1:
                raise RuntimeError('Applied region mix has an empty or invalid mask')
            target = int(targets[i])
            if self.transform.mix_strategy != 'cutmix':
                if target < 1 or not .75 <= metrics[1] <= 4 / 3 or min(metrics[2:4]) < .95:
                    raise RuntimeError('Applied organ mix violated target-retention limits')
                self.applied[target] += 1
            if self.transform.mix_strategy == 'protected' and (metrics[4] != 0 or metrics[6] != 0):
                raise RuntimeError('Protected mix overwrote/imported a non-target organ')
            key = (anchor, donor, str(meta['mix_anchor_name'][i]), str(meta['mix_donor_name'][i]), target)
            self.pairs[key] += 1
            self.pair_metrics.setdefault(key, np.zeros(len(MIX_METRICS) + len(REFINE_METRICS), float))[:] += np.r_[metrics, refined]
            self.applied_samples += 1
            self.metric_sum += metrics
            self.refine_sum += refined
            self.refine_changed_samples += int(refined[2] > 0)
            self.feathered_samples += int(refined[5] > 0)

    def finish(self, epoch, logger, directory):
        if self.transform.preset != 'v4':
            return
        summary = dict(epoch=epoch, strategy=self.transform.mix_strategy, samples=self.samples,
                       applied=self.applied_samples, candidate_attempts=self.attempts,
                       applied_rate=self.applied_samples / max(1, self.samples))
        summary.update({'reason_' + name: int(self.reasons[i]) for i, name in enumerate(REASONS)})
        summary.update({'mean_' + name: (float(self.metric_sum[i] / self.applied_samples)
                                        if self.applied_samples and
                                        not (self.transform.mix_strategy == 'cutmix' and i in (1, 2, 3)) else '')
                        for i, name in enumerate(MIX_METRICS)})
        summary.update(refinement=getattr(self.transform, 'mix_refinement', 'none'),
                       selection_epoch=getattr(self.transform, 'mix_selection_epoch', 0),
                       refined_changed_samples=self.refine_changed_samples,
                       feathered_samples=self.feathered_samples)
        summary.update({'mean_' + name: float(self.refine_sum[i] / max(1, self.applied_samples))
                        for i, name in enumerate(REFINE_METRICS)})
        append_csv(os.path.join(directory, 'region_mix_epoch.csv'), [summary])
        rows = []
        for key, count in self.pairs.items():
            anchor, donor, anchor_name, donor_name, target = key
            row = dict(epoch=epoch, anchor_case=anchor, donor_case=donor,
                       anchor_slice=anchor_name, donor_slice=donor_name,
                       target_class=target, applications=count)
            row.update({'mean_' + name: (float(self.pair_metrics[key][i] / count)
                                        if not (self.transform.mix_strategy == 'cutmix' and i in (1, 2, 3)) else '')
                        for i, name in enumerate(MIX_METRICS)})
            row.update({'mean_' + name: float(self.pair_metrics[key][len(MIX_METRICS) + i] / count)
                        for i, name in enumerate(REFINE_METRICS)})
            rows.append(row)
        append_csv(os.path.join(directory, 'region_mix_pairs.csv'), rows)
        target_rows = []
        for c in range(1, self.transform.num_classes):
            row = dict(epoch=epoch, class_id=c, selected=int(self.selected[c]),
                       applied=int(self.applied[c]), original_positive_samples=int(self.original_positive[c]))
            row.update({'reason_' + name: int(self.target_reasons[c, i]) for i, name in enumerate(REASONS)})
            target_rows.append(row)
        append_csv(os.path.join(directory, 'region_mix_targets.csv'), target_rows)
        logger.info('REGION_MIX_AUDIT ' + str(summary))


class ModuleAudit:
    def __init__(self, model, required):
        self.counts, self.handles = {}, []
        modules = dict(model.named_modules())
        for name in required:
            if name not in modules:
                raise RuntimeError("Required module does not exist: " + name)
            module = modules[name]
            parameters = list({id(p): p for p in module.parameters() if p.requires_grad}.values())
            if not parameters:
                raise RuntimeError("Required module has no trainable parameters: " + name)
            self.counts[name] = [0, 0]
            def forward_hook(_module, _inputs, _output, key=name):
                self.counts[key][0] += 1
            def gradient_hook(gradient, key=name):
                self.counts[key][1] += 1
                return gradient
            self.handles.append(module.register_forward_hook(forward_hook))
            self.handles.extend(p.register_hook(gradient_hook) for p in parameters)

    def begin_batch(self):
        self.previous = {k: tuple(v) for k, v in self.counts.items()}

    def reset_counts(self):
        # Validation forwards must not be counted as next epoch's training.
        for value in self.counts.values():
            value[:] = [0, 0]

    def check_batch(self):
        for name, (forward, gradient) in self.counts.items():
            old_forward, old_gradient = self.previous[name]
            if forward <= old_forward or gradient <= old_gradient:
                raise RuntimeError("Required module skipped forward or loss backward path: " + name)

    def finish(self, epoch, logger, directory):
        rows = [dict(epoch=epoch, module=name, forward_calls=v[0], parameter_gradient_callbacks=v[1])
                for name, v in self.counts.items()]
        append_csv(os.path.join(directory, "module_audit.csv"), rows)
        for row in rows:
            logger.info("MODULE_AUDIT " + str(row))
        for value in self.counts.values():
            value[:] = [0, 0]

    def close(self):
        for handle in self.handles:
            handle.remove()
