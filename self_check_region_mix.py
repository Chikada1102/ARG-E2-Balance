"""Synthetic checks for the V4 cross-case path; no Synapse data is required."""
import csv
import importlib.util
import logging
import tempfile
import os
from pathlib import Path
import numpy as np
from scipy import ndimage
from datasets.region_mix import TrainDonorPool, OrganRegionMixer, RegionMixDataset, REASONS
from datasets.multi_organ_aug import MultiOrganRandomGenerator, OPS
from datasets.multi_organ_aug_v1 import MultiOrganRandomGenerator as OriginalV1
from innovation_audit import AugmentationAudit
from self_check import batch_of, state_equal, must_fail


class FixtureRawDataset:
    def __init__(self, directory, names):
        self.directory, self.sample_list = str(directory), list(names)
        self.transform, self.split = None, 'train'

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        name = self.sample_list[index]
        with np.load(Path(self.directory) / (name + '.npz'), allow_pickle=False) as data:
            return dict(image=data['image'].copy(), label=data['label'].copy(), case_name=name)


def fixture(root):
    directory, lists = root / 'npz', root / 'lists'
    directory.mkdir()
    lists.mkdir()
    names = []
    gy, gx = np.indices((128, 128))
    for case in range(1, 19):
        for plane in range(3):
            name = 'case%04d_slice%03d' % (case, plane)
            names.append(name)
            label = np.zeros((128, 128), np.int64)
            shift = (case % 3) - 1
            label[23:62, 25 + shift:65 + shift] = 1
            if case <= 6:
                label[78:92 + case % 2, 77 + shift:91 + shift] = 2
            label[43 + plane:53 + plane, 88:100] = 6
            label[78:112, 20 + shift:43 + shift] = 7
            if case % 2:
                label[78:82, 20 + shift:24 + shift] = 0
            image = (.2 * np.sin(gx / 12.) + gy / 128. + label / 9. + case * .01).astype(np.float32)
            if case == 18 and plane == 2:
                image = ndimage.zoom(image, (1.25, 1.5), order=3)
                label = ndimage.zoom(label, (1.25, 1.5), order=0, prefilter=False)
            np.savez_compressed(directory / (name + '.npz'), image=image, label=label)
    (lists / 'train.txt').write_text('\n'.join(names) + '\n')
    (lists / 'test_vol.txt').write_text('\n'.join('case%04d' % c for c in range(19, 31)) + '\n')
    # This invalid, unlisted test file must never be read into the donor pool.
    np.savez_compressed(directory / 'case0030_slice000.npz',
                        image=np.full((128, 128), np.nan), label=np.full((128, 128), 99))
    return directory, lists, names


def make_dataset(raw, pool, probability=1., strategy='protected'):
    transform = MultiOrganRandomGenerator((128, 128), preset='v4',
                                          probabilities={'region_mix': probability})
    mixer = OrganRegionMixer(pool, probability=probability, strategy=strategy)
    return RegionMixDataset(raw, transform, mixer)


def rejection_checks(root):
    def pair_pool(folder, a, b):
        folder.mkdir()
        data, lists = folder / 'npz', folder / 'lists'
        data.mkdir()
        lists.mkdir()
        names = ['case0001_slice000', 'case0002_slice000']
        for name, label in zip(names, (a, b)):
            image = (label / 9. + np.indices(label.shape)[0] / 100.).astype(np.float32)
            np.savez_compressed(data / (name + '.npz'), image=image, label=label)
        (lists / 'train.txt').write_text('\n'.join(names) + '\n')
        return TrainDonorPool(data, lists, ['case0001', 'case0002'], a.shape)

    # An all-organ image offers no genuine surrounding background context.
    full = np.ones((64, 64), np.int64)
    pool = pair_pool(root / 'no_context', full, full)
    mixer = OrganRegionMixer(pool, probability=1.)
    image, label = pool.load(0)
    result, reason = mixer._organ_candidate(image, label, 0, 1, 1)
    assert result is None and reason == 'no_context'

    # A donor fills a concavity occupied by another recipient organ.
    a, b = np.zeros((96, 96), np.int64), np.zeros((96, 96), np.int64)
    a[35:55, 35:55] = b[35:55, 35:55] = 1
    a[35:45, 45:55] = 7
    b[65:75, 65:75] = 7
    pool = pair_pool(root / 'protected_overlap', a, b)
    mixer = OrganRegionMixer(pool, probability=1.)
    image, label = pool.load(0)
    result, reason = mixer._organ_candidate(image, label, 0, 1, 1)
    assert result is None and reason == 'target_clipped'
    print('PASS rejected missing-context and destructive-overlap organ pairs')


def run_region_checks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data, lists, names = fixture(root)
        original_lists = [(lists / name).read_bytes() for name in ('train.txt', 'test_vol.txt')]
        allowed = ['case%04d' % c for c in range(1, 19)]
        pool = TrainDonorPool(data, lists, allowed, (128, 128), root / 'index.json')
        pool2 = TrainDonorPool(data, lists, allowed, (128, 128), root / 'index.json')
        assert pool.fingerprint == pool2.fingerprint and pool.records == pool2.records
        assert len(pool.names) == 54 and set(pool.cases) == set(allowed)
        raw = FixtureRawDataset(data, names)
        for i in range(len(names)):
            for target in pool.tables:
                if str(target) in pool.records[i]['classes']:
                    assert all(pool.cases[int(j)] != pool.cases[i] for j in pool.candidates(i, target))

        disabled = make_dataset(raw, pool, probability=0.)
        reference = OriginalV1((128, 128))
        comparisons = 0
        comparison_indices = (0, 1, 8, 17, 26, 35, 44, 53)
        for seed in range(10):
            np.random.seed(seed)
            expected = [reference.augment_numpy(raw[i]['image'], raw[i]['label']) for i in comparison_indices]
            expected_state = np.random.get_state()
            np.random.seed(seed)
            for i, (x, y) in zip(comparison_indices, expected):
                xx, yy, meta = disabled.process_numpy(i)
                assert np.array_equal(x, xx) and np.array_equal(y, yy)
                comparisons += 1
            assert state_equal(expected_state, np.random.get_state())

        dataset = make_dataset(raw, pool)
        audit = AugmentationAudit(dataset.transform)
        applied = label_changed = 0
        reasons = {}
        successful = None
        for seed in range(180):
            index = seed % len(raw)
            base = raw[index]
            np.random.seed(seed)
            mixed = dataset.mixed_sample(index)
            report = mixed['region_mix_report']
            reasons[REASONS[int(report['mix_reason'])]] = reasons.get(REASONS[int(report['mix_reason'])], 0) + 1
            if report['mix_event'][2]:
                applied += 1
                target = int(report['mix_target'])
                other = (base['label'] != 0) & (base['label'] != target)
                assert np.array_equal(mixed['image'][other], base['image'][other])
                assert np.array_equal(mixed['label'][other], base['label'][other])
                assert report['mix_donor_case'] in allowed and report['mix_donor_case'] != report['mix_anchor_case']
                assert not report['mix_donor_name'].startswith('case0030')
                assert report['mix_metrics'][4] == report['mix_metrics'][6] == 0
                assert min(report['mix_metrics'][2:4]) >= .95
                assert set(np.unique(mixed['label'])) == set(np.unique(base['label']))
                label_changed += int(not np.array_equal(mixed['label'], base['label']))
                successful = (index, mixed)
            x, y, meta = dataset.transform.process(mixed)
            assert np.isfinite(x).all() and y.dtype == np.int64
            audit.consume(batch_of(x, y, meta))
            np.random.seed(seed)
            xx, yy, _ = dataset.process_numpy(index)
            assert np.array_equal(x, xx) and np.array_equal(y, yy)
        assert applied > 0, reasons
        assert label_changed > 0, 'Synthetic organ shapes never changed'
        with tempfile.TemporaryDirectory() as audit_dir:
            audit.finish(1, logging.getLogger('region_test'), audit_dir)
            pairs = list(csv.DictReader(open(Path(audit_dir) / 'region_mix_pairs.csv')))
            assert sum(int(r['applications']) for r in pairs) == applied
            assert all(r['anchor_case'] != r['donor_case'] for r in pairs)

        index, mixed = successful
        must_fail(lambda: dataset.transform.process(raw[index]))
        changed = dict(mixed, image=raw[index]['image'])
        must_fail(lambda: dataset.transform.process(changed))
        x, y, meta = dataset.transform.process(mixed)
        bad = batch_of(x, y, meta)
        bad['aug_audit']['mix_donor_case'] = ['case0030']
        must_fail(lambda: AugmentationAudit(dataset.transform).consume(bad))
        must_fail(lambda: pool.load(len(pool.names)))

        # A one-patient training pool has no valid cross-patient donor.
        one_lists = root / 'one_lists'
        one_lists.mkdir()
        one_names = names[:3]
        (one_lists / 'train.txt').write_text('\n'.join(one_names) + '\n')
        one = TrainDonorPool(data, one_lists, ['case0001'], (128, 128))
        impossible = make_dataset(FixtureRawDataset(data, one_names), one)
        sample = impossible.mixed_sample(0)
        event = sample['region_mix_report']['mix_event']
        assert event.tolist() == [1, 1, 0, 0, 1]
        assert np.array_equal(sample['image'], raw[0]['image'])
        assert REASONS[int(sample['region_mix_report']['mix_reason'])] == 'no_compatible_donor'

        # CutMix's image and label mask have an independent constant-value oracle.
        class ConstantPool:
            output_size = (32, 32)
            def load(self, index):
                return np.full((32, 32), 2., np.float32), np.full((32, 32), 2, np.int64)
        simple = OrganRegionMixer(ConstantPool(), probability=1., strategy='cutmix')
        for seed in range(15):
            np.random.seed(seed)
            xx, yy, _ = simple._cutmix(np.ones((32, 32), np.float32), np.ones((32, 32), np.int64), 0)
            assert np.array_equal(xx, yy)

        for strategy in ('organ', 'cutmix'):
            variant = make_dataset(raw, pool, strategy=strategy)
            count = 0
            for seed in range(20):
                np.random.seed(seed)
                x, y, meta = variant.process_numpy(seed % len(raw))
                count += int(meta['mix_event'][2])
                assert np.isfinite(x).all() and set(np.unique(y)).issubset({0, 1, 2, 6, 7})
            assert count > 0
        assert original_lists == [(lists / name).read_bytes() for name in ('train.txt', 'test_vol.txt')]
        rejection_checks(root)
        default = make_dataset(raw, pool, probability=.2)
        default_audit = AugmentationAudit(default.transform)
        np.random.seed(42)
        for i in range(162):
            x, y, meta = default.process_numpy(i % len(raw))
            default_audit.consume(batch_of(x, y, meta))
        row = default_audit.events[OPS.index('region_mix')]
        assert 0 < row[2] <= row[1] < row[0]
        with tempfile.TemporaryDirectory() as audit_dir:
            default_audit.finish(1, logging.getLogger('default_region_test'), audit_dir)
        print('PASS V4: %d exact disabled-mix/V1 output+RNG comparisons' % comparisons)
        print('PASS V4: 180 seeds, reproducibility, cross-patient provenance, protected foreground, label classes, guard failures')
        print('Protected mix actual applications:', applied, 'of 180; reasons:', reasons)
        print('Applied mixes with changed target-label geometry:', label_changed)
        print('Default p=0.2 check: triggered/applied/consumed =', int(row[1]), int(row[2]), int(row[0]))
        print('PASS index cache, unlisted poisoned test NPZ exclusion, no-donor rejection, unchanged lists, CutMix mask oracle, ablations')
        torch_region_checks(raw, pool)


def torch_region_checks(raw, pool):
    if importlib.util.find_spec('torch') is None:
        print('SKIP real V4 DataLoader checks: PyTorch is not installed; run in your training environment.')
        return
    import torch
    from torch.utils.data import DataLoader
    from innovation_audit import ModuleAudit
    worker_counts = (0,) if os.environ.get('VMUNET_SELF_CHECK_ZERO_WORKERS') == '1' else (0, 2)
    for workers in worker_counts:
        torch.manual_seed(42)
        np.random.seed(42)
        dataset = make_dataset(raw, pool, probability=.5)
        audit = AugmentationAudit(dataset.transform, strict=False)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vmunet = torch.nn.Conv2d(1, 9, 1)
            def forward(self, x):
                return self.vmunet(x)
        net = Net()
        module = ModuleAudit(net, ['vmunet'])
        for batch in DataLoader(dataset, batch_size=8, shuffle=True, num_workers=workers):
            audit.consume(batch)
            module.begin_batch()
            torch.nn.functional.cross_entropy(net(batch['image']), batch['label']).backward()
            module.check_batch()
            net.zero_grad()
        assert audit.samples == len(dataset)
        assert audit.region.applied_samples > 0
        module.close()
    print('PASS real V4 DataLoader workers='+str(worker_counts)+', nested metadata collation and parameter backward checks')


if __name__ == '__main__':
    run_region_checks()
