"""Run python self_check.py from the patched VM-UNet root before training."""
import csv
import importlib.util
import logging
import tempfile
import os
from pathlib import Path
import numpy as np
from datasets.multi_organ_aug import MultiOrganRandomGenerator, OPS
from datasets.multi_organ_aug_v1 import MultiOrganRandomGenerator as OriginalV1
from innovation_audit import AugmentationAudit, ModuleAudit
from multi_organ_metrics import binary_metrics
from split_guard import synapse_split_manifest


def phantom(kind=0):
    label = np.zeros((128, 128), dtype=np.int64)
    label[28:68, 25:65] = 1
    label[76:108, 78:110] = 2
    label[60:66, 80:86] = 3
    label[84:88, 36:40] = 4
    if kind == 1:
        label[1:6, 118:124] = 5
    if kind == 2:
        label[:] = 0
    grid_y, grid_x = np.indices(label.shape)
    image = (np.sin(grid_x / 13.) * .15 + grid_y / 128. + label / 10.).astype(np.float32)
    return image, label


def batch_of(image, label, meta):
    return dict(image=image[None, None], label=label[None],
                aug_audit={k: [v] if isinstance(v, str) else np.expand_dims(v, 0)
                           for k, v in meta.items()})


def state_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def must_fail(call, error=RuntimeError):
    try:
        call()
    except error:
        return
    raise AssertionError("Expected failure was not detected")


def v1_equivalence():
    count = 0
    for kind in range(3):
        image, label = phantom(kind)
        old = OriginalV1((112, 112))
        for seed in range(20):
            np.random.seed(seed)
            expected = [old.augment_numpy(image, label) for _ in range(5)]
            expected_state = np.random.get_state()
            for aug in (MultiOrganRandomGenerator((112, 112), preset="v1"),
                        MultiOrganRandomGenerator((112, 112), preset="v3",
                                                  disable=("adaptive_crop", "local_intensity"))):
                np.random.seed(seed)
                for a, b in expected:
                    c, d, _ = aug.process(dict(image=image, label=label))
                    assert np.array_equal(a, c), "V1 image differs"
                    assert np.array_equal(b, d), "V1 label differs"
                    count += 1
                assert state_equal(expected_state, np.random.get_state()), "V1 random state differs"
    print("PASS V1 exact output + RNG equivalence:", count, "comparisons, consecutive calls")


def augmentation_checks():
    image, label = phantom()
    aug = MultiOrganRandomGenerator((128, 128), preset="v3")
    audit = AugmentationAudit(aug)
    zooms = []
    for seed in range(160):
        np.random.seed(seed)
        x, y, meta = aug.process(dict(image=image, label=label))
        assert x.shape == y.shape == label.shape
        assert x.dtype == np.float32 and y.dtype == np.int64
        assert x.flags.c_contiguous and y.flags.c_contiguous
        assert np.isfinite(x).all() and set(np.unique(y)) == set(np.unique(label))
        assert np.array_equal(meta['events'][:, 1], meta['events'][:, 2:].sum(axis=1))
        zoom = float(meta['crop_zoom'])
        assert aug._retains_organs(label, y, zoom * zoom), "Cumulative organ retention failed"
        if meta['events'][OPS.index('adaptive_crop'), 2]:
            zooms.append(zoom)
        np.random.seed(seed)
        xx, yy, _ = aug.process(dict(image=image, label=label))
        assert np.array_equal(x, xx) and np.array_equal(y, yy)
        audit.consume(batch_of(x, y, meta))
    with tempfile.TemporaryDirectory() as directory:
        audit.finish(1, logging.getLogger('self_check'), directory)
        organ_rows = list(csv.DictReader(open(Path(directory) / 'augmentation_organs.csv')))
        assert any(int(r['local_intensity_applied_target_count']) > 0 for r in organ_rows)
        assert all(int(r['lost_samples']) == 0 for r in organ_rows)
    for op in ('adaptive_crop', 'rotate', 'intensity', 'local_intensity'):
        assert audit.events[OPS.index(op), 2] > 0, op + ' never applied'
    assert max(zooms) > 1 / .85, "Adaptive crop never exceeded V1's old scale range"
    must_fail(lambda: audit.consume(dict(label=label[None])))
    bad = batch_of(x, y, meta)
    bad['image'] = np.zeros_like(bad['image'])
    must_fail(lambda: audit.consume(bad))
    bad = batch_of(x, y, meta)
    bad['label'] = np.roll(bad['label'], 1, axis=2)
    must_fail(lambda: audit.consume(bad))
    must_fail(lambda: aug.process(dict(image=image, label=label + .1)), ValueError)
    photo = MultiOrganRandomGenerator((128, 128), probabilities=dict(
        crop=0., adaptive_crop=0., rotate=0., intensity=0., local_intensity=1.))
    for seed in range(30):
        np.random.seed(seed)
        x, y, meta = photo.process(dict(image=image, label=label))
        assert np.array_equal(y, label), "Photometric transform changed labels"
        assert x.min() >= image.min() and x.max() <= image.max()
        assert np.max(np.abs(x - image)) <= .15 * image.std() + 1e-6
    for kind in (1, 2):
        im, la = phantom(kind)
        for seed in range(20):
            np.random.seed(seed)
            x, y, meta = aug.process(dict(image=im, label=la))
            assert aug._retains_organs(la, y, float(meta['crop_zoom']) ** 2)
            assert np.isfinite(x).all()
    constant = np.zeros((128, 128), np.float32)
    x, y, meta = aug.process(dict(image=constant, label=np.zeros_like(label)))
    assert np.isfinite(x).all() and not y.any()
    print("PASS V3: 160 seeds, reproducibility, cumulative retention, local image-only changes, bypass guards")
    print("Actual application counts:", dict(zip(OPS, audit.events[:, 2].tolist())))
    print("Accepted adaptive linear zoom range:", min(zooms), max(zooms))


def split_and_metric_checks():
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        train = '\n'.join('case%04d_slice000' % i for i in range(1, 19)) + '\n'
        test = '\n'.join('case%04d' % i for i in range(19, 31)) + '\n'
        (p / 'train.txt').write_text(train)
        (p / 'test_vol.txt').write_text(test)
        manifest = synapse_split_manifest(p)
        assert len(manifest['train']['cases']) == 18 and len(manifest['test_vol']['cases']) == 12
        assert (p / 'train.txt').read_text() == train and (p / 'test_vol.txt').read_text() == test
        (p / 'test_vol.txt').write_text(test.replace('case0019', 'case0001'))
        must_fail(lambda: synapse_split_manifest(p), ValueError)
    a = np.zeros((20, 20), bool)
    a[5:10, 5:10] = True
    shifted = np.roll(a, 1, axis=1)
    assert binary_metrics(a, a)['dice'] == 1.
    assert np.isclose(binary_metrics(a, shifted)['dice'], .8)
    assert binary_metrics(a, np.zeros_like(a))['false_positive_only'] == 1
    assert binary_metrics(a, np.zeros_like(a))['dice'] == 0.
    assert np.isinf(binary_metrics(a, np.zeros_like(a))['hd95'])
    print("PASS read-only 18/12 split guard, overlap rejection, analytical strict metrics")


class SyntheticDataset:
    def __init__(self):
        self.transform = MultiOrganRandomGenerator((128, 128), preset='v3')

    def __len__(self):
        return 64

    def __getitem__(self, index):
        image, label = phantom()
        return self.transform(dict(image=image, label=label))


def torch_checks():
    if importlib.util.find_spec('torch') is None:
        print('SKIP PyTorch checks: torch is not installed; run again in your VM-UNet environment.')
        return
    import torch
    from torch.utils.data import DataLoader
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vmunet = torch.nn.Conv2d(1, 9, 1)
        def forward(self, x):
            return self.vmunet(x)
    worker_counts = (0,) if os.environ.get('VMUNET_SELF_CHECK_ZERO_WORKERS') == '1' else (0, 2)
    for workers in worker_counts:
        torch.manual_seed(42)
        np.random.seed(42)
        ds = SyntheticDataset()
        audit = AugmentationAudit(ds.transform, strict=False)
        net = Net()
        module = ModuleAudit(net, ['vmunet'])
        for data in DataLoader(ds, batch_size=8, num_workers=workers):
            audit.consume(data)
            module.begin_batch()
            torch.nn.functional.cross_entropy(net(data['image']), data['label']).backward()
            module.check_batch()
            net.zero_grad()
        assert audit.samples == len(ds)
        module.close()
    print('PASS real PyTorch DataLoader workers='+str(worker_counts)+', metadata collation and parameter backward callbacks')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zero-workers', action='store_true', help='restricted test environments only; skips multiprocess checks')
    if parser.parse_args().zero_workers:
        os.environ['VMUNET_SELF_CHECK_ZERO_WORKERS'] = '1'
    v1_equivalence()
    augmentation_checks()
    split_and_metric_checks()
    torch_checks()
    from self_check_region_mix import run_region_checks
    run_region_checks()
    from self_check_refinement import run_refinement_checks
    run_refinement_checks()
