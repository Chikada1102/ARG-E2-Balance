"""V4.1 checks: training-only feedback, worker stamps and image-only seams."""
import argparse
import ast
import contextlib
import copy
import csv
import importlib.util
import io
import json
import logging
import multiprocessing
import sys
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from datasets.multi_organ_aug import MultiOrganRandomGenerator
from datasets.region_mix import TrainDonorPool, OrganRegionMixer, RegionMixDataset
from datasets.region_mix_refinement import (TargetBalance, REFINEMENTS, REFINE_METRICS,
                                            refine_patch, background_seam)
from innovation_audit import AugmentationAudit
from self_check import batch_of, state_equal, must_fail
from self_check_region_mix import fixture, FixtureRawDataset


def dataset_for(raw, pool, refinement, probability=1.):
    transform = MultiOrganRandomGenerator((128, 128), preset='v4',
                                          probabilities={'region_mix': probability})
    mixer = OrganRegionMixer(pool, probability=probability, refinement=refinement)
    return RegionMixDataset(raw, transform, mixer)


def spawn_sample(arguments):
    dataset, index, seed = arguments
    np.random.seed(seed)
    return dataset.process_numpy(index)


def numpy_process_checks(raw, pool):
    if os.environ.get('VMUNET_SELF_CHECK_ZERO_WORKERS') == '1':
        print('SKIP spawned NumPy workers: --zero-workers test mode')
        return
    dataset = dataset_for(raw, pool, 'both')
    with tempfile.TemporaryDirectory() as directory:
        with multiprocessing.get_context('spawn').Pool(2) as workers:
            for epoch in (1, 2):
                dataset.set_mix_epoch(epoch)
                audit = AugmentationAudit(dataset.transform, strict=False)
                arguments = [(dataset, i, 2000 + i) for i in range(8)]
                results = workers.map(spawn_sample, arguments)
                for argument, (x, y, meta) in zip(arguments, results):
                    xx, yy, expected = spawn_sample(argument)
                    assert np.array_equal(x, xx) and np.array_equal(y, yy)
                    assert np.array_equal(meta['mix_balance_weights'], expected['mix_balance_weights'])
                    audit.consume(batch_of(x, y, meta))
                # The feedback update itself must see the entire original list.
                complete = AugmentationAudit(dataset.transform, strict=False)
                for index in range(len(raw)):
                    x, y, meta = dataset.process_numpy(index)
                    complete.consume(batch_of(x, y, meta))
                dataset.finish_mix_epoch(complete.region, epoch, logging.getLogger('numpy_spawn'), directory)
    print('PASS two spawned NumPy workers: two-epoch state transfer, exact local outputs and consumer audits')


def controller_checks():
    frequency = np.array([0, 100, 100, 0, 0, 0, 100, 0, 0])
    balance = TargetBalance(frequency, 'training-fixture')
    state = np.random.get_state()
    assert np.array_equal(balance.weights(), np.ones(9))
    balance.update(1, [0, 20, 20, 0, 0, 0, 20, 0, 0], [0, 20, 2, 0, 0, 0, 10, 0, 0])
    weights = balance.weights()
    assert weights[2] > weights[6] > weights[1]
    assert .5 <= weights.min() <= weights.max() <= 2.
    assert np.array_equal(weights[frequency == 0], np.ones((frequency == 0).sum()))
    assert state_equal(state, np.random.get_state()), 'Feedback consumed augmentation RNG'
    restored = TargetBalance(frequency, 'training-fixture')
    restored.load_state_dict(json.loads(json.dumps(balance.state_dict())))
    assert np.array_equal(weights, restored.weights())
    must_fail(lambda: balance.update(3, np.zeros(9), np.zeros(9)))
    must_fail(lambda: balance.update(2, np.zeros(9), np.ones(9)), ValueError)
    for key, value in [('fingerprint', 'test-fixture'), ('completed_epochs', -1),
                       ('visible_frequency', [0] * 9), ('selected', [-1] * 9),
                       ('applied', [float('nan')] * 9)]:
        broken = copy.deepcopy(balance.state_dict())
        broken[key] = value
        must_fail(lambda: restored.load_state_dict(broken))
    print('PASS balance: low-coverage class gains weight, bounded updates, no RNG draw, state guards')


def seam_oracles():
    gy, gx = np.indices((64, 64))
    anchor = (.1 * gx / 64. + gy / 64.).astype(np.float32)
    donor = (anchor * 1.05 + .4).astype(np.float32)
    labels = np.zeros((64, 64), np.int64)
    labels[24:40, 24:40] = 1
    mask = np.ones((64, 64), bool)
    mask[5:12, 5:12] = False
    output, metrics = refine_patch(anchor, donor, labels, labels, mask, (1., 1.), 1.)
    bg = mask & (labels == 0)
    median = float(np.median(donor[bg]))
    matched = ((donor - median) * metrics[3] + median + metrics[4]).astype(np.float32)
    # Foreground is a hard donor assignment. Its labels never enter a blend.
    assert np.allclose(output[labels == 1], matched[labels == 1], rtol=1e-6, atol=1e-7)
    assert np.array_equal(output[~mask], anchor[~mask])
    # At a one-pixel ROI border, width=2 gives an independently known alpha=.5.
    assert np.isclose(output[0, 30], .5 * matched[0, 30] + .5 * anchor[0, 30])
    assert np.isfinite(output).all() and .9 <= metrics[3] <= 1.1
    assert metrics[5] > 0 and metrics[2] > 0
    assert abs(metrics[4]) <= .25 + 1e-8
    assert np.mean(np.abs(output[bg] - anchor[bg])) < np.mean(np.abs(donor[bg] - anchor[bg]))
    # A known vertical step yields exactly 16 adjacent background seam pairs.
    a = np.zeros((16, 16), np.float32)
    a[:, 8:] = 1.
    b = a.copy()
    b[:, 8] = .5
    m = np.zeros((16, 16), bool)
    m[:, 8:] = True
    empty = np.zeros((16, 16), np.int64)
    count, before, after = background_seam(a, b, m, empty, empty)
    assert count == 16 and before == 1. and after == .5
    # Constant context and nonsquare physical sampling cannot produce NaNs.
    for sampling in ((1., 1.), (.5, .75), (2., 3.)):
        out, report = refine_patch(np.zeros((64, 64), np.float32),
                                   np.ones((64, 64), np.float32), labels, labels, mask,
                                   sampling, 1.)
        assert np.isfinite(out).all() and np.isfinite(report).all()
        assert np.array_equal(out[~mask], np.zeros((64, 64), np.float32)[~mask])
    print('PASS seam: analytic alpha, hard foreground, protected pixels, bounded gray transform, edge-jump oracle')


def wiring_checks():
    source = (Path(__file__).parent / 'train_synapse.py').read_text()
    tree = ast.parse(source)
    cli = ast.Module(body=tree.body[-1].body, type_ignores=[])
    code = compile(cli, '<isolated training CLI>', 'exec')
    def parse(arguments):
        config = SimpleNamespace(model_config={'load_ckpt_path': 'fixture'})
        captured = []
        namespace = dict(argparse=argparse, np=np, REFINEMENTS=REFINEMENTS,
                         Path=Path, json=json, __file__=str(Path(__file__).with_name('train_synapse.py')),
                         setting_config=config, main=lambda c: captured.append(c))
        previous = sys.argv
        try:
            sys.argv = ['train_synapse.py'] + arguments
            exec(code, namespace)
        finally:
            sys.argv = previous
        assert len(captured) == 1
        return captured[0]
    for mode in REFINEMENTS:
        config = parse(['--aug-preset', 'v4', '--mix-refinement', mode])
        assert config.mix_refinement == mode and config.mix_prob == .2
    assert parse([]).mix_refinement == 'balance'
    assert parse(['--arg-experiment', 'E2']).arg_experiment == 'E2'
    assert parse(['--arg-experiment', 'E1']).work_dir.endswith('arg_e1_balance/')
    assert parse(['--arg-experiment', 'E2']).work_dir.endswith('arg_e2_balance/')
    assert parse(['--aug-preset', 'v1', '--mix-refinement', 'none']).mix_refinement == 'none'
    for arguments in (['--aug-preset', 'v1', '--mix-refinement', 'balance'],
                      ['--mix-refinement', 'seam', '--mix-strategy', 'cutmix'],
                      ['--mix-refinement', 'balance', '--mix-prob', '0'],
                      ['--mix-refinement', 'both', '--disable-aug', 'region_mix']):
        with contextlib.redirect_stderr(io.StringIO()):
            must_fail(lambda: parse(arguments), SystemExit)
    assert 'train_dataset.set_mix_epoch(epoch)' in source
    assert "mixer.load_state_dict(checkpoint.get('mix_refinement_state'), saved_epoch)" in source
    assert "'mix_refinement_state': mixer.state_dict()" in source
    assert "refinement=config.mix_refinement" in source
    assert 'persistent_workers=False' in source
    engine = (Path(__file__).parent / 'engine_synapse.py').read_text()
    assert engine.index('finish_mix_epoch(audit.region') < engine.index('return mean')
    print('PASS training CLI, invalid combinations, epoch propagation, feedback-before-checkpoint and resume wiring')


def run_refinement_checks():
    controller_checks()
    seam_oracles()
    wiring_checks()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data, lists, names = fixture(root)
        list_bytes = [(lists / n).read_bytes() for n in ('train.txt', 'test_vol.txt')]
        allowed = ['case%04d' % c for c in range(1, 19)]
        pool = TrainDonorPool(data, lists, allowed, (128, 128))
        raw = FixtureRawDataset(data, names)
        control = dataset_for(raw, pool, 'none')
        seam = dataset_for(raw, pool, 'seam')
        control.set_mix_epoch(1)
        seam.set_mix_epoch(1)
        changed, feathered, applications = 0, 0, 0
        for seed in range(100):
            index = seed % len(raw)
            np.random.seed(seed)
            a = control.mixed_sample(index)
            state = np.random.get_state()
            np.random.seed(seed)
            b = seam.mixed_sample(index)
            assert state_equal(state, np.random.get_state()), 'Seam-only changed donor RNG'
            ar, br = a['region_mix_report'], b['region_mix_report']
            assert ar['mix_donor_name'] == br['mix_donor_name']
            assert np.array_equal(a['label'], b['label']), 'Seam-only changed geometry'
            assert np.array_equal(ar['mix_event'], br['mix_event'])
            assert np.array_equal(ar['mix_metrics'], br['mix_metrics'])
            if br['mix_event'][2]:
                applications += 1
                assert br['mix_refine_metrics'][0] == 1
                assert b['label'].dtype == np.int64
                target = int(br['mix_target'])
                original = raw[index]
                other = (original['label'] != 0) & (original['label'] != target)
                assert np.array_equal(b['image'][other], original['image'][other])
                changed += int(not np.array_equal(a['image'], b['image']))
                feathered += int(br['mix_refine_metrics'][5] > 0)
            x, y, meta = seam.transform.process(b)
            audit = AugmentationAudit(seam.transform, strict=False)
            batch = batch_of(x, y, meta)
            audit.consume(batch)
            broken = copy.deepcopy(batch)
            broken['aug_audit']['mix_refine_metrics'][0, 0] = 0
            if br['mix_event'][2]:
                must_fail(lambda: AugmentationAudit(seam.transform).consume(broken))
        assert applications and changed and feathered
        print('PASS seam-only 100 seeds: same labels/donors/RNG; applications=%d image_changed=%d feathered=%d' %
              (applications, changed, feathered))

        balanced = dataset_for(raw, pool, 'balance')
        for epoch in (1, 2, 3):
            balanced.set_mix_epoch(epoch)
            audit = AugmentationAudit(balanced.transform)
            np.random.seed(100 + epoch)
            for index in range(len(raw)):
                x, y, meta = balanced.process_numpy(index)
                audit.consume(batch_of(x, y, meta))
            outdir = root / 'balance_audit'
            audit.finish(epoch, logging.getLogger('refinement_test'), str(outdir))
            balanced.finish_mix_epoch(audit.region, epoch, logging.getLogger('refinement_test'), str(outdir))
            assert balanced.mixer.balance.completed_epochs == epoch
            restored = dataset_for(raw, pool, 'balance')
            saved = json.loads(json.dumps(balanced.mixer.state_dict()))
            restored.mixer.load_state_dict(saved, epoch)
            restored.set_mix_epoch(epoch + 1)
            assert np.array_equal(restored.mixer.active_weights, balanced.mixer.balance.weights())
            must_fail(lambda: restored.mixer.load_state_dict(saved, epoch + 1))
            if epoch == 2:
                assert not np.array_equal(balanced.mixer.active_weights, np.ones(9))
                broken = batch_of(x, y, meta)
                broken['aug_audit']['mix_balance_weights'][0, 1] += .01
                must_fail(lambda: AugmentationAudit(balanced.transform).consume(broken))
                broken = batch_of(x, y, meta)
                broken['aug_audit']['mix_selection_epoch'][0] = 1
                must_fail(lambda: AugmentationAudit(balanced.transform).consume(broken))
        rows = list(csv.DictReader((root / 'balance_audit' / 'region_mix_balance.csv').open()))
        assert len(rows) == 24 and all(float(r['multiplier_used']) == 1 for r in rows if r['epoch'] == '1')
        reasons = list(csv.DictReader((root / 'balance_audit' / 'region_mix_targets.csv').open()))
        assert len(reasons) == 24
        assert all(sum(int(v) for k, v in r.items() if k.startswith('reason_')) == int(r['selected']) for r in reasons)
        print('PASS 3 complete online training passes: feedback, per-class reasons, stale-worker rejection, JSON resume')
        print('Epoch-four multipliers:', np.round(balanced.mixer.balance.weights(), 4).tolist())

        # Both mode passes through image/label hashes and the consumer audit.
        both = dataset_for(raw, pool, 'both')
        both.set_mix_epoch(1)
        audit = AugmentationAudit(both.transform)
        for seed in range(30):
            np.random.seed(seed)
            x, y, meta = both.process_numpy(seed % len(raw))
            audit.consume(batch_of(x, y, meta))
        assert audit.region.refine_changed_samples > 0
        with tempfile.TemporaryDirectory() as outdir:
            audit.finish(1, logging.getLogger('both_test'), outdir)
        assert list_bytes == [(lists / n).read_bytes() for n in ('train.txt', 'test_vol.txt')]
        must_fail(lambda: OrganRegionMixer(pool, strategy='cutmix', refinement='both'), ValueError)
        print('PASS combined mode, unchanged input lists and no test-file access')
        for mode in ('balance', 'seam', 'both'):
            default = dataset_for(raw, pool, mode, probability=.2)
            for epoch in (1, 2):
                default.set_mix_epoch(epoch)
                audit = AugmentationAudit(default.transform, strict=False)
                np.random.seed(42 + epoch)
                for index in range(len(raw)):
                    x, y, meta = default.process_numpy(index)
                    audit.consume(batch_of(x, y, meta))
                assert 0 < audit.region.applied_samples < audit.samples
                with tempfile.TemporaryDirectory() as outdir:
                    audit.finish(epoch, logging.getLogger('default_refinement'), outdir)
                    default.finish_mix_epoch(audit.region, epoch, logging.getLogger('default_refinement'), outdir)
        print('PASS default p=0.2 for all three refinements: trigger/bypass accounting and two-epoch feedback')
        numpy_process_checks(raw, pool)
        torch_refinement_checks(raw, pool)


def torch_refinement_checks(raw, pool):
    if importlib.util.find_spec('torch') is None:
        print('SKIP V4.1 real DataLoader: PyTorch unavailable here; checks run automatically in the VM-UNet environment.')
        return
    import torch
    from torch.utils.data import DataLoader
    with tempfile.TemporaryDirectory() as directory:
        worker_counts = (0,) if os.environ.get('VMUNET_SELF_CHECK_ZERO_WORKERS') == '1' else (0, 2)
        for mode in ('balance', 'seam', 'both'):
            for workers in worker_counts:
                np.random.seed(42)
                torch.manual_seed(42)
                dataset = dataset_for(raw, pool, mode)
                loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=workers,
                                    persistent_workers=False)
                for epoch in (1, 2):
                    dataset.set_mix_epoch(epoch)
                    audit = AugmentationAudit(dataset.transform, strict=False)
                    for batch in loader:
                        audit.consume(batch)
                    assert audit.samples == len(dataset)
                    audit.finish(epoch, logging.getLogger('worker_test'), directory)
                    dataset.finish_mix_epoch(audit.region, epoch, logging.getLogger('worker_test'), directory)
        # Spawn also transfers the updated second-epoch state into fresh workers.
        if os.environ.get('VMUNET_SELF_CHECK_ZERO_WORKERS') == '1':
            print('PASS real V4.1 DataLoader workers=(0,), 3 modes and two-epoch feedback; SKIP spawn/2-worker checks')
            return
        dataset = dataset_for(raw, pool, 'both')
        for epoch in (1, 2):
            dataset.set_mix_epoch(epoch)
            audit = AugmentationAudit(dataset.transform, strict=False)
            loader = DataLoader(dataset, batch_size=8, num_workers=2, persistent_workers=False,
                                multiprocessing_context='spawn')
            for batch in loader:
                audit.consume(batch)
            dataset.finish_mix_epoch(audit.region, epoch, logging.getLogger('spawn_test'), directory)
    print('PASS real V4.1 DataLoader 0/2 workers for 3 modes, two-epoch feedback and spawn propagation')


if __name__ == '__main__':
    run_refinement_checks()
