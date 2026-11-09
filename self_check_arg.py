"""CPU autograd/reference scan + automatic native-CUDA model checks.

python self_check_arg.py --cuda-required  # required preflight on training host
Reference scan is test-only: never installed as a production fallback.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import sys
import types
from types import SimpleNamespace
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from models.vmunet import vmamba
from models.vmunet.vmunet import VMUNet
from models.vmunet.arg_vss import AnatomicalRelation, pack4, ARG_NAME, required_arg_modules
from arg_losses import auxiliary_targets, auxiliary_losses
from arg_audit import ARGTrainingAudit
from innovation_audit import ModuleAudit
from reference.official_main_loss import CeDiceLoss


def reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    """Real differentiable selective SSM recurrence; small CPU tests only."""
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, channels, length = u.shape
    B = B.repeat_interleave(channels//B.shape[1], dim=1)
    C = C.repeat_interleave(channels//C.shape[1], dim=1)
    state = u.new_zeros(batch, channels, A.shape[-1])
    ys = []
    for t in range(length):
        step = delta[:, :, t, None]
        state = (step*A).exp()*state + step*B[:, :, :, t]*u[:, :, t, None]
        ys.append((state*C[:, :, :, t]).sum(-1))
    y = torch.stack(ys, -1)
    if D is not None:
        y = y + u*D[None, :, None]
    if z is not None:
        y = y*F.silu(z)
    return (y, state) if return_last_state else y


def load_reference():
    p = Path(__file__).parent/'reference/vmamba_original.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '7ae11522bf750198e7245df7d2f099da3ef8448156a19ef10dccf9c2b14273b5'
    spec = importlib.util.spec_from_file_location('arg_original_vmamba_test', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_and_gate_checks():
    x = torch.arange(2*3*3*5).reshape(2, 3, 3, 5)
    y = pack4(x)
    assert torch.equal(y[:, 0], x.flatten(2))
    assert torch.equal(y[:, 1], x.transpose(2, 3).flatten(2))
    assert torch.equal(y[:, 2], y[:, 0].flip(-1))
    assert torch.equal(y[:, 3], y[:, 1].flip(-1))
    labels = torch.zeros(2, 224, 224, dtype=torch.long)
    labels[0, 15, 19] = 8
    soft, present = auxiliary_targets(labels, (14, 14), 9)
    assert soft[0, 8].max() > 0 and present[0, 7] == 1
    assert present[1].sum() == 0 and torch.allclose(soft.sum(1), torch.ones(2, 14, 14))
    labels.zero_()  # post-augmentation label determines the target, not a cached original
    assert auxiliary_targets(labels, (14, 14), 9)[1].sum() == 0
    semantic = torch.randn(2, 9, 14, 14, requires_grad=True)
    presence = torch.randn(2, 8, requires_grad=True)
    losses = auxiliary_losses(semantic, presence, labels)
    sum(losses).backward()
    assert all(torch.isfinite(v) for v in losses) and semantic.grad.norm() > 0 and presence.grad.norm() > 0
    m = AnatomicalRelation(12, dim=8, mode='E2')
    v, p = torch.randn(2, 8, 3, 5), torch.randn(2, 9, 3, 5).softmax(1)
    for head in (m.write_head, m.read_head):
        nn.init.normal_(head.weight)
        nn.init.constant_(head.bias, 3.)
    absent = m.relation_fields(v, p, torch.zeros(2, 8))
    for k in ('support', 'contrast', 'rho_b', 'rho_c', 'relation'):
        assert torch.count_nonzero(absent[k]) == 0
    for field, gate, head in [('support', 'rho_b', m.write_head), ('contrast', 'rho_c', m.read_head)]:
        assert torch.count_nonzero(.1*pack4(absent[gate])*head(absent[field]).tanh()) == 0
    p.zero_(); p[:, 3] = 1
    single = m.relation_fields(v, p, torch.ones(2, 8))
    assert single['rho_b'].min() > 0 and torch.count_nonzero(single['rho_c']) == 0
    p = torch.randn_like(p).softmax(1)
    field = m.relation_fields(v, p, torch.ones(2, 8))
    assert torch.allclose(field['relation'].sum(-1), torch.ones(2, 8), atol=1e-5)
    out = m(torch.randn(2, 12, 3, 5))
    assert out['delta_B'].abs().max() <= .1 and out['delta_C'].abs().max() <= .1
    assert all(torch.isfinite(t).all() for t in (out['delta_B'], out['delta_C']))
    print('PASS rectangle four-direction packing; tiny-organ/post-augmentation targets; background and absence/pair gates; bounded residuals')


def ss2d_checks(original):
    saved = getattr(vmamba, 'selective_scan_fn', None)
    saved_v1 = getattr(vmamba, 'selective_scan_fn_v1', None)
    original.selective_scan_fn = reference_scan
    vmamba.selective_scan_fn = reference_scan
    vmamba.selective_scan_fn_v1 = reference_scan
    try:
        torch.manual_seed(31); baseline = original.SS2D(8, d_state=16)
        torch.manual_seed(31); e1 = vmamba.SS2D(8, d_state=16)
        e2 = copy.deepcopy(e1)
        for seed, module, mode in [(83, e1, 'E1'), (83, e2, 'E2')]:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                module.arg = AnatomicalRelation(16, dim=8, mode=mode)
        for n, p in e1.arg.state_dict().items():
            assert torch.equal(p, e2.arg.state_dict()[n]), n
        assert not any('write_head' in n or 'query' in n for n, _ in e1.arg.named_parameters())
        x = torch.randn(2, 3, 5, 8)
        baseline.eval(); e1.eval(); e2.eval()
        expect = baseline(x)
        assert torch.equal(expect, e1(x)) and torch.equal(expect, e2(x))
        for module in (e1, e2):
            aux = module(x, return_aux=True)
            assert torch.equal(expect, aux[0]) and aux[1]['semantic_logits'].shape == (2, 9, 3, 5)
        captures = []
        def capture(*args, **kwargs):
            captures.append([a.detach().clone() for a in args[:5]])
            return reference_scan(*args, **kwargs)
        vmamba.selective_scan_fn = capture
        e1(x); e2(x)
        assert all(torch.equal(a, b) for a, b in zip(*captures))
        optimizer = torch.optim.AdamW(e2.parameters(), lr=.01)
        labels = torch.randint(0, 9, (2, 12, 20))
        relation_gradient_seen = False
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            output, aux = e2(x, return_aux=True)
            sem, presence = auxiliary_losses(aux['semantic_logits'], aux['presence_logits'], labels)
            loss = output.square().mean()+.1*sem+.05*presence
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e2.arg.parameters())
            relation_gradient_seen |= e2.arg.query.weight.grad.norm().item() > 0
            assert e2.arg.write_head.weight.grad.norm() > 0 and e2.arg.read_head.weight.grad.norm() > 0
            optimizer.step()
        assert relation_gradient_seen
        # Main loss alone now reaches the graph, beyond the auxiliary supervision.
        optimizer.zero_grad(set_to_none=True)
        e2(x).square().mean().backward()
        assert e2.arg.query.weight.grad.norm() > 0 and e2.arg.pair_encoder[0].weight.grad.norm() > 0
        output, aux = e2(x, return_aux=True)
        assert aux['diagnostics']['delta_b_rms'].max() > 0 and aux['diagnostics']['delta_c_rms'].max() > 0
        enabled_scan = captures[-1]
        branch = e2.arg; del e2.arg
        e2(x); disabled_scan = captures[-1]
        e2.arg = branch
        assert all(torch.equal(enabled_scan[k], disabled_scan[k]) for k in (0, 1, 2))
        assert all(not torch.equal(enabled_scan[k], disabled_scan[k]) for k in (3, 4))
        # Compare the SAME trained model, with changed geometry/support/pair context.
        trained = e2(x).detach()
        q, geo = e2.arg.query.weight.detach().clone(), e2.arg.geometry[-1].weight.detach().clone()
        pair = e2.arg.pair_encoder[0].weight.detach().clone()
        with torch.no_grad():
            e2.arg.query.weight.mul_(-3); e2.arg.geometry[-1].weight.mul_(-3)
        changed = e2(x).detach()
        assert (trained-changed).abs().max() > 0, 'Support relation has no causal effect on main output'
        with torch.no_grad():
            e2.arg.query.weight.copy_(q); e2.arg.geometry[-1].weight.copy_(geo)
            e2.arg.pair_encoder[0].weight.mul_(-1)
        assert (trained-e2(x).detach()).abs().max() > 0, 'Pair relation has no causal effect on main output'
        with torch.no_grad():
            e2.arg.pair_encoder[0].weight.copy_(pair)
        # v1 scan implementation has the same B/C injection point.
        before = e2(x).detach(); e2.forward_core = e2.forward_corev1
        assert torch.allclose(before, e2(x), atol=1e-6, rtol=1e-5)
        print('PASS official SS2D exact initial output; identical auxiliary initialization; A/delta/B/C zero-init equivalence; E2 head and upstream/main-only gradients; causal relation effect; both scan cores')
    finally:
        if saved is None:
            delattr(vmamba, 'selective_scan_fn')
        else:
            vmamba.selective_scan_fn = saved
        if saved_v1 is None:
            delattr(vmamba, 'selective_scan_fn_v1')
        else:
            vmamba.selective_scan_fn_v1 = saved_v1


def compact_model(mode):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.vmunet = vmamba.VSSM(in_chans=1, num_classes=9, depths=[1, 1, 2, 1],
                                     depths_decoder=[1, 1, 1, 1], dims=[8, 16, 32, 64],
                                     dims_decoder=[64, 32, 16, 8], drop_path_rate=0.)
            self.vmunet.layers[2].blocks[1].self_attention.arg = AnatomicalRelation(64, mode=mode)
        def forward(self, x, return_aux=False):
            return self.vmunet(x, return_aux=return_aux)
    return Wrapper()


def pipeline_checks():
    saved = getattr(vmamba, 'selective_scan_fn', None)
    vmamba.selective_scan_fn = reference_scan
    try:
        for mode in ('E1', 'E2'):
            torch.manual_seed(4)
            model = compact_model(mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            hooks = ModuleAudit(model, ['vmunet']+required_arg_modules(mode))
            audit = ARGTrainingAudit(model, mode)
            criterion = CeDiceLoss(9, [1, 1])
            sums = dict(main=0., semantic=0., presence=0., optimization=0.)
            with tempfile.TemporaryDirectory() as directory:
                for _ in range(2):
                    x, target = torch.randn(2, 1, 32, 32), torch.randint(0, 9, (2, 32, 32))
                    optimizer.zero_grad(set_to_none=True); hooks.begin_batch()
                    out = model(x, return_aux=True)
                    assert out['logits'].shape == (2, 9, 32, 32)
                    assert out['semantic_logits'].shape == (2, 9, 2, 2)
                    assert out['presence_logits'].shape == (2, 8)
                    main = criterion(out['logits'], target)
                    sem, presence = auxiliary_losses(out['semantic_logits'], out['presence_logits'], target)
                    total = main+.1*sem+.05*presence
                    total.backward(); hooks.check_batch(); audit.consume(out); audit.gradients(); optimizer.step()
                    for k, t in zip(sums, (main, sem, presence, total)):
                        sums[k] += t.item()/2
                hooks.finish(1, logging.getLogger('check'), directory)
                audit.finish(1, logging.getLogger('check'), directory, sums)
                rows = list(csv.DictReader(open(Path(directory)/'arg_parameters.csv')))
                assert all(int(r['gradient_steps']) == 2 for r in rows)
                assert any(float(r['epoch_parameter_change_l2']) > 0 for r in rows)
                assert sums['optimization'] > sums['main']
                # Native inference keeps a single Tensor and needs no label.
                model.eval()
                with torch.no_grad():
                    assert model(x).shape == (2, 9, 32, 32)
            hooks.close()
        print('PASS full compact encoder/decoder auxiliary propagation, original CeDiceLoss + optimization backward, required module hooks, per-parameter/epoch CSVs, Tensor-only inference (E1/E2)')
    finally:
        if saved is None:
            delattr(vmamba, 'selective_scan_fn')
        else:
            vmamba.selective_scan_fn = saved


def installation_checks():
    torch.manual_seed(41)
    model = VMUNet(num_classes=9, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1])
    state = {n: p.detach().clone() for n, p in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model.enable_arg('E2', seed=42)
    assert torch.equal(rng, torch.get_rng_state())
    for n, p in state.items():
        assert torch.equal(p, model.state_dict()[n]), n
    assert len([n for n, _ in model.named_modules() if n.endswith('self_attention.arg')]) == 1
    assert ARG_NAME in dict(model.named_modules())
    assert all(n in dict(model.named_modules()) for n in required_arg_modules('E2'))
    try:
        model.enable_arg('E1')
        raise AssertionError('Duplicate installation accepted')
    except RuntimeError:
        pass
    # Check original encoder preload plus mirror-to-decoder mapping, before ARG install.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'synthetic_pretrain.pth'
        key = 'layers.2.blocks.1.self_attention.in_proj.weight'
        value = torch.full_like(model.vmunet.state_dict()[key], .125)
        torch.save({'model': {key: value}}, path)
        model2 = VMUNet(num_classes=9, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1], load_ckpt_path=str(path))
        model2.load_from(); model2.enable_arg('E1')
        assert torch.equal(model2.vmunet.state_dict()[key], value)
        assert torch.equal(model2.vmunet.state_dict()[key.replace('layers.2', 'layers_up.1')], value)
    print('PASS actual VM-UNet Small installation at (2,1); common weights/RNG untouched; original encoder/decoder preload mapping; no duplicate branch')


def engine_checks():
    # Isolate the unrelated MedPy/SimpleITK validation imports on a CPU-only host.
    # Training uses the exact official CeDiceLoss above; no metric is called here.
    previous_utils = sys.modules.get('utils')
    shim = types.ModuleType('utils')
    def never_called(*args, **kwargs):
        raise AssertionError('Training must not read test metrics')
    shim.calculate_metric_percase = never_called
    sys.modules['utils'] = shim
    try:
        spec = importlib.util.spec_from_file_location('arg_engine_check', Path(__file__).with_name('engine_synapse.py'))
        engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    finally:
        if previous_utils is None:
            sys.modules.pop('utils', None)
        else:
            sys.modules['utils'] = previous_utils
    from datasets.multi_organ_aug import MultiOrganRandomGenerator
    from datasets.region_mix import TrainDonorPool, OrganRegionMixer, RegionMixDataset
    from self_check_region_mix import fixture, FixtureRawDataset
    import numpy as np
    saved = getattr(vmamba, 'selective_scan_fn', None)
    vmamba.selective_scan_fn = reference_scan
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, lists, names = fixture(root)
            pool = TrainDonorPool(data, lists, [f'case{c:04d}' for c in range(1, 19)], output_size=(32, 32))
            for mode in ('E1', 'E2'):
                np.random.seed(42); torch.manual_seed(42)
                transform = MultiOrganRandomGenerator((32, 32), preset='v4', probabilities={'region_mix': .2})
                mixer = OrganRegionMixer(pool, probability=.2, refinement='balance')
                dataset = RegionMixDataset(FixtureRawDataset(data, names), transform, mixer)
                loader = torch.utils.data.DataLoader(dataset, batch_size=18, shuffle=True, num_workers=0)
                model = nn.DataParallel(compact_model(mode))
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
                hooks = ModuleAudit(model.module, ['vmunet']+required_arg_modules(mode))
                audit_dir = str(root/mode)
                config = SimpleNamespace(strict_aug_audit=False, arg_experiment=mode,
                                         arg_spec={'lambda_semantic': .1, 'lambda_presence': .05},
                                         amp=False, num_classes=9, print_interval=100, audit_dir=audit_dir)
                for epoch in (1, 2):
                    dataset.set_mix_epoch(epoch)
                    selected_loss = engine.train_one_epoch(loader, model, CeDiceLoss(9, [1, 1]), optimizer,
                                                          scheduler, epoch, logging.getLogger('engine'), config,
                                                          module_audit=hooks)
                    rows = list(csv.DictReader(open(Path(audit_dir)/'training_losses.csv')))
                    assert abs(selected_loss-float(rows[-1]['main'])) < 1e-6
                    assert float(rows[-1]['optimization']) > selected_loss
                    assert mixer.state_dict()['balance']['completed_epochs'] == epoch
                for name in ('arg_epoch.csv', 'arg_parameters.csv', 'training_losses.csv',
                             'module_audit.csv', 'region_mix_balance.csv', 'augmentation_epoch.csv'):
                    assert (Path(audit_dir)/name).is_file(), name
                hooks.close()
        print('PASS actual train_one_epoch E1/E2: online balance loader, two-epoch feedback, original main loss selection, auxiliary optimization, backward/update and all audit files (CPU reference scan)')
    finally:
        if saved is None:
            delattr(vmamba, 'selective_scan_fn')
        else:
            vmamba.selective_scan_fn = saved


def cuda_checks(required):
    if not torch.cuda.is_available():
        if required:
            raise RuntimeError('--cuda-required: CUDA unavailable; native kernel checks did not run')
        print('SKIP native CUDA 224x224/compiled selective scan: CUDA unavailable; run --cuda-required on training host')
        return False
    if not callable(getattr(vmamba, 'selective_scan_fn', None)):
        raise RuntimeError('CUDA available but original mamba_ssm selective_scan_fn is missing')
    for mode in ('E1', 'E2'):
        torch.manual_seed(42)
        net = VMUNet(num_classes=9, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1], drop_path_rate=.2)
        net.enable_arg(mode); net = nn.DataParallel(net.cuda(), device_ids=[0])
        hooks = ModuleAudit(net.module, ['vmunet']+required_arg_modules(mode))
        x = torch.randn(1, 1, 224, 224, device='cuda')
        labels = torch.randint(0, 9, (1, 224, 224), device='cuda')
        net.eval()
        with torch.no_grad():
            expected = net(x)
            out = net(x, return_aux=True)
            assert expected.shape == (1, 9, 224, 224) and out['semantic_logits'].shape == (1, 9, 14, 14)
            assert torch.allclose(expected, out['logits'], atol=1e-6, rtol=1e-5)
            if mode == 'E2':
                att = net.module.vmunet.layers[2].blocks[1].self_attention
                branch = att.arg; del att.arg
                baseline = net(x); att.arg = branch
                assert torch.allclose(expected, baseline, atol=1e-6, rtol=1e-5)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
        criterion = CeDiceLoss(9, [1, 1])
        net.train(); hooks.reset_counts()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True); hooks.begin_batch()
            out = net(x, return_aux=True)
            sem, presence = auxiliary_losses(out['semantic_logits'], out['presence_logits'], labels)
            total = criterion(out['logits'], labels)+.1*sem+.05*presence
            total.backward(); hooks.check_batch()
            branch = dict(net.module.named_modules())[ARG_NAME]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
            optimizer.step()
        if mode == 'E2':
            assert out['diagnostics']['delta_b_rms'].max() > 0 and out['diagnostics']['delta_c_rms'].max() > 0
        hooks.close(); del net, optimizer, out, total
        torch.cuda.empty_cache()
    print('PASS native CUDA VM-UNet Small 224x224 E1/E2: DataParallel auxiliary gathering, compiled scan forward/backward, module hooks, active B/C feedback')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-required', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=FutureWarning)
    torch.set_num_threads(1)
    target_and_gate_checks()
    ss2d_checks(load_reference())
    pipeline_checks()
    installation_checks()
    engine_checks()
    cuda_checks(args.cuda_required)
    print('ARG self-check completed; SKIP lines are unverified checks, not PASS.')
