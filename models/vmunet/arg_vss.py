"""ARG-VSS: E1 auxiliary supervision; E2 support-write / pair-read feedback.

Forward accepts features only. Labels are used exclusively by arg_losses.py.
No external adjacency/template, dataset statistics, EMA or test-time labels.
"""
import math
import torch
from torch import nn

ARG_SITE = (2, 1)  # encoder Stage 3, last block; zero-based
ARG_NAME = 'vmunet.layers.2.blocks.1.self_attention.arg'
ARG_VERSION = 'arg-vss-1'


def pack4(x):
    """BCHW -> B,4,C,HW, exactly matching official SS2D scan order."""
    b, c, h, w = x.shape
    forward = torch.stack((x.reshape(b, c, h*w),
                           x.transpose(2, 3).contiguous().reshape(b, c, h*w)), 1)
    return torch.cat((forward, forward.flip(-1)), 1)


class PixelFeatures(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.act(self.norm(x)).permute(0, 3, 1, 2).contiguous()


class DirectionalProjection(nn.Module):
    def __init__(self, dim, states):
        super().__init__()
        # Only these final projections are zero-initialized. Alpha stays 0.1.
        self.weight = nn.Parameter(torch.zeros(4, states, dim))
        self.bias = nn.Parameter(torch.zeros(4, states))

    def forward(self, field):
        return torch.einsum('bkdl,knd->bknl', pack4(field), self.weight) + self.bias[None, :, :, None]


class AnatomicalRelation(nn.Module):
    def __init__(self, channels, classes=9, dim=64, states=16, mode='E2', alpha=.1):
        super().__init__()
        if mode not in ('E1', 'E2') or classes < 3 or dim < 1 or states < 1:
            raise ValueError('ARG requires E1/E2, >=2 foreground classes and positive dimensions')
        if not math.isfinite(alpha) or not 0 < alpha <= .1:
            raise ValueError('ARG alpha must be in (0, 0.1]')
        self.mode, self.classes, self.dim, self.states, self.alpha = mode, classes, dim, states, float(alpha)
        self.eps = 1e-6
        self.feature = PixelFeatures(channels, dim)
        self.semantic_head = nn.Conv2d(dim, classes, 1)
        self.presence_head = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.Linear(dim, classes-1))
        if mode == 'E2':
            self.query = nn.Linear(dim, dim, bias=False)
            self.key = nn.Linear(dim, dim, bias=False)
            self.geometry = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
            self.pair_encoder = nn.Sequential(nn.Linear(3*dim+4, dim), nn.GELU(), nn.Linear(dim, dim))
            pairs = torch.triu_indices(classes-1, classes-1, offset=1)
            self.register_buffer('pair_i', pairs[0], persistent=False)
            self.register_buffer('pair_j', pairs[1], persistent=False)
            self.write_head = DirectionalProjection(dim, states)
            self.read_head = DirectionalProjection(dim, states)

    def relation_fields(self, v, probabilities, confidence):
        """Differentiable soft organ graph; confidence=0 gives exactly zero feedback."""
        b, d, h, w = v.shape
        p = probabilities[:, 1:].flatten(2)  # exclude background
        g = confidence
        mass = p.sum(-1)
        z = torch.einsum('bkn,bdn->bkd', p, v.flatten(2)) / (mass[..., None] + self.eps)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=v.device, dtype=v.dtype),
                                torch.linspace(-1, 1, w, device=v.device, dtype=v.dtype), indexing='ij')
        coords = torch.stack((xx.flatten(), yy.flatten()), -1)
        centers = torch.einsum('bkn,nt->bkt', p, coords) / (mass[..., None] + self.eps)
        area = mass / (h*w)
        diff = centers[:, None, :, :] - centers[:, :, None, :]
        log_ratio = torch.log((area[:, None, :] + self.eps) / (area[:, :, None] + self.eps))
        distance = (diff.square().sum(-1) + self.eps).sqrt()
        geometry = torch.cat((diff, log_ratio[..., None], distance[..., None]), -1)
        score = torch.einsum('bid,bjd->bij', self.query(z), self.key(z)) / math.sqrt(d)
        score = score + self.geometry(geometry).squeeze(-1)
        raw = (score - score.amax(-1, keepdim=True)).exp() * g[:, None, :]
        relation = raw / (raw.sum(-1, keepdim=True) + self.eps)
        support_nodes = torch.einsum('bij,bjd->bid', relation, z)
        foreground = p * g[..., None]
        support = torch.einsum('bkn,bkd->bdn', foreground, support_nodes)
        support = support / (foreground.sum(1, keepdim=True) + self.eps)
        rho_b = foreground.sum(1, keepdim=True).clamp(0, 1)
        i, j = self.pair_i, self.pair_j
        pair_nodes = self.pair_encoder(torch.cat((z[:, i], z[:, j], z[:, i]-z[:, j], geometry[:, i, j]), -1))
        pair_weights = 4*p[:, i]*p[:, j]*g[:, i, None]*g[:, j, None]
        pair_sum = pair_weights.sum(1, keepdim=True)
        contrast = torch.einsum('bpn,bpd->bdn', pair_weights, pair_nodes) / (pair_sum + self.eps)
        rho_c = pair_sum.clamp(0, 1)
        offdiag = relation.sum((1, 2)) - relation.diagonal(dim1=1, dim2=2).sum(1)
        return dict(support=support.reshape(b, d, h, w), contrast=contrast.reshape(b, d, h, w),
                    rho_b=rho_b.reshape(b, 1, h, w), rho_c=rho_c.reshape(b, 1, h, w),
                    relation=relation, offdiag=offdiag/(self.classes-1))

    def forward(self, u):
        # Softmax/prototype accumulation and scan parameters are kept in float32.
        with torch.cuda.amp.autocast(enabled=False):
            v = self.feature(u.float())
            logits = self.semantic_head(v)
            presence = self.presence_head(torch.cat((v.mean((2, 3)), v.amax((2, 3))), 1))
            probabilities, confidence = logits.softmax(1), presence.sigmoid()
            zeros = u.new_zeros((u.shape[0],), dtype=torch.float32)
            diagnostics = dict(presence_mean=confidence.mean(1).detach(),
                               semantic_entropy=(-(probabilities*probabilities.clamp_min(self.eps).log()).sum(1)).mean((1, 2)).detach(),
                               relation_offdiag=zeros, write_gate=zeros, read_gate=zeros,
                               delta_b_rms=zeros, delta_c_rms=zeros, delta_b_ratio=zeros, delta_c_ratio=zeros)
            result = dict(semantic_logits=logits, presence_logits=presence, diagnostics=diagnostics)
            if self.mode == 'E2':
                fields = self.relation_fields(v, probabilities, confidence)
                delta_b = self.alpha * pack4(fields['rho_b']) * self.write_head(fields['support']).tanh()
                delta_c = self.alpha * pack4(fields['rho_c']) * self.read_head(fields['contrast']).tanh()
                result.update(delta_B=delta_b, delta_C=delta_c)
                diagnostics.update(relation_offdiag=fields['offdiag'].detach(),
                                   write_gate=fields['rho_b'].mean((1, 2, 3)).detach(),
                                   read_gate=fields['rho_c'].mean((1, 2, 3)).detach(),
                                   delta_b_rms=delta_b.detach().square().mean((1, 2, 3)).sqrt(),
                                   delta_c_rms=delta_c.detach().square().mean((1, 2, 3)).sqrt())
            return result


def arg_spec(mode):
    if mode not in ('E1', 'E2'):
        raise ValueError('experiment must be E1 or E2')
    return dict(version=ARG_VERSION, experiment=mode, site=list(ARG_SITE), feature_dim=64,
                state_dim=16, alpha=.1, lambda_semantic=.1, lambda_presence=.05,
                auxiliary_target='post_augmentation_area_pool', selection='min_main_train_loss')


def required_arg_modules(mode):
    base = [ARG_NAME+'.'+s for s in ('feature', 'semantic_head', 'presence_head')]
    if mode == 'E2':
        base += [ARG_NAME+'.'+s for s in ('query', 'key', 'geometry', 'pair_encoder', 'write_head', 'read_head')]
    return base
