"""Training-consumer ARG diagnostics. Never used to choose weights or gates."""
import os
import torch
from innovation_audit import append_csv
from models.vmunet.arg_vss import ARG_NAME


class ARGTrainingAudit:
    def __init__(self, model, mode):
        self.mode = mode
        self.module = dict(model.named_modules())[ARG_NAME]
        if self.module.mode != mode:
            raise RuntimeError('ARG experiment mismatch')
        self.parameters = dict(self.module.named_parameters())
        self.initial = {n: p.detach().clone() for n, p in self.parameters.items()}
        self.grad_steps = {n: 0 for n in self.parameters}
        self.nonzero_steps = {n: 0 for n in self.parameters}
        self.grad_norms = {n: 0. for n in self.parameters}
        self.values, self.samples, self.batches = {}, 0, 0

    def consume(self, aux):
        size = aux['semantic_logits'].shape[0]
        for name, value in aux['diagnostics'].items():
            value = value.detach().float()
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite or incorrectly gathered ARG diagnostics: '+name)
            self.values[name] = self.values.get(name, 0.) + value.sum().item()
        self.samples += size
        self.batches += 1

    def gradients(self):
        for name, p in self.parameters.items():
            if p.grad is None:
                raise RuntimeError('ARG parameter disconnected from optimization loss: '+name)
            norm = p.grad.detach().float().norm().item()
            if not torch.isfinite(p.grad).all():
                raise RuntimeError('Nonfinite ARG gradient: '+name)
            self.grad_steps[name] += 1
            self.nonzero_steps[name] += int(norm > 0)
            self.grad_norms[name] += norm

    def finish(self, epoch, logger, directory, losses):
        if not self.samples:
            raise RuntimeError('No ARG training outputs consumed')
        summary = dict(epoch=epoch, experiment=self.mode, batches=self.batches, samples=self.samples,
                       **{k: v/self.samples for k, v in self.values.items()}, **losses)
        rows = []
        for name, p in self.parameters.items():
            delta = (p.detach()-self.initial[name]).float().norm().item()
            rows.append(dict(epoch=epoch, experiment=self.mode, parameter=name,
                             gradient_steps=self.grad_steps[name], nonzero_gradient_steps=self.nonzero_steps[name],
                             mean_gradient_l2=self.grad_norms[name]/self.batches, epoch_parameter_change_l2=delta))
        append_csv(os.path.join(directory, 'arg_epoch.csv'), [summary])
        append_csv(os.path.join(directory, 'arg_parameters.csv'), rows)
        append_csv(os.path.join(directory, 'training_losses.csv'),
                   [dict(epoch=epoch, experiment=self.mode, selection_metric='main', **losses)])
        logger.info('ARG_AUDIT '+str(summary))
