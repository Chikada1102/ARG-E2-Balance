"""Official training objective + mandatory audit and extended volume evaluation."""
import os
import time
import uuid
import numpy as np
import torch
from torch.cuda.amp import autocast
from scipy import ndimage
from tqdm import tqdm
from utils import calculate_metric_percase
from innovation_audit import AugmentationAudit, append_csv
from multi_organ_metrics import binary_metrics, summarize
from arg_losses import auxiliary_losses
from arg_audit import ARGTrainingAudit


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                    epoch, logger, config, scaler=None, module_audit=None):
    start = time.time()
    model.train()
    if module_audit:
        module_audit.reset_counts()
    audit = AugmentationAudit(train_loader.dataset.transform, strict=config.strict_aug_audit)
    arg_audit = ARGTrainingAudit(model.module, config.arg_experiment)
    sums = dict(main=0., semantic=0., presence=0., optimization=0.)
    total_loss, batches = 0., 0
    for iteration, data in enumerate(train_loader):
        # Consumer-side aggregation works with num_workers=0,32,... alike.
        audit.consume(data)
        optimizer.zero_grad(set_to_none=True)
        device = next(model.parameters()).device
        images = data["image"].to(device, non_blocking=True).float()
        targets = data["label"].to(device, non_blocking=True).long()
        if module_audit:
            module_audit.begin_batch()
        with autocast(enabled=config.amp):
            aux = model(images, return_aux=True)
            if not isinstance(aux, dict) or set(aux) != {'logits', 'semantic_logits', 'presence_logits', 'diagnostics'}:
                raise RuntimeError('ARG training must explicitly return every supervised output')
            output = aux['logits']
            if (not isinstance(output, torch.Tensor) or output.ndim != 4 or
                    output.shape[1] != config.num_classes or output.shape[2:] != targets.shape[1:]):
                raise RuntimeError("Expected [B,num_classes,H,W] logits; no auxiliary output is silently ignored")
            main_loss = criterion(output, targets)
            semantic_loss, presence_loss = auxiliary_losses(aux['semantic_logits'], aux['presence_logits'], targets)
            loss = main_loss + config.arg_spec['lambda_semantic']*semantic_loss + config.arg_spec['lambda_presence']*presence_loss
        arg_audit.consume(aux)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        if config.amp:
            scaler.scale(loss).backward()
            if module_audit:
                module_audit.check_batch()
            scaler.unscale_(optimizer)
            arg_audit.gradients()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if module_audit:
                module_audit.check_batch()
            arg_audit.gradients()
            optimizer.step()
        total_loss += main_loss.item()
        for key, value in (('main', main_loss), ('semantic', semantic_loss), ('presence', presence_loss), ('optimization', loss)):
            sums[key] += value.detach().item()
        batches += 1
        if iteration % config.print_interval == 0:
            message = f"train: epoch {epoch}, iter:{iteration}, main:{main_loss.item():.4f}, semantic:{semantic_loss.item():.4f}, presence:{presence_loss.item():.4f}, optimization:{loss.item():.4f}, lr:{optimizer.param_groups[0]['lr']}"
            print(message)
            logger.info(message)
    if not batches:
        raise RuntimeError("Empty training loader")
    audit.finish(epoch, logger, config.audit_dir)
    if hasattr(train_loader.dataset, 'finish_mix_epoch'):
        if train_loader.persistent_workers:
            raise RuntimeError('Epoch balance feedback requires persistent_workers=False')
        train_loader.dataset.finish_mix_epoch(audit.region, epoch, logger, config.audit_dir)
    if module_audit:
        module_audit.finish(epoch, logger, config.audit_dir)
    arg_audit.finish(epoch, logger, config.audit_dir, {k: v/batches for k, v in sums.items()})
    scheduler.step()
    mean = total_loss/batches
    logger.info("Finish one epoch train: epoch %d loss %.6f time(s) %.2f", epoch, mean, time.time()-start)
    return mean


def predict_volume(image, model, patch_size, classes):
    device = next(model.parameters()).device
    prediction = np.zeros(image.shape, dtype=np.int64)
    slices = image[None] if image.ndim == 2 else image
    outputs = prediction[None] if image.ndim == 2 else prediction
    for index, plane in enumerate(slices):
        h, w = plane.shape
        if (h, w) != tuple(patch_size):
            plane = ndimage.zoom(plane, (patch_size[0]/h, patch_size[1]/w), order=3)
        tensor = torch.from_numpy(np.ascontiguousarray(plane)).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(tensor)
        if not isinstance(logits, torch.Tensor) or logits.shape != (1, classes, *patch_size):
            raise RuntimeError("Unexpected validation output shape")
        mask = logits.argmax(dim=1)[0].cpu().numpy()
        if mask.shape != (h, w):
            mask = ndimage.zoom(mask, (h/mask.shape[0], w/mask.shape[1]), order=0)
        outputs[index] = mask
    return prediction


def val_one_epoch(test_datasets, test_loader, model, epoch, logger, config,
                  test_save_path, val_or_test=False):
    model.eval()
    rows, legacy = [], []
    split = "test" if val_or_test else "eval"
    evaluation = dict(eval_id=uuid.uuid4().hex[:12],
                      stage=getattr(config, "eval_stage", split),
                      weight_epoch=getattr(config, "eval_weight_epoch", epoch),
                      checkpoint=getattr(config, "eval_checkpoint", "current"))
    with torch.no_grad():
        for data in tqdm(test_loader):
            image = data["image"][0].cpu().numpy()
            label = data["label"][0].cpu().numpy().astype(np.int64)
            if image.shape != label.shape or image.ndim not in (2, 3):
                raise ValueError("Expected matching 2D/3D validation volumes")
            case = data["case_name"][0]
            prediction = predict_volume(image, model, [config.input_size_h, config.input_size_w], config.num_classes)
            spacing = config.metric_spacing[-image.ndim:] if config.metric_spacing else None
            case_legacy = []
            for c in range(1, config.num_classes):
                # Unchanged official empty-mask conventions, reported separately.
                old = calculate_metric_percase((prediction == c).copy(), (label == c).copy())
                case_legacy.append(old)
                metrics = binary_metrics(prediction == c, label == c, spacing,
                                         tolerance=config.surface_tolerance)
                rows.append(dict(epoch=epoch, split=split, **evaluation, case=case, class_id=c,
                                 distance_unit="mm" if spacing else "voxel",
                                 surface_tolerance=config.surface_tolerance,
                                 legacy_dice=old[0], legacy_hd95=old[1], **metrics))
            legacy.append(case_legacy)
            if val_or_test and test_save_path:
                import SimpleITK as sitk
                os.makedirs(test_save_path, exist_ok=True)
                for name, array in (("pred", prediction), ("img", image), ("gt", label)):
                    itk = sitk.GetImageFromArray(array.astype(np.float32))
                    physical = spacing or ((config.z_spacing, 1., 1.) if image.ndim == 3 else (1., 1.))
                    itk.SetSpacing(tuple(reversed(physical)))
                    sitk.WriteImage(itk, os.path.join(test_save_path, f"{case}_{name}.nii.gz"))
    if not rows:
        raise RuntimeError("Empty evaluation loader")
    if len(legacy) != len(test_datasets):
        raise RuntimeError("Evaluation did not cover every volume")
    summary, macro = summarize(rows, config.num_classes)
    append_csv(os.path.join(config.audit_dir, "metrics_per_case.csv"), rows)
    append_csv(os.path.join(config.audit_dir, "metrics_per_organ.csv"),
               [dict(epoch=epoch, split=split, **evaluation, **row) for row in summary])
    append_csv(os.path.join(config.audit_dir, "metrics_summary.csv"),
               [dict(epoch=epoch, split=split, **evaluation, **macro)])
    for row in summary:
        logger.info("STRICT_METRICS " + str(row))
    logger.info("STRICT_MACRO " + str(macro))
    array = np.asarray(legacy).mean(axis=0)
    for index, pair in enumerate(array, 1):
        logger.info("Mean class %d legacy_dice %.6f legacy_hd95 %.6f", index, pair[0], pair[1])
    mean_dice, mean_hd95 = array.mean(axis=0)
    message = f"eval epoch {epoch} stage={evaluation['stage']} weight_epoch={evaluation['weight_epoch']} checkpoint={evaluation['checkpoint']}: legacy_mean_dice={mean_dice:.6f} legacy_mean_hd95={mean_hd95:.6f}; strict_mean_dice={macro['dice']:.6f}"
    print(message)
    logger.info(message)
    return float(mean_dice), float(mean_hd95)
