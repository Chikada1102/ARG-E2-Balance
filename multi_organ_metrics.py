"""Explicit per-case/per-class overlap and surface-voxel metrics."""
import numpy as np
from scipy import ndimage

METRICS = ("dice", "iou", "precision", "recall", "hd95", "assd", "surface_dice")


def binary_metrics(prediction, truth, spacing=None, tolerance=1.):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or prediction.ndim not in (2, 3):
        raise ValueError("Expected matching 2D/3D masks")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Surface tolerance must be positive")
    sampling = tuple(spacing) if spacing is not None else (1.,)*truth.ndim
    if len(sampling) != truth.ndim or any(s <= 0 or not np.isfinite(s) for s in sampling):
        raise ValueError("Invalid voxel spacing")
    npred, ngt = int(prediction.sum()), int(truth.sum())
    info = dict(gt_voxels=ngt, pred_voxels=npred, both_empty=int(npred == ngt == 0),
                missed=int(ngt > 0 and npred == 0), false_positive_only=int(ngt == 0 and npred > 0))
    if npred == ngt == 0:
        return dict(info, **dict.fromkeys(METRICS, float("nan")))
    intersection = int(np.count_nonzero(prediction & truth))
    result = dict(info, dice=2*intersection/(npred+ngt),
                  iou=intersection/(npred+ngt-intersection),
                  precision=intersection/npred if npred else 0.,
                  recall=intersection/ngt if ngt else 0.)
    if not npred or not ngt:
        return dict(result, hd95=float("inf"), assd=float("inf"), surface_dice=0.)
    structure = ndimage.generate_binary_structure(truth.ndim, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure, border_value=0)
    gt_surface = truth ^ ndimage.binary_erosion(truth, structure, border_value=0)
    to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=sampling)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=sampling)[gt_surface]
    distances = np.concatenate((to_gt, to_pred))
    return dict(result, hd95=float(np.percentile(distances, 95)),
                assd=float(distances.mean()),
                surface_dice=float(((to_gt <= tolerance).sum()+(to_pred <= tolerance).sum()) /
                                   (len(to_gt)+len(to_pred))))


def summarize(rows, classes):
    summary = []
    for class_id in range(1, classes):
        group = [r for r in rows if r["class_id"] == class_id and not r["both_empty"]]
        result = dict(class_id=class_id, evaluated_cases=len(group),
                      both_empty_cases=sum(r["both_empty"] for r in rows if r["class_id"] == class_id),
                      empty_mask_failures=sum(r["missed"]+r["false_positive_only"] for r in group))
        for name in METRICS:
            values = np.array([r[name] for r in group], dtype=float)
            # Infinite distances remain infinite in the primary aggregate.
            result[name] = float(values.mean()) if len(values) else float("nan")
        for name in ("hd95", "assd"):
            finite = [r[name] for r in group if np.isfinite(r[name])]
            result[name+"_finite_only"] = float(np.mean(finite)) if finite else float("nan")
        summary.append(result)
    dices = np.array([r["dice"] for r in summary if np.isfinite(r["dice"])])
    macro = {name: float(np.mean([r[name] for r in summary if not np.isnan(r[name])]))
             if any(not np.isnan(r[name]) for r in summary) else float("nan") for name in METRICS}
    macro.update(dice_std_across_organs=float(dices.std()) if len(dices) else float("nan"),
                 worst_organ_dice=float(dices.min()) if len(dices) else float("nan"),
                 empty_mask_failures=sum(r["empty_mask_failures"] for r in summary))
    return summary, macro
