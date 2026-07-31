import numpy as np
from skimage.measure import euler_number
from skimage.measure import label
from skimage.morphology import skeletonize

__all__ = [
    "dice_coefficient",
    "iou",
    "sensitivity",
    "specificity",
    "cl_dice",
    "betti_numbers",
    "betti_error",
    "evaluate",
]


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return 1.0

    return float(numerator) / float(denominator)


def _binarize(mask, threshold=0.5):
    mask = np.asarray(mask)

    if mask.dtype == bool:
        return mask

    mask = mask.astype(np.float64)

    if mask.size > 0 and mask.max() > 1.0:
        mask = mask / 255.0

    return mask > threshold


def _confusion_counts(pred, gt, roi=None, threshold=0.5):
    pred_b = _binarize(pred, threshold)
    gt_b = _binarize(gt, threshold)

    if roi is not None:
        roi_b = _binarize(roi, threshold)
        pred_b = pred_b[roi_b]
        gt_b = gt_b[roi_b]

    tp = int(np.count_nonzero(pred_b & gt_b))
    tn = int(np.count_nonzero(~pred_b & ~gt_b))
    fp = int(np.count_nonzero(pred_b & ~gt_b))
    fn = int(np.count_nonzero(~pred_b & gt_b))

    return tp, tn, fp, fn


def dice_coefficient(pred, gt, roi=None, threshold=0.5):
    tp, tn, fp, fn = _confusion_counts(pred, gt, roi, threshold)

    return _safe_ratio(2 * tp, 2 * tp + fp + fn)


def iou(pred, gt, roi=None, threshold=0.5):
    tp, tn, fp, fn = _confusion_counts(pred, gt, roi, threshold)

    return _safe_ratio(tp, tp + fp + fn)


def sensitivity(pred, gt, roi=None, threshold=0.5):
    tp, tn, fp, fn = _confusion_counts(pred, gt, roi, threshold)

    return _safe_ratio(tp, tp + fn)


def specificity(pred, gt, roi=None, threshold=0.5):
    tp, tn, fp, fn = _confusion_counts(pred, gt, roi, threshold)

    return _safe_ratio(tn, tn + fp)


def _cl_score(volume, skeleton):
    return _safe_ratio(np.count_nonzero(volume & skeleton), np.count_nonzero(skeleton))


def cl_dice(pred, gt, roi=None, threshold=0.5):
    pred_b = _binarize(pred, threshold)
    gt_b = _binarize(gt, threshold)

    if roi is not None:
        roi_b = _binarize(roi, threshold)
        pred_b = pred_b & roi_b
        gt_b = gt_b & roi_b

    pred_empty = not pred_b.any()
    gt_empty = not gt_b.any()

    if pred_empty and gt_empty:
        return 1.0

    if pred_empty or gt_empty:
        return 0.0

    skel_pred = skeletonize(pred_b)
    skel_gt = skeletonize(gt_b)
    t_prec = _cl_score(pred_b, skel_gt)
    t_sens = _cl_score(gt_b, skel_pred)
    denominator = t_prec + t_sens

    if denominator == 0.0:
        return 0.0

    return 2.0 * t_prec * t_sens / denominator


def betti_numbers(mask, threshold=0.5, connectivity=2):
    mask_b = _binarize(mask, threshold)
    _, beta_0 = label(mask_b, connectivity=connectivity, return_num=True)
    chi = euler_number(mask_b, connectivity=connectivity)
    beta_1 = int(beta_0) - int(chi)

    return int(beta_0), int(beta_1)


def betti_error(pred, gt, threshold=0.5, connectivity=2):
    b0_pred, b1_pred = betti_numbers(pred, threshold, connectivity)
    b0_gt, b1_gt = betti_numbers(gt, threshold, connectivity)

    return abs(b0_pred - b0_gt), abs(b1_pred - b1_gt)


def evaluate(pred, gt, roi=None, threshold=0.5, connectivity=2):
    b0_pred, b1_pred = betti_numbers(pred, threshold, connectivity)
    b0_gt, b1_gt = betti_numbers(gt, threshold, connectivity)

    return {
        "dice": dice_coefficient(pred, gt, roi, threshold),
        "iou": iou(pred, gt, roi, threshold),
        "cldice": cl_dice(pred, gt, roi, threshold),
        "sensitivity": sensitivity(pred, gt, roi, threshold),
        "specificity": specificity(pred, gt, roi, threshold),
        "betti0": b0_pred,
        "betti1": b1_pred,
        "betti0_error": abs(b0_pred - b0_gt),
        "betti1_error": abs(b1_pred - b1_gt),
    }
