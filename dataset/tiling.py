import numpy as np
from skimage.transform import resize as _sk_resize

from util.evaluate_metrics import evaluate

__all__ = ["sliding_window_predict", "upsample_to_native", "score_native"]


def _hann_window(size):
    w = np.hanning(size)
    win = np.outer(w, w).astype(np.float32)

    return np.clip(win, 1e-3, None)


def sliding_window_predict(predict_fn, image, tile=512, overlap=0.25):
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)

    if pad_h or pad_w:
        mode = "reflect" if (pad_h < h and pad_w < w) else "edge"
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode=mode)

    hp, wp = image.shape
    stride = max(1, int(round(tile * (1.0 - overlap))))
    ys = sorted(set(list(range(0, hp - tile + 1, stride)) + [hp - tile]))
    xs = sorted(set(list(range(0, wp - tile + 1, stride)) + [wp - tile]))
    window = _hann_window(tile)
    acc = np.zeros((hp, wp), dtype=np.float32)
    weight = np.zeros((hp, wp), dtype=np.float32)

    for y in ys:

        for x in xs:
            prob = np.asarray(predict_fn(image[y:y + tile, x:x + tile]), dtype=np.float32)
            acc[y:y + tile, x:x + tile] += prob * window
            weight[y:y + tile, x:x + tile] += window

    blended = acc / np.maximum(weight, 1e-6)

    return blended[:h, :w]


def upsample_to_native(prob, native_shape):
    out = _sk_resize(prob, native_shape, order=1, anti_aliasing=True, preserve_range=True)

    return out.astype(np.float32)


def score_native(prob_scaled, native_shape, native_label, native_fov, threshold=0.5):
    prob_native = upsample_to_native(prob_scaled, native_shape)

    return evaluate(prob_native, native_label, roi=native_fov, threshold=threshold)
