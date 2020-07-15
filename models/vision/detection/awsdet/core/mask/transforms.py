import tensorflow as tf
import numpy as np
from scipy import interpolate
import cv2
from collections import defaultdict
import pycocotools.mask as mask_util

import tensorflow as tf

def paste_single_mask(box, mask, shape):
    y0 = tf.cast(box[0] + 0.5, tf.int32)
    x0 = tf.cast(box[1] + 0.5, tf.int32)
    y1 = tf.cast(box[2] + 0.5, tf.int32)
    x1 = tf.cast(box[3] + 0.5, tf.int32)
    x1 = tf.math.maximum(x0, x1)    # require at least 1x1
    y1 = tf.math.maximum(y0, y1)
    w = x1 - x0
    h = y1 - y0
    mask = tf.cast(tf.image.resize(mask, (h, w))>0.5, tf.int32)
    int_box = tf.stack([y0, x0, y1, x1])
    pad_shape = [[int_box[0], shape[0] - int_box[2]], [int_box[1], shape[1]-int_box[3]], [0, 0]]
    mask = tf.pad(mask, pad_shape)
    return mask

def paste_masks(boxes, masks, img_metas):
    shape = tf.cast(img_metas[6:8], tf.int32)
    masks = tf.sigmoid(masks)
    mask_count = tf.shape(masks)[0]
    mask_array = tf.TensorArray(tf.int32, size=mask_count)
    for idx in range(mask_count):
        mask_array = mask_array.write(idx, paste_single_mask(boxes[idx], masks[idx], shape))
    mask_array = mask_array.stack()
    return mask_array

def scale_box(box, scale):
    h_half = (box[3] - box[1]) * 0.5
    w_half = (box[2] - box[0]) * 0.5
    y_c = (box[2] + box[0]) * 0.5
    x_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = y_c - w_half
    scaled_box[2] = y_c + w_half
    scaled_box[1] = x_c - h_half
    scaled_box[3] = x_c + h_half
    return scaled_box

def paste_mask_np(box, mask, shape, fast=True, threshold=0.5):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    assert mask.shape[0] == mask.shape[1], mask.shape

    if not fast:
        # This method is accurate but much slower.
        mask = np.pad(mask, [(1, 1), (1, 1)], mode='constant')
        box = scale_box(box, float(mask.shape[0]) / (mask.shape[0] - 2))

        mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
        mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
        h, w = shape
        ys = np.arange(0.0, h) + 0.5
        xs = np.arange(0.0, w) + 0.5
        ys = (ys - box[0]) / (box[2] - box[0]) * mask.shape[0]
        xs = (xs - box[1]) / (box[3] - box[1]) * mask.shape[1]
        # Waste a lot of compute since most indices are out-of-border
        res = mask_continuous(xs, ys)
        return (res >= 0.5).astype('uint8')
    else:
        # This method (inspired by Detectron) is less accurate but fast.

        # int() is floor
        # box fpcoor=0.0 -> intcoor=0.0
        y0, x0 = list(map(int, box[:2] + 0.5))
        # box fpcoor=h -> intcoor=h-1, inclusive
        y1, x1 = list(map(int, box[2:] - 0.5))    # inclusive
        x1 = max(x0, x1)    # require at least 1x1
        y1 = max(y0, y1)

        w = x1 + 1 - x0
        h = y1 + 1 - y0

        # rounding errors could happen here, because masks were not originally computed for this shape.
        # but it's hard to do better, because the network does not know the "original" scale
        mask = (cv2.resize(mask, (w, h)) > threshold).astype('uint8')
        ret = np.zeros(shape, dtype='uint8')
        ret[y0:y1 + 1, x0:x1 + 1] = mask
        return ret
    
def mask2result(masks, bboxes, labels, meta, num_classes=80, threshold=0.5, fast=True):
    if tf.is_tensor(bboxes):
        bboxes = bboxes.numpy()
    if tf.is_tensor(masks):
        masks = tf.squeeze(masks).numpy()
    if tf.is_tensor(meta):
        meta = meta.numpy().astype(np.int32)
    if tf.is_tensor(labels):
        labels = labels.numpy()
    padded_shape = meta[6:8]
    unpadded_height = meta[3]
    unpadded_width = meta[4]
    original_height = meta[0]
    original_width = meta[1]
    masks = np.array([paste_mask_np(box, mask, padded_shape, 
                                 threshold=threshold, fast=fast) for box, mask in zip(bboxes, masks)])
    masks = masks[:,:unpadded_height,:unpadded_width]
    masks = np.array([cv2.resize(mask, (original_width, original_height)) for mask in masks])
    if meta[-1]==1:
        masks_np = np.flip(masks_np, axis=2)
    lists = defaultdict(list)
    for i,j in enumerate(labels):
        lists[j].append(mask_util.encode(
                    np.array(
                        masks[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])
    return masks