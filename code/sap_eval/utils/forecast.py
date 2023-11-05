import torch
import numpy as np
import pycocotools.mask as maskUtils

def extrap_clean_up(bboxes, w_img, h_img, min_size=75, lt=False):
    # bboxes in the format of [cx or l, cy or t, w, h]
    wh_nz = bboxes[:, 2:] > 0
    keep = np.logical_and(wh_nz[:, 0], wh_nz[:, 1])

    if lt:
        # convert [l, t, w, h] to [l, t, r, b]
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    else:
        # convert [cx, cy, w, h] to [l, t, r, b]
        bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:]/2
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]

    # clip to the image
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w_img)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h_img)

    # convert [l, t, r, b] to [l, t, w, h]
    bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]

    # int conversion is neccessary, otherwise, there are very small w, h that round up to 0
    keep = np.logical_and(keep, bboxes[:, 2].astype(np.int)*bboxes[:, 3].astype(np.int) >= min_size)
    bboxes = bboxes[keep]
    return bboxes, keep

def iou_assoc(bboxes1, labels1, tracks1, tkidx, bboxes2, labels2, match_iou_th, no_unmatched1=False):
    # iou-based association
    # shuffle all elements so that matched stays in the front
    # bboxes are in the form of a list of [l, t, w, h]
    m, n = len(bboxes1), len(bboxes2)
        
    _ = n*[0]
    ious = maskUtils.iou(bboxes1, bboxes2, _)

    match_fwd = m*[None]
    matched1 = []
    matched2 = []
    unmatched2 = []

    for j in range(n):
        best_iou = match_iou_th
        match_i = None
        for i in range(m):
            if match_fwd[i] is not None \
                or labels1[i] != labels2[j] \
                or ious[i, j] < best_iou:
                continue
            best_iou = ious[i, j]
            match_i = i
        if match_i is None:
            unmatched2.append(j)
        else:
            matched1.append(match_i)
            matched2.append(j)
            match_fwd[match_i] = j

    if no_unmatched1:
        order1 = matched1
    else:
        unmatched1 = list(set(range(m)) - set(matched1))
        order1 = matched1 + unmatched1
    order2 = matched2 + unmatched2

    n_matched = len(matched2)
    n_unmatched2 = len(unmatched2)
    tracks2 = np.concatenate((tracks1[order1][:n_matched],
        np.arange(tkidx, tkidx + n_unmatched2, dtype=tracks1.dtype)))
    tkidx += n_unmatched2

    return order1, order2, n_matched, tracks2, tkidx

def ltwh2ltrb_(bboxes):
    if len(bboxes):
        if bboxes.ndim == 1:
            bboxes[2:] += bboxes[:2]
        else:
            bboxes[:, 2:] += bboxes[:, :2]
    return bboxes

def ltrb2ltwh_(bboxes):
    if len(bboxes):
        if bboxes.ndim == 1:
            bboxes[2:] -= bboxes[:2]
        else:
            bboxes[:, 2:] -= bboxes[:, :2]
    return bboxes

def ltrb2ltwh(bboxes):
    bboxes = bboxes.copy()
    return ltrb2ltwh_(bboxes)

def bbox2z(bboxes):
    return torch.from_numpy(bboxes).unsqueeze_(2)

def bbox2x(bboxes):
    x = torch.cat((torch.from_numpy(bboxes), torch.zeros(bboxes.shape)), dim=1)
    return x.unsqueeze_(2)

def x2bbox(x):
    return x[:, :4, 0].numpy()

def make_F(F, dt):
    F[[0, 1, 2, 3], [4, 5, 6, 7]] = dt
    return F

def make_Q(Q, dt):
    # assume the base Q is identity
    Q[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]] = dt*dt
    return Q

def batch_kf_predict_only(F, x):
    return F @ x

def batch_kf_predict(F, x, P, Q):
    x = F @ x
    P = F @ P @ F.t() + Q
    return x, P

def batch_kf_update(z, x, P, R):
    # assume H is just slicing operation
    # y = z - Hx
    y = z - x[:, :4]

    # S = HPH' + R
    S = P[:, :4, :4] + R

    # K = PH'S^(-1)
    K = P[:, :, :4] @ S.inverse()

    # x = x + Ky
    x += K @ y

    # P = (I - KH)P
    P -= K @ P[:, :4]
    return x, P