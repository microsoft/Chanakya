import os
import numpy as np

from PIL import Image
import pickle, json, cv2, mmcv

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Empirical():
    def __init__(self, samples, perf_factor=1):
        self.samples = np.array(samples)
        assert perf_factor > 0, perf_factor
        if perf_factor != 1:
            self.samples /= perf_factor
        self.sidx = 0

    def draw(self):
        return np.random.choice(self.samples)

    def draw_sequential(self):
        sample = self.samples[self.sidx]
        self.sidx = (self.sidx + 1) % len(self.samples)
        return sample

    def mean(self):
        return self.samples.mean()

    def std(self):
        return self.samples.std(ddof=1)
    
    def min(self):
        return self.samples.min()

    def max(self):
        return self.samples.max()

def dist_from_dict(dist_dict, perf_factor=1):
    if dist_dict['type'] == 'empirical':
        return Empirical(dist_dict['samples'], perf_factor)
    else:
        raise ValueError(f'Unknown distribution type "{dist_dict["type"]}"')

def mkdir2(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def print_stats(var, name='', fmt='%.3g', cvt=lambda x: x):
    var = np.asarray(var)
    
    if name:
        prefix = name + ': '
    else:
        prefix = ''

    if len(var) == 1:
        print(('%sscalar: ' + fmt) % (
            prefix,
            cvt(var[0]),
        ))
    else:
        fmt_str = 'mean: %s; std: %s; min: %s; max: %s' % (
            fmt, fmt, fmt, fmt
        )
        print(('%s' + fmt_str) % (
            prefix,
            cvt(var.mean()),
            cvt(var.std(ddof=1)),
            cvt(var.min()),
            cvt(var.max()),
        ))

from os.path import dirname

def imread(path, method='PIL'):
    if method == 'PIL':
        # using "array" istead of "asarray" since
        # "torch.from_numpy" requires writeable array in PyTorch 1.6
        return np.array(Image.open(path))
    else:
        return mmcv.imread(path)

def imwrite(img, path, method='PIL', auto_mkdir=True):
    if method == 'PIL':
        if auto_mkdir:
            mkdir2(dirname(path))
        Image.fromarray(img).save(path)
    else:
        mmcv.imwrite(img, path, auto_mkdir=auto_mkdir)

# parse detector output into format required by evaluation server
def parse_det_result(result, class_mapping=None, n_class=None, separate_scores=True, return_sel=False):
    if len(result) > 2:
        bboxes_scores, labels, masks = result
    else:
        bboxes_scores, labels = result
        masks = None

    if class_mapping is not None:
        labels = class_mapping[labels]
        sel = labels < n_class
        bboxes_scores = bboxes_scores[sel]
        labels = labels[sel]
        if masks is not None:
            masks = masks[sel]
    else:
        sel = None
    if separate_scores:
        if len(labels):
            bboxes = bboxes_scores[:, :4]
            scores = bboxes_scores[:, 4]
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        outs = [bboxes, scores, labels, masks]
    else:
        outs = [bboxes_scores, labels, masks]
    if return_sel:
        outs.append(sel)
    return tuple(outs)

def vis_det(img, bboxes, labels, class_names,
    masks=None, scores=None, score_th=0,
    out_scale=1, out_file=None):
    # img with RGB channel order
    # bboxes in the form of n*[left, top, right, bottom]
    # adapted from mmdet's visualization code

    if out_scale != 1:
        img = mmcv.imrescale(img, out_scale, interpolation='bilinear')

    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)
    if masks is not None:
        masks = np.asarray(masks)

    empty = len(bboxes) == 0
    if not empty and scores is not None and score_th > 0:
        sel = scores >= score_th
        bboxes = bboxes[sel]
        labels = labels[sel]
        scores = scores[sel]
        if masks is not None:
            masks = masks[sel]
        empty = len(bboxes) == 0

    if empty:
        if out_file is not None:
            imwrite(img, out_file)
        return img

    if out_scale != 1:
        bboxes = out_scale*bboxes
        # we don't want in-place operations like bboxes *= out_scale

    if masks is not None:
        img = np.array(img) # make it writable
        for mask in masks:
            color = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8
            )
            m = maskUtils.decode(mask)
            if out_scale != 1:
                m = mmcv.imrescale(
                    m.astype(np.uint8), out_scale,
                    interpolation='nearest'
                )
            m = m.astype(np.bool)
            img[m] = 0.5*img[m] + 0.5*color

    bbox_color = (0, 255, 0)
    text_color = (0, 255, 0)
    thickness = 1
    font_scale = 0.5

    bboxes = bboxes.round().astype(np.int32)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        lt = (bbox[0], bbox[1])
        rb = (bbox[2], bbox[3])
        cv2.rectangle(
            img, lt, rb, bbox_color, thickness=thickness
        )
        if class_names is None:
            label_text = f'class {label}'
        else:
            label_text = class_names[label]
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        cv2.putText(
            img, label_text, (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_COMPLEX, font_scale,
            text_color,
        )

    if out_file is not None:
        imwrite(img, out_file)
    return img


def eval_ccf(db, results, img_ids=None, class_subset=None, iou_type='bbox'):
    # ccf means CoCo Format
    if isinstance(results, str):
        if results.endswith('.pkl'):
            results = pickle.load(open(results, 'rb'))
        else:
            results = json.load(open(results, 'r'))

    results = db.loadRes(results)
    cocoEval = COCOeval(db, results, iou_type)
    if img_ids is not None:
        cocoEval.params.imgIds = img_ids
    if class_subset is not None:
        cocoEval.params.catIds = class_subset
        
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return {
        'eval': cocoEval.eval,
        'stats': cocoEval.stats,
    }