import os
import copy
import time
# import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
# from pathos.multiprocessing import ProcessingPool as Pool
import cv2
import numpy as np

class ObjectInfo():
    def __init__(self, box, conf, cat):
        self.box = box
        self.cat = cat
        self.conf = conf
    
    def get_box_as_tuple(self):
        return tuple(self.box)
    
    def update_confidence(self, decay):
        self.conf = self.conf * decay

    def get_cat_as_string(self):
        return str(self.cat)

    def xy2wh(self, b):
        return [ b[0], b[1], b[2] - b[0], b[3] - b[1] ]

    def get_object_as_result(self):
        # print(self.box, self.conf)
        return self.cat, np.array(self.box + [self.conf])

    def get_object_as_coco_result(self):
        return {
            "category_id" : self.cat,
            "bbox" : self.xy2wh(self.box),
            "score" : self.conf,
        }

def tracker_update_pool_func(tupl):
    tracker, frame = tupl
    try:
        ok, bbox = tracker.update(frame)
        if bbox[2] > 0  and bbox[3] > 0:
            return (ok, bbox)
        else:
            return (None, None)
    except Exception as e:
        return (None, None)

class OpencvMultiTracker():
    def __init__(
        self,
        tracker_name,
        num_processes,
    ):
        self.tracker_name = tracker_name
        self.curr_scale = None
        self.box_objects = []
        self.tracker_objects = []
        self.num_processes = num_processes
        self.pool = ThreadPool(self.num_processes)
        self.min_confidence = 0.5
        self.decay = 0.98

    def preprocess_image(self, img):
        pass
        # return self.preprocessor(img, self.device)

    def change_scale(self, scale):
        self.curr_scale = scale

    def init_tracker(self, curr_frame, bbox):
        is_okay = True
        if self.tracker_name == "opencv_kcf":
            tracker_obj = cv2.TrackerKCF_create()
        elif self.tracker_name == "opencv_csrt":
            tracker_obj = cv2.TrackerCSRT_create()
        try:
            tracker_obj.init(curr_frame, bbox)
        except Exception as e:
            is_okay = False
            tracker_obj = None
        return is_okay, tracker_obj

    def update_trackers(self, curr_frame):
        # success, boxes = self.multi_tracker.update(curr_frame)
        # for i, box in enumerate(boxes):
        #     self.box_objects[i].box = list(box)
        #     self.box_objects[i] *= self.decay        
        to_del = []
        vals = self.pool.map(tracker_update_pool_func, [ (x, curr_frame.copy()) for x in self.tracker_objects ])
        for i, out in enumerate(vals):
            ok, box = out
            if not ok:
                to_del.append(i)
            elif self.box_objects[i].conf < self.min_confidence:
                to_del.append(i)
            else:
                self.box_objects[i].box = list(box)
                self.box_objects[i].conf *= self.decay
        self.delete_trackers(to_del)

    def delete_trackers(self, to_del):
        for d in to_del[::-1]:
            del self.tracker_objects[d]
            del self.box_objects[d]

    def reset(self):
        self.box_objects = []
        self.tracker_objects = []

    def _result2boxes(self, result):
        dets = []
        for i in range(self.num_classes):
            cat = i
            for j in range(result[i].shape[0]):
                x1, y1, x2, y2, c = result[i][j, :]
                dets.append(ObjectInfo([x1, y1, x2, y2], c, cat))
        return dets

    def init(self, image, result):
        self.reset()
        self.num_classes = len(result)
        box_objs = self._result2boxes(result)
        for box_obj in box_objs:
            is_okay, tracker_obj = self.init_tracker(image, box_obj.get_box_as_tuple())
            if is_okay:
                self.box_objects.append(box_obj)
                self.tracker_objects.append(tracker_obj)

    def dispatch_result(self):
        dets = [ np.zeros((0,5)) for i in range(self.num_classes) ]
        for box in self.box_objects:
            cat, bbox_w_conf = box.get_object_as_result()
            dets[cat] = np.vstack([dets[cat], bbox_w_conf])
        return dets
        
    def forward(self, image):
        self.update_trackers(image)
        return self.dispatch_result()

    def track(self, image, prev_result=None):
        return self.forward(image)


if __name__ == "__main__":
    import itertools
    import random

    from data_loader import CustomCocoDetection, CustomCocoResult
    from detector_with_regressors import DetectorWithRegressors

    from setup_info import *

    model_name = "faster_rcnn"
    dataset = "imagenet_vid_argoverse_format"
    device = "cuda:0"
    # tracker_name = "tracktor_faster_rcnn"
    tracker_name = "opencv_csrt"

    data_loader = CustomCocoDetection(
        dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
    )
    det = DetectorWithRegressors(
        model_name,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        device,
        "resnet_50",
        None,  # models_info[dataset][model_name]["regressors"],
    )
    if tracker_name.split("_")[0] == "tracktor":
        tracker = Tracktor(
            tracker_name,
            tracker_info[dataset][tracker_name]["config_file"],
            tracker_info[dataset][tracker_name]["checkpoint_file"],
            device,
            # regressor_config=tracker_info[dataset][tracker_name]["regressors"]
        )
    elif tracker_name.split("_")[0] == "opencv":
        tracker = OpencvMultiTracker(
            tracker_name, 8
        )
    else:
        raise ("Invalid Tracker!!!")

    det.change_num_proposals(100)
    det.change_scale((2000, 600))
    tracker.change_scale((2000, 480))

    # seqs = data_loader.get_sequences_list() # [ sid ]
    # random.shuffle(seqs)
    # seqs = seqs[:1]
    seqs = [99001]

    # for i in range(2):
    #     image, bboxes, classes = data_loader.__getitem__(2)
    #     data = det.preprocess_image(image)
    #     result = det(data)
    #     result = tracker.track(image, result)

    proposal_vals = [100]  # [ 100, 300, 500, 1000 ]
    det_scale_vals = [(2000, 600), (2000, 480), (2000, 360), (2000, 240)]
    tracker_scale_vals = [(2000, 600), (2000, 480), (2000, 360)]
    time_strides = [5, 15]
    all_combs = itertools.product(
        proposal_vals, det_scale_vals, tracker_scale_vals, time_strides
    )

    def filter_func(x):
        if x[1] == "ada" or x[2] == "ada":
            return True
        return x[1][1] >= x[2][1]

    all_combs = filter(lambda x: filter_func(x), all_combs)

    for pr, det_scale, tracker_scale, time_stride in all_combs:

        det.change_num_proposals(pr)
        if det_scale is not "ada":
            det.change_scale(det_scale)
        if tracker_scale is not "ada":
            tracker.change_scale(tracker_scale)

        times = []
        results_obj = CustomCocoResult(dataset, dataset_info[dataset]["test_json"])

        for i, seq in enumerate(seqs):
            frames_info = data_loader.get_sequence(seq)
            seq_times = []
            prev_result = None
            for j, frame_info in enumerate(frames_info):
                image, bboxes, classes = data_loader.get_frame(frame_info["id"])
                ts = time.time()
                if j == 0 or j % time_stride == 0:
                    result = det.detect(image)
                    tracker.reset()
                    tracker.init(image, result)
                else:
                    # try:
                    result = tracker.track(image, result)
                    # except:
                    #     result = prev_result  # copy prev result in worst case if tracker fails
                seq_times.append((time.time() - ts) * 1000)
                results_obj.add_mmdet_results(frame_info, result)
                prev_result = result
            times.append(np.mean(seq_times))
        print(pr, det_scale, tracker_scale, time_stride, np.mean(times))
        results_obj.evaluate()
