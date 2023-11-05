import copy
import numpy as np

class AdaptiveCrop:
    def __init__(self, dataset):
        if dataset == "argoverse":
            self.patch_size = 30
            self.reasonable_window = 4
        elif dataset == "imagenet_vid_argoverse_format" or dataset == "imagenet_vid":
            self.patch_size = 30
            self.reasonable_window = 2

    def get_box_extents(self, video_actual_scale, result):
        w, h = video_actual_scale
        loc_spread = np.zeros((w//self.patch_size, h//self.patch_size)) 
        for i in range(len(result)):
            if result[i].shape[0] > 0:
                x1min = np.min((result[i][:, 0] // self.patch_size)).astype(int)
                y1min = np.min((result[i][:, 1] // self.patch_size + 1)).astype(int)
                x2max = np.max((result[i][:, 2] // self.patch_size)).astype(int)
                y2max = np.max((result[i][:, 3] // self.patch_size + 1)).astype(int)
                loc_spread[x1min:x2max, y1min:y2max] += 1
        return np.sum(loc_spread, axis=0), np.sum(loc_spread, axis=1)

    def get_crop_extents(self, video_actual_scale, prev_result, normalized=False):
        w, h = video_actual_scale
        top_bottom, left_right = self.get_box_extents(video_actual_scale, prev_result)

        patch_size = self.patch_size
        top = 0
        for i in range(0, top_bottom.shape[0] // 2):
            flag = 0
            for j in range(self.reasonable_window):
                if top_bottom[i + j] != 0:
                    flag = 1
                    break
            if flag == 1:
                break
            else:
                top = i

        bottom = top_bottom.shape[0]
        for i in range(top_bottom.shape[0] - 1, top_bottom.shape[0]//2, -1):
            flag = 0
            for j in range(self.reasonable_window):
                if top_bottom[i - j] != 0:
                    flag = 1
                    break
            if flag == 1:
                break
            else:
                bottom = i

        left = 0
        for i in range(0, left_right.shape[0] // 2):
            flag = 0
            for j in range(self.reasonable_window):
                if left_right[i + j] != 0:
                    flag = 1
                    break
            if flag == 1:
                break
            else:
                left = i

        right = left_right.shape[0]
        for i in range(left_right.shape[0] - 1, left_right.shape[0]//2, -1):
            flag = 0
            for j in range(self.reasonable_window):
                if left_right[i - j] != 0:
                    flag = 1
                    break
            if flag == 1:
                break
            else:
                right = i
        if normalized:
            return [ (top *patch_size)/h, (bottom * patch_size)/h, (left * patch_size)/w, (right * patch_size)/w ]
        return [top * patch_size, bottom * patch_size, left * patch_size, right * patch_size ] # 1900, 1200

    def result_to_cropped_result(self, crop_extents, result):
        """
            result = [ bboxes of each class ]
            top, bottom, left, right = crop_extents
            Coordinates to be shifted from (0, 0) to (top, left),
            change bboxes accordingly
        """
        cropped_result = copy.deepcopy(result)
        for i in range(len(cropped_result)):
            cropped_result[i][:, 0] -= crop_extents[2]
            cropped_result[i][:, 1] -= crop_extents[0]
            cropped_result[i][:, 2] -= crop_extents[2]
            cropped_result[i][:, 3] -= crop_extents[0]
        return cropped_result

    def cropped_result_to_result(self, crop_extents, cropped_result):
        """
            result = [ bboxes of each class ]
            top, bottom, left, right = crop_extents
            Coordinates to be shifted from (top, left) to (0, 0),
            change bboxes accordingly
        """
        result = copy.deepcopy(cropped_result)
        for i in range(len(cropped_result)):
            result[i][:, 0] += crop_extents[2]
            result[i][:, 1] += crop_extents[0]
            result[i][:, 2] += crop_extents[2]
            result[i][:, 3] += crop_extents[0]
        return result

class BboxMetrics:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == "argoverse":
            self.patch_size = 30
            self.reasonable_window = 4
            self.argoverse_idx = [0, 1, 2, 3, 5, 7, 9, 11]
            self.argoverse_class_mean = np.array([ 1.516, 0.9,    4.426, 0.9,    0.094, 1.426, 0.359, 0.9   ])
            self.argoverse_class_std = np.array([ 2.144, 0.9,    1.521, 0.9,    0.292, 0.495, 0.974, 0.9   ])
            self.argoverse_size_mean = np.array([7.35285152, 7.83856217, 4.2782929 ])
            self.argoverse_size_std = np.array([4.88071581, 4.60258285, 2.68658503])
        if dataset == "imagenet_vid" or self.dataset == "imagenet_vid_argoverse_format":
            self.patch_size = 30
            self.reasonable_window = 2
            self.imagenet_vid_class_mean = np.array([0.11894643, 0.04106539, 0.02926775, 0.02969273, 0.03592875, 0.11945456, 0.034312 ,  0.05435044, 0.02969273, 0.01942869, 0.04781878, 0.03605809, 0.03557769, 0.04319026, 0.04984202, 0.07102603, 0.04878883, 0.03300937, 0.05489551, 0.0653628 , 0.07766856, 0.04418802, 0.04870568, 0.07910977, 0.03198389, 0.02778034, 0.10530108, 0.03167902, 0.09715268, 0.05402709])
            self.imagenet_vid_class_std = np.array([0.65087579, 0.2074548,  0.18285767, 0.16973823, 0.24234693, 0.4501688, 0.22336838, 0.24814786, 0.25227055, 0.15735515, 0.28220719, 0.23331531, 0.28052442, 0.20888873, 0.35179635, 0.64549092, 0.46395297, 0.36195652, 0.43605763, 0.3784947 , 0.55046035, 0.26980844, 0.29761069, 0.5324383, 0.23675482, 0.20841006, 0.49275322, 0.24708669, 0.31054023, 0.34226506])
            self.imagenet_vid_size_mean = np.array([0.05079951, 0.34014128, 1.20373019])
            self.imagenet_vid_size_std = np.array([0.38954713, 1.00283785, 0.90128105])
        
    def get_box_extents(self, video_actual_scale, result):
        w, h = video_actual_scale
        loc_spread = np.zeros((w//self.patch_size, h//self.patch_size)) 
        for i in range(len(result)):
            if result[i].shape[0] > 0:
                x1min = np.min((result[i][:, 0] // self.patch_size)).astype(int)
                y1min = np.min((result[i][:, 1] // self.patch_size + 1)).astype(int)
                x2max = np.max((result[i][:, 2] // self.patch_size)).astype(int)
                y2max = np.max((result[i][:, 3] // self.patch_size + 1)).astype(int)
                loc_spread[x1min:x2max, y1min:y2max] += 1
        return np.sum(loc_spread, axis=0), np.sum(loc_spread, axis=1)


    def get_localization_spread(self, video_actual_scale, result):
        ## TODO: TOO SLOW!!!
        w, h = video_actual_scale
        loc_spread = np.zeros((w//self.patch_size, h//self.patch_size)) 
        for i in range(len(result)):
            for j in range(result[i].shape[0]):
                x1, y1, x2, y2, _ = result[i][j, :]
                start_x = int(x1 // self.patch_size)
                end_x = int(x2 // self.patch_size) + 1
                start_y = int(y1 // self.patch_size)
                end_y = int(y2 // self.patch_size) + 1
                loc_spread[start_x:end_x, start_y: end_y] += 1
        return loc_spread

    def get_class_info_metric(self, result, normalized=False):
        class_info = np.zeros((len(result), ))
        for i in range(len(result)):
            class_info[i] += result[i].shape[0]
        if self.dataset == "argoverse":
            class_info = np.array(class_info[self.argoverse_idx])
        if normalized:
            if self.dataset == "argoverse":
                class_info = (class_info - self.argoverse_class_mean)/self.argoverse_class_std
            elif self.dataset == "imagenet_vid" or self.dataset == "imagenet_vid_argoverse_format":
                class_info = (class_info - self.imagenet_vid_class_mean)/self.imagenet_vid_class_std
            else:
                raise Exception("Not supported!")
        return class_info

    def get_box_size_metric(self, result, normalized=False):
        res = result.copy()
        ## borrowed from coco: small, medium, large
        area_rngs = [ 0.0, 32 ** 2, 96 ** 2, 1e5 ** 2 ]
        ars = []
        for i in range(len(res)):
            x = res[i]
            ar = (x[:, 2] - x[:, 0])*(x[:, 3] - x[:, 1])
            ars.append(ar)
        ars = np.hstack(ars)
        hist = np.histogram(ars, area_rngs)[0]
        if normalized:
            if self.dataset == "argoverse":
                hist = (hist - self.argoverse_size_mean)/self.argoverse_size_std
            elif self.dataset == "imagenet_vid" or self.dataset == "imagenet_vid_argoverse_format":
                hist = (hist - self.imagenet_vid_size_mean)/self.imagenet_vid_size_std
            else:
                raise Exception("Not supported!")
        return hist

    def get_confidence_metric(self, result):
        confs = []
        for i in range(len(result)):
            confs.append(result[i][:, -1])
        confs = np.hstack(confs)
        return np.array([ np.average(confs), np.std(confs) ])


if __name__ == "__main__":
    import random
    import cv2

    from data_loader import CustomCocoDetection, CustomCocoResult
    from detector_with_regressors import DetectorWithRegressors

    from setup_info import *

    model_name = "faster_rcnn"
    dataset = "argoverse"
    # dataset = "imagenet_vid_argoverse_format"
    device = "cuda:0"

    metrics_calc = BboxMetrics(dataset)

    data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
    )
    det = DetectorWithRegressors(
        model_name,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        device,
        None,  # models_info[dataset][model_name]["regressors"],
    )

    # seqs = data_loader.get_sequences_list() # [ sid ]
    # random.shuffle(seqs)
    # print(seqs[:1])

    seqs = [2]

    for i, seq in enumerate(seqs):
        frames_info = data_loader.get_sequence(seq)
        seq_times = []
        prev_result = None
        for j, frame_info in enumerate(frames_info):
            if j < 10:
                continue
            print(frame_info)
            print(data_loader.coco_json["seq_dirs"][seq])
            image, bboxes, classes = data_loader.get_frame(frame_info["id"])
            result = det.detect(image)
            x = metrics_calc.get_localization_spread((frame_info["width"], frame_info["height"]), result)
            y = metrics_calc.get_class_info_metric(result)
            print(np.sum(y))
            crop_vals = metrics_calc.get_crop_extents(x) # top bottom left right
            print(crop_vals)
            det.detector.show_result(image, result, out_file="temp.jpg")
            im = cv2.imread("temp.jpg")
            cv2.line(im, (0, crop_vals[0]), (1920, crop_vals[0]), (255, 0, 0), thickness=2)
            cv2.line(im, (0, crop_vals[1]), (1920, crop_vals[1]), (0, 255, 0), thickness=2)
            cv2.line(im, (crop_vals[2], 0), (crop_vals[2], 1200), (255/2, 0, 0), thickness=2)
            cv2.line(im, (crop_vals[3], 0), (crop_vals[3], 1200), (0, 255/2, 0), thickness=2)
            cv2.imwrite("tmp.jpg", im)
            top, bottom, left, right = crop_vals
            im = im[top:bottom, left:right]
            cv2.imwrite("tmpcropped.jpg", im)
            image = image[top:bottom, left:right]
            cv2.imwrite("tmpcropped2.jpg", image)
            break
        break