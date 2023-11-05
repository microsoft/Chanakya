import os, sys
import argparse, json, pickle
import copy
import random
import time as time_module

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter, time
import multiprocessing as mp
import traceback

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

# # the line below is for running in both the current directory 
# # and the repo's root directory
# import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')

from utils import imread, imwrite, mkdir2, print_stats, parse_det_result, dist_from_dict, eval_ccf
from utils.forecast import ltrb2ltwh, ltrb2ltwh_, ltwh2ltrb, ltwh2ltrb_, iou_assoc, extrap_clean_up, \
    bbox2z, bbox2x, x2bbox, make_F, make_Q, \
    batch_kf_predict_only, batch_kf_predict, \
    batch_kf_update

from chanakya.detector_with_regressors import DetectorWithRegressors
# from chanakya.dynamic_tracktor import Tracktor
from chanakya.setup_info import *

from chanakya.RL_gpu import MultiOutputMAB
from chanakya.rl_configs import *

from io import StringIO 

np.set_printoptions(precision=3)
np.seterr(all="ignore")

class StdOutCapturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1') # frame

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)

    # parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    
    parser.add_argument('--dynamic-schedule', action='store_true', default=False)
    parser.add_argument('--dynamic-schedule-type',  type=str, required=True)
    parser.add_argument('--perf-factor', type=float, default=1)

    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--forecast-rt-ub', type=float, default=0.003) # seconds

    parser.add_argument('--rl-config', type=str, required=True)
    parser.add_argument('--rl-model-load-folder', type=str)
    parser.add_argument('--rl-model-prefix', type=str)

    parser.add_argument('--fixed-advantage-reward', action='store_true', default=False)
    parser.add_argument('--fixed-policy-results-folder', type=str)
    parser.add_argument('--fixed-advantage-scale-factor', type=float, default=5)

    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def det_process(opts, frame_recv, det_res_send, w_img, h_img):
    try:

        
        model_name = opts.model_name
        dataset = opts.dataset
        model = DetectorWithRegressors(
                model_name,
                dataset,
                models_info[dataset][model_name]["config_file"],
                models_info[dataset][model_name]["checkpoint_file"],
                opts.device,
                models_info[dataset][model_name]["regressors"]
        )

        prop_choices = mab_configs[opts.rl_config]["actions"][0][1]
        det_scales_choices =  mab_configs[opts.rl_config]["actions"][1][1]
        init_action = mab_configs[opts.rl_config]["rl"]["init_action"]
        rl_model = MultiOutputMAB(mab_configs[opts.rl_config], device=opts.device)

        model.change_num_proposals(prop_choices[init_action[0]])
        model.change_scale(det_scales_choices[init_action[1]])

        # warm up the GPU
        _ = model.detect(np.zeros((h_img, w_img, 3), np.uint8))
        torch.cuda.synchronize()

        sample_for_rl_prob = 0.06
        while 1:
            fidx = frame_recv.recv()
            if type(fidx) is list:
                # new video, read all images in advance 
                frame_list = fidx
                frames = [imread(img_path) for img_path in frame_list]
                # signal ready, no errors
                det_res_send.send('ready')
                continue
            elif type(fidx) is tuple:
                if len(fidx) == 2:
                    fidx, t1 = fidx
                elif len(fidx) == 3:
                    if fidx[0] == "load_model":
                        _, out_path, rl_model_prefix = fidx
                        rl_model.load(out_path, rl_model_prefix)
                        model.change_num_proposals(prop_choices[init_action[0]])
                        model.change_scale(det_scales_choices[init_action[1]])
                        det_res_send.send("loaded_model")
                        continue
                    elif fidx[0] == "save_model":
                        _, out_path, rl_model_prefix = fidx
                        rl_model.save(out_path, rl_model_prefix)
                        det_res_send.send("saved_model")
                        continue
            elif fidx is None:
                # exit flag
                break
            img = frames[fidx]
            t2 = perf_counter() 
            t_send_frame = t2 - t1

            sample_for_rl = np.random.binomial(1, sample_for_rl_prob)

            if sample_for_rl:
                result, metrics = model.detect(img, get_metrics=True, get_switch_metric=True, get_area_metric=True)
                if np.isnan(metrics).any() or np.isinf(metrics).any():
                    sample_for_rl = 0
                else:
                    actions = rl_model.act(metrics)
                    print(prop_choices[actions[0]], det_scales_choices[actions[1]])
                    model.change_num_proposals(prop_choices[actions[0]])
                    model.change_scale(det_scales_choices[actions[1]])
            else:
                result = model.detect(img) #, get_metrics=True)
            
            parsed_result = model.parse_result_for_sap(result)
            torch.cuda.synchronize()

            t3 = perf_counter()
            if sample_for_rl:
                det_res_send.send([parsed_result, t_send_frame, t3, t3-t2, metrics, actions])
            else:
                det_res_send.send([parsed_result, t_send_frame, t3, t3-t2])
    
    except Exception:
        # report all errors from the child process to the parent
        # forward traceback info as well
        print("".join(traceback.format_exception(*sys.exc_info())))
        det_res_send.send(Exception("".join(traceback.format_exception(*sys.exc_info()))))


def sAP(opts, db, frame_list, results_parsed, input_fidx, timestamps, curr_frame_list_idx):
    results_ccf = []
    in_time = 0
    miss = 0
    mismatch = 0    
    tidx_p1 = 0
    for ii, img in enumerate(frame_list):
        ## adjust frame idx
        ii = ii + curr_frame_list_idx

        # pred, gt association by time
        t = (ii - opts.eta)/opts.fps
        while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t:
            tidx_p1 += 1
        if tidx_p1 == 0:
            # no output
            miss += 1
            bboxes, scores, labels  = [], [], []
        else:
            tidx = tidx_p1 - 1
            ifidx = input_fidx[tidx]
            in_time += int(ii == ifidx)
            mismatch += ii - ifidx

            result = results_parsed[tidx]
            bboxes, scores, labels, masks = result[:4]
                
        # convert to coco fmt
        n = len(bboxes)
        if n:
            bboxes_ltwh = ltrb2ltwh(bboxes)

        for i in range(n):
            result_dict = {
                'image_id': img['id'],
                'bbox': bboxes_ltwh[i],
                'score': scores[i],
                'category_id': labels[i],
            }
            if masks is not None:
                result_dict['segmentation'] = masks[i]

            results_ccf.append(result_dict)
    return eval_ccf(db, results_ccf, img_ids=[x['id'] for x in frame_list])

def compute_reward(opts, sid, coco_gt_obj, frame_list, results, fidx, timestamps, curr_frame_list_idx):
    try:
        with StdOutCapturing() as output:
            eval_summary = sAP(opts, coco_gt_obj, frame_list, results, fidx, timestamps, curr_frame_list_idx)
        vals = eval_summary['stats']
        return vals[0]
    except:
        return 0.0

def get_first_time_stamp_idx_greater_than(timestamps, curr_timestamp):
    for i, timestamp in enumerate(timestamps):
        if timestamp > curr_timestamp:
            break
    return i

def transform_reward_minmax(sap_list):
    minimum_sap = min(sap_list)
    maximum_sap = max(sap_list)
    reward_list = []
    for idx in range(len(sap_list)):
        reward = (sap_list[idx] - minimum_sap) / (maximum_sap - minimum_sap)
        reward_list.append(reward)
    return reward_list

def train_rl(opts, sid, seq, rl_model, rl_replay, results_parsed, input_fidx, timestamps, t_total, n_seq, fixed_advantage=False, plot_rewards=False):
    with StdOutCapturing() as output:
        coco_gt_obj = COCO(opts.annot_path)

    if fixed_advantage:
        fixed_advantage_factor = opts.fixed_advantage_scale_factor
        with open(os.path.join(opts.fixed_policy_results_folder, seq + ".pkl"), "rb") as f:
            x = pickle.load(f)
            results_parsed_fixed, timestamps_fixed =  x["results_parsed"], x["timestamps"]

    update_stride = 10
    all_frames_list = [im_info for im_info in coco_gt_obj.imgs.values() if im_info['sid'] == sid]
    sap_list = []
    for idx in range(len(rl_replay)):

        metrics, actions, curr_timestamp = rl_replay[idx]
        if idx != len(rl_replay) - 1:
            _, __, next_timestamp = rl_replay[idx + 1]
        else:
            next_timestamp = t_total

        curr_frame_list_idx = int(np.round(curr_timestamp*opts.fps))
        next_frame_list_idx = int(np.round(next_timestamp*opts.fps))

        curr_timestamp_idx = get_first_time_stamp_idx_greater_than(timestamps, curr_timestamp)
        next_timestamp_idx = get_first_time_stamp_idx_greater_than(timestamps, next_timestamp) - 1

        frame_list_till_now = all_frames_list[curr_frame_list_idx:next_frame_list_idx]
        results_till_now = results_parsed[curr_timestamp_idx:next_timestamp_idx]
        fidx_till_now = input_fidx[curr_timestamp_idx:next_timestamp_idx]
        timestamps_till_now = timestamps[curr_timestamp_idx:next_timestamp_idx]

        reward_current = compute_reward(opts, sid, coco_gt_obj, frame_list_till_now, results_till_now, fidx_till_now, timestamps_till_now, curr_frame_list_idx)

        if fixed_advantage:
            curr_ts_idx_fixed = get_first_time_stamp_idx_greater_than(timestamps_fixed, curr_timestamp)
            next_ts_idx_fixed = get_first_time_stamp_idx_greater_than(timestamps_fixed, next_timestamp) - 1

            results_tn_fixed = results_parsed_fixed[curr_ts_idx_fixed:next_ts_idx_fixed]
            fidx_tn_fixed = input_fidx[curr_ts_idx_fixed:next_ts_idx_fixed]
            timestamps_tn_fixed = timestamps_fixed[curr_ts_idx_fixed:next_ts_idx_fixed]

            reward_fixed = compute_reward(opts, sid, coco_gt_obj, frame_list_till_now, results_tn_fixed, fidx_tn_fixed, timestamps_tn_fixed, curr_frame_list_idx)


        if fixed_advantage:
            reward = fixed_advantage_factor*(reward_current - reward_fixed)         
        else:
            reward = reward_current

        sap_list.append(reward)

    # reward_list = transform_reward_minmax(sap_list) if minmax_reward else sap_list
    reward_list = sap_list

    avg_rewards = 0
    for idx in range(len(rl_replay)):
        reward = reward_list[idx]
        avg_rewards += reward
        metrics, actions, curr_timestamp = rl_replay[idx]
        if not np.isnan(metrics).any() and not np.isinf(metrics).any() and not np.isnan(reward).any() and not np.isinf(reward).any():
            rl_model.remember(metrics, actions, reward)
        if idx > 0 and idx % update_stride == 0:
            rl_model.update()
    rl_model.update()
    avg_rewards /= len(rl_replay)
    rl_model.save("./models", opts.rl_config + "_latest")
    if plot_rewards:
        rl_model.plot_rewards('plots', '{}_{}'.format(opts.rl_config,  opts.dataset), n_seq)
    return rl_model, avg_rewards

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()
    mkdir2(opts.out_dir)

    with StdOutCapturing() as output:
        db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    # print(seqs)
    # print(seq_dirs)
    if opts.dataset == "imagenet_vid_argoverse_format":
        with open("chanakya/imagenet_vid_seqs.json") as f:
            seqs_subsets = json.load(f)
        seqs = seqs_subsets["seqs_500_1000"]

    # img = db.imgs[0]
    w_img, h_img = 1280, 720 #img['width'], img['height']

    mp.set_start_method('spawn')
    frame_recv, frame_send = mp.Pipe(False)
    det_res_recv, det_res_send = mp.Pipe(False)
    det_proc = mp.Process(target=det_process, args=(opts, frame_recv, det_res_send, w_img, h_img))
    det_proc.start()

    rl_model = MultiOutputMAB(mab_configs[opts.rl_config], device=opts.device)
    output_models_folder = "./models/{}_{}/".format(opts.rl_config, opts.dataset)
    if not os.path.exists(output_models_folder):
        os.makedirs(output_models_folder)
    rl_model_temp_prefix = "{}_{}_temp".format(opts.rl_config, opts.dataset)

    model_load_from_folder = opts.rl_model_load_folder
    model_load_from_prefix = opts.rl_model_prefix
    if model_load_from_prefix is not None:
        frame_send.send(("load_model", model_load_from_folder, model_load_from_prefix))
        while 1:
            msg = det_res_recv.recv() # wait till the detector is ready
            if msg == 'loaded_model':
                break
            elif isinstance(msg, Exception):
                raise msg
        rl_model.load(model_load_from_folder, model_load_from_prefix)
 
    prop_choices = mab_configs[opts.rl_config]["actions"][0][1]
    det_scales_choices =  mab_configs[opts.rl_config]["actions"][1][1]
    init_action = mab_configs[opts.rl_config]["rl"]["init_action"]

    if opts.dynamic_schedule and opts.dynamic_schedule_type != "mean":
        with open("/FatigueDataDrive/HAMS-Edge/sim/rtfs/{}_{}.pkl".format(opts.model_name, opts.dataset), "rb") as f:
            mean_rtfs = pickle.load(f)
        for x in mean_rtfs.keys():
            mean_rtfs[x] = np.mean(mean_rtfs[x])
    if opts.dynamic_schedule:
        if opts.dynamic_schedule_type == "mean":
            mean_rtf = 0.0
        else:
            mean_rtf = mean_rtfs[ prop_choices[init_action[0]], det_scales_choices[init_action[1]][1] ]*opts.fps*opts.perf_factor

    n_total = 0
    t_det_all = []
    t_send_frame_all = []
    t_recv_res_all = []
    t_assoc_all = []
    t_forecast_all = []

    rl_epochs = 10

    if opts.dataset != "imagenet_vid_argoverse_format":
        seqs = list(enumerate(seqs))

    kf_F = torch.eye(8)
    kf_Q = torch.eye(8)
    kf_R = 10*torch.eye(4)
    kf_P_init = 100*torch.eye(8).unsqueeze(0)

    for epoch in range(rl_epochs):
        epoch_reward = 0.0
        random.shuffle(seqs)
        n_seq = 0
        for sid, seq in seqs: 
            frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
            w_img, h_img = frame_list[0]['width'], frame_list[0]['height']
            frame_list = [join(opts.data_root, seq_dirs[sid], img['name']) for img in frame_list]
            n_frame = len(frame_list)
            n_total += n_frame
            

            timestamps = []
            results_parsed = []
            input_fidx = []

            new_rl_sample_bool = False

            processing = False
            fidx_t2 = None            # detection input index at t2
            fidx_latest = None
            tkidx = 0                 # track starting index
            kf_x = torch.empty((0, 8, 1))
            kf_P = torch.empty((0, 8, 8))
            n_matched12 = 0

            # let detector process to read all the frames
            frame_send.send(frame_list)
            # it is possible that unfetched results remain in the pipe
            while 1:
                msg = det_res_recv.recv() # wait till the detector is ready
                if msg == 'ready':
                    break
                elif isinstance(msg, Exception):
                    raise msg

            t_total = n_frame/opts.fps
            t_unit = 1/opts.fps
            t_start = perf_counter()
            count_detections = 0
            
            rl_replay = []

            while 1:
                with torch.no_grad():
                    t1 = perf_counter()
                    t_elapsed = t1 - t_start
                    if t_elapsed >= t_total:
                        break

                    # identify latest available frame
                    fidx_continous = t_elapsed*opts.fps
                    fidx = int(np.floor(fidx_continous))
                    if fidx == fidx_latest:
                        # algorithm is fast and has some idle time
                        wait_for_next = True
                    else:
                        wait_for_next = False
                        if opts.dynamic_schedule:
                            if mean_rtf >= 1:
                                # when runtime < 1, it should always process every frame
                                fidx_remainder = fidx_continous - fidx
                                if mean_rtf < np.floor(fidx_remainder + mean_rtf):
                                    # wait till next frame
                                    wait_for_next = True

                    if wait_for_next:
                        # sleep
                        continue

                    if not processing:
                        t_start_frame = perf_counter()
                        frame_send.send((fidx, t_start_frame))
                        fidx_latest = fidx
                        processing = True

                    # wait till query - forecast-rt-ub
                    wait_time = t_unit - opts.forecast_rt_ub

                    if det_res_recv.poll(wait_time): # wait
                        # new result
                        result = det_res_recv.recv() 
                        if isinstance(result, Exception):
                            raise result
                        if len(result) == 4:
                            result, t_send_frame, t_start_res, t_det = result
                        else:
                            result, t_send_frame, t_start_res, t_det, metrics, actions = result
                            new_rl_sample_bool = True


                        if opts.dynamic_schedule and new_rl_sample_bool:
                            if opts.dynamic_schedule_type == "mean":
                                sum_rtf = mean_rtf*count_detections + t_det*opts.fps*opts.perf_factor
                                count_detections += 1
                                mean_rtf = sum_rtf/count_detections
                            else:
                                mean_rtf = mean_rtfs[ prop_choices[actions[0]], det_scales_choices[actions[1]][1] ]*opts.fps*opts.perf_factor

                        bboxes_t2, scores_t2, labels_t2, _ = \
                            parse_det_result(result, coco_mapping, n_class)
                        processing = False
                        t_det_end = perf_counter()
                        t_det_all.append(t_det_end - t_start_frame)
                        t_send_frame_all.append(t_send_frame)
                        t_recv_res_all.append(t_det_end - t_start_res)

                        if new_rl_sample_bool:
                            t_rl_sample = t_det_end - t_start                            
                            rl_replay.append((metrics, actions, t_rl_sample))
                            new_rl_sample_bool = False

                        # associate across frames
                        t_assoc_start = perf_counter()
                        if len(kf_x):
                            dt = fidx_latest - fidx_t2

                            kf_F = make_F(kf_F, dt)
                            kf_Q = make_Q(kf_Q, dt)
                            kf_x, kf_P = batch_kf_predict(kf_F, kf_x, kf_P, kf_Q)
                            bboxes_f = x2bbox(kf_x)
                                            
                        fidx_t2 = fidx_latest

                        n = len(bboxes_t2)
                        if n:
                            # put high scores det first for better iou matching
                            score_argsort = np.argsort(scores_t2)[::-1]
                            bboxes_t2 = bboxes_t2[score_argsort]
                            scores_t2 = scores_t2[score_argsort]
                            labels_t2 = labels_t2[score_argsort]

                            ltrb2ltwh_(bboxes_t2)

                        updated = False
                        if len(kf_x):
                            order1, order2, n_matched12, tracks, tkidx = iou_assoc(
                                bboxes_f, labels, tracks, tkidx,
                                bboxes_t2, labels_t2, opts.match_iou_th,
                                no_unmatched1=True,
                            )

                            if n_matched12:
                                kf_x = kf_x[order1]
                                kf_P = kf_P[order1]
                                kf_x, kf_P = batch_kf_update(
                                    bbox2z(bboxes_t2[order2[:n_matched12]]),
                                    kf_x,
                                    kf_P,
                                    kf_R,
                                )
                        
                                kf_x_new = bbox2x(bboxes_t2[order2[n_matched12:]])
                                n_unmatched2 = len(bboxes_t2) - n_matched12
                                kf_P_new = kf_P_init.expand(n_unmatched2, -1, -1)
                                kf_x = torch.cat((kf_x, kf_x_new))
                                kf_P = torch.cat((kf_P, kf_P_new))
                                labels = labels_t2[order2]
                                scores = scores_t2[order2]
                                updated = True

                        if not updated:
                            # start from scratch
                            kf_x = bbox2x(bboxes_t2)
                            kf_P = kf_P_init.expand(len(bboxes_t2), -1, -1)
                            labels = labels_t2
                            scores = scores_t2
                            tracks = np.arange(tkidx, tkidx + n, dtype=np.uint32)
                            tkidx += n

                        t_assoc_end = perf_counter()
                        t_assoc_all.append(t_assoc_end - t_assoc_start)

                    # apply forecasting for the current query
                    t_forecast_start = perf_counter()
                    query_pointer = fidx + opts.eta + 1
                    
                    if len(kf_x):
                        dt = (query_pointer - fidx_t2)

                        kf_x_np = kf_x[:, :, 0].numpy()
                        bboxes_t3 = kf_x_np[:n_matched12, :4] + dt*kf_x_np[:n_matched12, 4:]
                        if n_matched12 < len(kf_x):
                            bboxes_t3 = np.concatenate((bboxes_t3, kf_x_np[n_matched12:, :4]))
                            
                        bboxes_t3, keep = extrap_clean_up(bboxes_t3, w_img, h_img, lt=True)
                        labels_t3 = labels[keep]
                        scores_t3 = scores[keep]
                        tracks_t3 = tracks[keep]

                    else:
                        bboxes_t3 = np.empty((0, 4), dtype=np.float32)
                        scores_t3 = np.empty((0,), dtype=np.float32)
                        labels_t3 = np.empty((0,), dtype=np.int32)
                        tracks_t3 = np.empty((0,), dtype=np.int32)

                    t_forecast_end = perf_counter()
                    t_forecast_all.append(t_forecast_end - t_forecast_start)
                    
                    t3 = perf_counter()
                    t_elapsed = t3 - t_start
                    if t_elapsed >= t_total:
                        break

                    if len(bboxes_t3):
                        ltwh2ltrb_(bboxes_t3)
                    if fidx_t2 is not None:
                        timestamps.append(t_elapsed)
                        results_parsed.append((bboxes_t3, scores_t3, labels_t3, None, tracks_t3))
                        input_fidx.append(fidx_t2) 


            n_seq += 1
            frame_send.send(("save_model", output_models_folder, rl_model_temp_prefix))
            while 1:
                msg = det_res_recv.recv() # wait till the detector is ready
                if msg == 'saved_model':
                    break
                elif isinstance(msg, Exception):
                    raise msg
            rl_model.load(output_models_folder, rl_model_temp_prefix)
            
            ## replay the whole thing and train the rl model
            avg_seq_reward = 0.0
            prev_rl_model = copy.deepcopy(rl_model)
            rl_model, avg_seq_reward = train_rl(opts, sid, seq, rl_model, rl_replay, results_parsed, input_fidx, timestamps, t_total, n_seq, fixed_advantage=opts.fixed_advantage_reward)

            print("n_seq", n_seq, "sid", sid, "seq_reward", avg_seq_reward)
            epoch_reward += avg_seq_reward
            rl_model.save(output_models_folder, rl_model_temp_prefix)
            if n_seq % 10 == 0:
                rl_model.plot(".",
                        prefixes = [ "{}_{}_prop_latest".format(opts.rl_config,  opts.dataset), "{}_{}_scale_latest".format(opts.rl_config,  opts.dataset) ], 
                        action_sets = [ 
                                [ str(x) for x in mab_configs[opts.rl_config]["actions"][0][1] ], 
                                [ str(x[-1]) for x in mab_configs[opts.rl_config]["actions"][1][1] ]
                            ]
                        )
                # rl_model.features_importance(".")
            if n_seq % 50 == 0:
                rl_model.save(output_models_folder, "epoch_{}_n_seq_{}".format(epoch, n_seq))


            frame_send.send(("load_model", output_models_folder, rl_model_temp_prefix))
            while 1:
                msg = det_res_recv.recv() # wait till the detector is ready
                if msg == 'loaded_model':
                    break
                elif isinstance(msg, Exception):
                    raise msg

            if opts.dynamic_schedule:
                if opts.dynamic_schedule_type == "mean":
                    pass
                else:
                    mean_rtf = mean_rtfs[ prop_choices[init_action[0]], det_scales_choices[init_action[1]][1] ]*opts.fps*opts.perf_factor

            out_path = join(opts.out_dir, seq + '.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump({
                    'results_parsed': results_parsed,
                    'timestamps': timestamps,
                    'input_fidx': input_fidx,
                }, open(out_path, 'wb'))
        
        rl_model.save(output_models_folder, "epoch_{}".format(epoch))
        rl_model.plot(output_models_folder,
                prefixes = [ "{}_{}_prop_epoch_{}".format(opts.rl_config,  opts.dataset, epoch), "{}_{}_scale_epoch_{}".format(opts.rl_config,  opts.dataset, epoch) ], 
                action_sets = [ 
                        [ str(x) for x in mab_configs[opts.rl_config]["actions"][0][1] ], 
                        [ str(x[-1]) for x in mab_configs[opts.rl_config]["actions"][1][1] ]
                    ]
                )

        epoch_reward /= len(seqs)
        print("Epoch {}: Avg Reward = {}".format(epoch, epoch_reward))

    # terminates the child process
    frame_send.send(None)
    
    out_path = join(opts.out_dir, 'time_info.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'n_total': n_total,
            't_det': t_det_all,
            't_send_frame': t_send_frame_all,
            't_recv_res': t_recv_res_all,
            't_assoc': t_assoc_all,
            't_forecast': t_forecast_all,
        }, open(out_path, 'wb'))
 
    # convert to ms for display
    s2ms = lambda x: 1e3*x
    print_stats(t_det_all, 'Runtime detection (ms)', cvt=s2ms)
    print_stats(t_send_frame_all, 'Runtime sending the frame (ms)', cvt=s2ms)
    print_stats(t_recv_res_all, 'Runtime receiving the result (ms)', cvt=s2ms)
    print_stats(t_assoc_all, "Runtime association (ms)", cvt=s2ms)
    print_stats(t_forecast_all, "Runtime forecasting (ms)", cvt=s2ms)

if __name__ == '__main__':
    main()