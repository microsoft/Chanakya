import argparse, json, pickle

import os, sys
from os.path import join, isfile, basename
from glob import glob
from time import perf_counter
import multiprocessing as mp
import traceback
from numpy.core.fromnumeric import mean
from tqdm import tqdm

# setting environment variables - we have noticed that
# numpy creates extra threads which slows down computation,
# these environment variables prevent that
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"

import numpy as np
import torch
import os

from pycocotools.coco import COCO

# utility functions for mmdetection
from utils import print_stats

#############
### DONT UNCOMMENT THIS!!!! THEY OVERWRITE MMDETECTION FUNCTION CALLS
### AND THAT SCREWS UP OUR CODE!!!!
#############
## from utils.mmdet import init_detector, inference_detector, parse_det_result

# utility functions to forecast
from utils.forecast import ltrb2ltwh_, ltwh2ltrb_, iou_assoc, extrap_clean_up, \
    bbox2z, bbox2x, x2bbox, make_F, make_Q, \
    batch_kf_predict_only, batch_kf_predict, \
    batch_kf_update

# multiprocessing
import multiprocessing as mp

# benchmark toolkit API
from sap_toolkit.client import EvalClient

# chanakya
from chanakya.detector_with_regressors import DetectorWithRegressors

from chanakya.setup_info import *

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot-path', type=str, required=True)
    # 30 is the fps of the stream received from evaluation server, don't change this
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1') # frame

    parser.add_argument('--eval-config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    
    parser.add_argument('--dynamic-schedule', action='store_true', default=False)

    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--forecast-rt-ub', type=float, default=0.003) # seconds
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--perf-factor', type=float, default=1)

    opts = parser.parse_args()
    return opts


def det_process(opts, frame_recv, det_res_send, w_img, h_img, config, client_state):

    # initialize evaluation client using state of old client
    os.nice(45)

    try:
        model_name = opts.model_name
        dataset = opts.dataset
        model = DetectorWithRegressors(
                model_name,
                dataset,
                models_info[dataset][model_name]["config_file"],
                models_info[dataset][model_name]["checkpoint_file"],
                "cuda:0",
                None
        )
        model.change_num_proposals(100)
        model.change_scale((2000, 480))

        # print(1)
        # warm up the GPU
        _ = model.detect(np.zeros((h_img, w_img, 3), np.uint8))
        # _ = tracker.track(np.zeros((h_img, w_img, 3), np.uint8), _)
        torch.cuda.synchronize()
        eval_client = EvalClient(config, state=client_state, verbose=False)

        while 1:
            fidx = frame_recv.recv()
            if fidx == 'wait_for_ready':
                det_res_send.send('ready')
                continue
            if fidx == "new_seq":
                det_res_send.send('ready_new_seq')
                continue
            if fidx is None:
                # exit flag
                break
            fidx , t1 = fidx
            _, img = eval_client.get_frame(fid=fidx)
            t2 = perf_counter() 
            t_send_frame = t2 - t1
            result = model.detect(img)
            # print(2)
            parsed_result = model.parse_result_for_sap(result)
            torch.cuda.synchronize()
            t3 = perf_counter()
            det_res_send.send([parsed_result, t_send_frame, t3, t3-t2])

    except Exception:
        print("".join(traceback.format_exception(*sys.exc_info())))
        # report all errors from the child process to the parent
        # forward traceback info as well
        det_res_send.send(Exception("".join(traceback.format_exception(*sys.exc_info()))))


def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing
    opts = parse_args()

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']
    
    # initialize model and mapping
    config = json.load(open(opts.eval_config, 'r'))

    img = db.imgs[0]
    w_img, h_img = img['width'], img['height']

    # initialize evaluation client 
    eval_client = EvalClient(config, verbose=False)

    # launch detector process
    frame_recv, frame_send = mp.Pipe(False)
    det_res_recv, det_res_send = mp.Pipe(False)
    det_proc = mp.Process(target=det_process, args=(opts, frame_recv, det_res_send, w_img, h_img, config, eval_client.get_state()))
    det_proc.start()

    # dynamic scheduling
    if opts.dynamic_schedule:
        mean_rtf = 0.0

    # initialize arrays to store detection, sending, receiving, association and forecasting times
    t_det_all = []
    t_send_frame_all = []
    t_recv_res_all = []
    t_assoc_all = []
    t_forecast_all = []

    with torch.no_grad():
        kf_F = torch.eye(8)
        kf_F[3, 7] = 1
        kf_Q = torch.eye(8)
        kf_R = 10*torch.eye(4)
        kf_P_init = 100*torch.eye(8).unsqueeze(0)

        for seq in tqdm(seqs):

            # Request stream for current sequence from evaluation server
            eval_client.request_stream(bytes(seq))

            fidx = 0
            processing = False  
            fidx_t2 = None            # detection input index at t2
            fidx_latest = None
            tkidx = 0                 # track starting index
            kf_x = torch.empty((0, 8, 1))
            kf_P = torch.empty((0, 8, 8))
            n_matched12 = 0

            # let detector process get ready to process sequence 
            # it is possible that unfetched results remain in the pipe, this asks the detector to flush those
            frame_send.send('wait_for_ready')
            # print("main 1")
            while 1:
                msg = det_res_recv.recv() # wait till the detector is ready
                if msg == 'ready':
                    break
                elif isinstance(msg, Exception):
                    raise msg
            # print("main 2")

            t_unit = 1/opts.fps

            # get the time when stream's first frame was received
            t_start = eval_client.get_stream_start_time()

            count_detections = 0
            while fidx is not None:
                print(fidx)
                t1 = perf_counter()
                t_elapsed = t1 - t_start

                # identify latest available frame
                fidx_continous = t_elapsed*opts.fps*opts.perf_factor
                fidx, _ = eval_client.get_frame()

                if fidx is None:
                    break

                if fidx == fidx_latest:
                    # algorithm is fast and has some idle time
                    wait_for_next = True
                else:
                    wait_for_next = False
                    if opts.dynamic_schedule:
                        if mean_rtf >= 1: # when runtime < 1, it should always process every frame
                            fidx_remainder = fidx_continous - fidx
                            if mean_rtf < np.floor(fidx_remainder + mean_rtf):
                                # wait till next frame
                                wait_for_next = True

                if wait_for_next:
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

                    if opts.dynamic_schedule:
                        sum_rtf = mean_rtf*count_detections + t_det*opts.fps*opts.perf_factor
                        count_detections += 1
                        mean_rtf = sum_rtf/count_detections
                       
                    bboxes_t2, scores_t2, labels_t2, _ = \
                        parse_det_result(result, coco_mapping, n_class)

                    processing = False
                    t_det_end = perf_counter()
                    t_det_all.append(t_det)
                    t_send_frame_all.append(t_send_frame)
                    t_recv_res_all.append(t_det_end - t_start_res)

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

                if len(bboxes_t3):
                    ltwh2ltrb_(bboxes_t3)

                # send result to benchmark toolkit
                if fidx_t2 is not None:
                    eval_client.send_result_to_server(bboxes_t3, scores_t3, labels_t3)

            print("detection time: ", 1e3*np.array(t_det_all).mean())
            print("association time: ", 1e3*np.array(t_assoc_all).mean())
            print("sending time: ", 1e3*np.array(t_send_frame_all).mean())
            print("receiving time: ", 1e3*np.array(t_recv_res_all).mean())
            print("forecasting time: ", 1e3*np.array(t_forecast_all).mean())
        
            # stop current stream
            print("Stopping stream ", seq)
            eval_client.stop_stream()

            frame_send.send('new_seq')
            while 1:
                msg = det_res_recv.recv() # wait till the detector is ready
                if msg == 'ready_new_seq':
                    break
                elif isinstance(msg, Exception):
                    raise msg


    # shut down evaluation client
    eval_client.close()

    # terminates the child process
    frame_send.send(None)
 
    # convert to ms for display
    s2ms = lambda x: 1e3*x
    print_stats(t_det_all, 'Runtime detection (ms)', cvt=s2ms)
    print_stats(t_send_frame_all, 'Runtime sending the frame (ms)', cvt=s2ms)
    print_stats(t_recv_res_all, 'Runtime receiving the result (ms)', cvt=s2ms)
    print_stats(t_assoc_all, "Runtime association (ms)", cvt=s2ms)
    print_stats(t_forecast_all, "Runtime forecasting (ms)", cvt=s2ms)

if __name__ == '__main__':
    main()