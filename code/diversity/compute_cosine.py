import pickle 
from scipy import spatial
import sys

type_data = sys.argv[1]
models_name = ['yolo', 'fcos', 'faster_rcnn', 'faster_rcnn_101', 'cascade_rcnn']
input_file_path  =  "/mnt/IoU_results_coco/" + 'per_class_mAP_' + type_data + '.pkl'

with open(input_file_path, 'rb') as handle:
    final_result_dict = pickle.load(handle)

result_dict = {}
for model_name_1 in models_name:
    for model_name_2 in models_name:
        diff_string = model_name_1 + '_' +model_name_2
        # result_dict[diff_string] = spatial.distance.cosine(normalize(final_result_dict[model_name_1], norm="l1"), normalize(final_result_dict[model_name_2], norm = "l1"))
        result_dict[diff_string] = spatial.distance.cosine(final_result_dict[model_name_1], final_result_dict[model_name_2])

print(result_dict)