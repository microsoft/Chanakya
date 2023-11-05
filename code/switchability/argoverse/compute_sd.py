import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

model_name = ['yolov3', 'fcos', 'faster_rcnn']
# model_name = ['yolo', 'faster_rcnn_101', 'cascade_rcnn']
# model_name = ['yolo', 'faster_rcnn', 'cascade_rcnn']
# model_name = ['fcos', 'faster_rcnn', 'yolo']
# model_name = ['fcos', 'faster_rcnn', 'cascade_rcnn']

file_name = ""
for i in range(len(model_name)):
    if i == len(model_name)-1:
        file_name += model_name[i]
    else:
        file_name += model_name[i] + "_" 

#  metric = ['score', 'coco', 'imagenet_vid', 'argoverse']
metric = sys.argv[1] 

annType = 'bbox'
dataDir='../datasets/coco'
type_data = sys.argv[2]
dataType= type_data + '2017'
sd_path = '/mnt/IoU_results_argoverse/'

for i in range(len(model_name)):
    path = '/mnt/IoU_results_argoverse/' + model_name[i] + '/' + type_data + '/'
    csv_path_mean = path + type_data + '_mean_all_'  + metric +'.csv'
    if i ==0:
        df = np.array(pd.read_csv(csv_path_mean,  usecols = ['total_mean']))
        # print(df.shape)
    else:
        df_2 = np.array(pd.read_csv(csv_path_mean, usecols = ['total_mean']))
        df = np.hstack((df, df_2))   

sd = np.std(df, axis = 1)
columns_name = ['image_id', 'SD']
data =  np.array(pd.read_csv(csv_path_mean, usecols=['image_id']))
data = np.squeeze(data)
print(data.shape)
print(sd.shape)
data = np.vstack((data, sd))
print(data.shape)
df = pd.DataFrame(np.transpose(data), columns = columns_name) 
csv_path_sd =  sd_path + type_data + '_' + file_name +'_' + "argoverse" + '_sd' +'.csv'
# csv_path_sd =  sd_path + type_data + '_' + file_name + '_score_sd' +'.csv'

df.to_csv(csv_path_sd, index = False)

print(sd)
# k = np.histogram(sd)
# print(k)
b, bins, patches = plt.hist(sd, bins = 50 , range = (0, 0.5))
fig = './sd_iou_yolo_fcos_faster_rcnn_'+type_data+'.png'
plt.savefig(fig)



# # plt.show()  
# # print(df.shape)
