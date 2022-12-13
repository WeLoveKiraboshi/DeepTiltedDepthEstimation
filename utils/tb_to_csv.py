import numpy as np
import argparse
import time
import datetime
import os
import cv2
import sys
sys.path.append(os.getcwd())
import pickle
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import csv
import pandas as pd

def export_list_csv(export_list, csv_dir):
    with open(csv_dir, "w") as f:
        writer = csv.writer(f, lineterminator='\n')

        if isinstance(export_list[0], list): #多次元の場合
            writer.writerows(export_list)

        else:
            writer.writerow(export_list)
            
            

path = '/home/yukisaito/TiltedDepthEstimation/results/sr_resnetunet_full_sr_1_net_1/tb/events.out.tfevents.1657341769.DL-BoxII'
event_acc = EventAccumulator(path, size_guidance={'scalars': 0})
event_acc.Reload() # ログファイルのサイズによっては非常に時間がかかる

scalars = {}
for tag in event_acc.Tags()['scalars']:
    if tag == 'SRDenseDepthFullLoss_train':
    	events = event_acc.Scalars(tag)
    	scalars[tag] = [event.value for event in events]


#with open('full_loss_scalars.json', 'w') as fout:
#    json.dump(scalars, fout)

csvfile = "./full_loss_scalar_sr_1_net_1.csv"
#for tag in scalars.keys():    
my_df = pd.DataFrame(scalars)
my_df.to_csv(csvfile, index=False, header=False)
