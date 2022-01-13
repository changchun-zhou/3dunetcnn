import cv2
import os
import torch 
import numpy as np
from fnmatch import fnmatch 
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--checkpoint", type=str, default="")
# parser.add_argument("--save_file", type=str, default="")
args = parser.parse_args()

State_FileName = os.path.join(args.checkpoint+'.txt')
checkpoint = torch.load(os.path.join(args.checkpoint),map_location=lambda storage, loc: storage)

with open (State_FileName,'w') as State_File:
	State_File.writelines(str(checkpoint))

# model.load_state_dict(checkpoint["state_dict"])
# params = model.state_dict()
params = checkpoint
for k in params["state_dict"]:
    print(k)
for k in params["opt_dict"]:
    print(k, params['opt_dict'][k])
