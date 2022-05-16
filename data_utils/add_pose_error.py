#adds a percent of pose error to the current pose estimates (HeT, colmap, extrinsics, intrinsics)
from pathlib import Path
import argparse
import numpy as np
np.random.seed(seed=0)
import pdb
import shutil
from os.path import join
import os
import glob

parser = argparse.ArgumentParser(description='Adds a percent of pose error to the current pose estimates (HeT or colmap)')
parser.add_argument('--root_dir', type=str, help='rootdir')
parser.add_argument('--pose', type=str, default='colmap', choices=['colmap','HeT'],)
parser.add_argument('--pose_type', type=str, default='extrinsics', choices=['extrinsics','intrinsics'],)
parser.add_argument('--percent_of_error', type=float, default=10.0)

args = parser.parse_args()

poses_bounds = np.load(join(args.root_dir,args.pose+'_poses',"poses_bounds.npy"))
destdir = join(args.root_dir,args.pose+'_poses',f"poses_bounds_{args.pose_type}_{args.percent_of_error}.npy")
poses = poses_bounds[:,:15].reshape(-1,3,5)
num_instances, _, _ = poses.shape
poses_orig = poses.copy()
perror = args.percent_of_error/100.
if args.pose_type == 'extrinsics':
    poses[:,:,:4] = poses[:,:,:4]+np.random.uniform(low=-poses[:,:,:4]*perror, high=poses[:,:,:4]*perror)
poses = poses.reshape(num_instances,-1)
poses_bounds[:,:15] = poses
np.save(destdir,poses_bounds)
# pdb.set_trace()