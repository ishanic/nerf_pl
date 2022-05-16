from genericpath import exists
from project_hand_eye_to_pv import load_pv_data, match_timestamp
from pathlib import Path
import argparse
import numpy as np
import pdb
import imageio
import shutil
from os.path import join
import os
import glob
parser = argparse.ArgumentParser(description='Convert Hololens2ForCV data to Nerf_pl format')
parser.add_argument('--recording_path', type=str, help='recording_path')
args = parser.parse_args()
recording_path = Path(args.recording_path)
pv_info_path = sorted(recording_path.glob(r'*pv.txt'))
has_pv = len(list(pv_info_path)) > 0
if has_pv:
    (pv_timestamps, focal_lengths, pv2world_transforms, ox,
        oy, _, _) = load_pv_data(list(pv_info_path)[0])
    principal_point = np.array([ox, oy])


pv_image_paths = sorted(recording_path.glob('PV/*.png'))
depth_image_paths = sorted(recording_path.glob('Depth Long Throw/*.pgm'))
image_dir = join(args.recording_path,'images/')
os.makedirs(image_dir, exist_ok=True)
H, W, _ = imageio.imread(pv_image_paths[0]).shape
focal = np.mean(focal_lengths)
intrinsics = np.array([H,W,focal])[:,None]
poses_bounds = []
target_ids = []
for i, depth_image_path in enumerate(depth_image_paths):
    if '_ab.pgm' in str(depth_image_path): continue
    depth_timestamp = float(Path(depth_image_path).stem)
    depth_image = imageio.imread(depth_image_path)
    far = np.percentile(depth_image,99.9)/1000
    far = min(far, 10)
    depth_image[depth_image==0] = 1e6
    near = np.percentile(depth_image,0.1)/1000
    near = max(near, 0.1)
    target_id = match_timestamp(depth_timestamp, pv_timestamps)
    if target_id in target_ids: continue
    target_ids.append(target_id)
    pv2world = pv2world_transforms[target_id]
    if pv2world.min() < -100: print('pv2world.min() < -100'); continue
    if pv2world.max() > 100: print('pv2world.max() > 100'); continue
    poses = np.concatenate((pv2world[:3,:], intrinsics), axis=1).reshape(-1)
    near_far = np.array([near,far])
    poses_bounds.append(np.concatenate((poses, near_far), axis=0))
    shutil.copy(pv_image_paths[target_id], image_dir)
    # num_images = sum(1 for x in recording_path.glob('images/*.png'))
    # if num_images!=len(poses_bounds):
    #     print('doesnt match')
poses_bounds = np.array(poses_bounds)
np.save(join(args.recording_path,"poses_bounds.npy"), poses_bounds)
# np.save(join(args.recording_path,"missing_idx.npy"), np.empty(0))
num_images = sum(1 for x in recording_path.glob('images/*.png'))
if num_images!=poses_bounds.shape[0]:
    print('doesnt match')

# reduced number of images    

poses_bounds = np.load(join(args.recording_path,"poses_bounds.npy"))

image_dir = join(args.recording_path,'few/images')
os.makedirs(image_dir, exist_ok=True)
original_image_list = sorted(glob.glob(join(args.recording_path,'images/*.png')))
num_images = len(original_image_list)
indexes = np.unique((np.linspace(0,num_images-1,50)).astype(int))
poses_bounds = poses_bounds[indexes]
for index in indexes:
    shutil.copy(original_image_list[index], image_dir)
np.save(join(args.recording_path,'few/poses_bounds.npy'), poses_bounds)

num_images = sum(1 for x in recording_path.glob('few/images/*.png'))
if num_images!=len(poses_bounds):
    pdb.set_trace()
    



# poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
# self.bounds = poses_bounds[:, -2:] # (N_images, 2)
# # Step 1: rescale focal length according to training resolution
# H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
    
    

