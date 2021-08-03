import torch
from collections import defaultdict
import numpy as np
import mcubes
import trimesh

from models.rendering import *
from models.nerf import *

from datasets import dataset_dict

from utils import load_ckpt

img_wh = (1440, 1920) # full resolution of the input images
# img_wh = (4032, 3024) # full resolution of the input images
dataset_name = 'llff' # blender or llff (own data)
scene_name = 'test' # whatever you want
root_dir = "/home/ischakra/data/objectron-cup/example_0/" # the folder containing data (images/ and poses_bounds.npy)
ckpt_path = "/home/ischakra/src/code/nerf_pl/ckpts/objectron-cup/epoch=2_v0.ckpt" # the model path

# root_dir = "/home/ischakra/data/silica" # the folder containing data (images/ and poses_bounds.npy)
# ckpt_path = "/home/ischakra/src/code/nerf_pl/ckpts/exp/silica.ckpt" # the model path

###############

kwargs = {'root_dir': root_dir,
          'img_wh': img_wh}
if dataset_name == 'llff':
    kwargs['spheric_poses'] = True
    kwargs['split'] = 'test'
else:
    kwargs['split'] = 'train'
    
chunk = 1024*32
dataset = dataset_dict[dataset_name](**kwargs)

embedding_xyz = Embedding(3, 10)
embedding_dir = Embedding(3, 4)

nerf_fine = NeRF()
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')
nerf_fine.cuda().eval();


### Tune these parameters until the whole object lies tightly in range with little noise ###
N = 128 # controls the resolution, set this number small here because we're only finding
        # good ranges here, not yet for mesh reconstruction; we can set this number high
        # when it comes to final reconstruction.
xmin, xmax = -1, 1 # left/right range
ymin, ymax = -1, 1 # forward/backward range
zmin, zmax = -2.64, -0.64 # up/down range
## Attention! the ranges MUST have the same length!
sigma_threshold = 20. # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
############################################################################################

x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
z = np.linspace(zmin, zmax, N)

xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
dir_ = torch.zeros_like(xyz_).cuda()

with torch.no_grad():
    B = xyz_.shape[0]
    out_chunks = []
    for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_[i:i+chunk]) # (N, embed_xyz_channels)
        dir_embedded = embedding_dir(dir_[i:i+chunk]) # (N, embed_dir_channels)
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
        out_chunks += [nerf_fine(xyzdir_embedded)]
    rgbsigma = torch.cat(out_chunks, 0)
    
sigma = rgbsigma[:, -1].cpu().numpy()
sigma = np.maximum(sigma, 0)
sigma = sigma.reshape(N, N, N)

vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)

mesh = trimesh.Trimesh(vertices/N, triangles)
# mesh.show()

# You can already export "colorless" mesh if you don't need color
mcubes.export_mesh(vertices, triangles, f"{scene_name}.dae")
