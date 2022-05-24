import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
# from ray_utils import *
import pdb
import random
# import cv2
# use tips mentioned here for arbitrary datasets:
# https://github.com/kwea123/nerf_pl/issues/50

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center: all the last columns (translations), mean across images
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    # the average pose should be identity after this step
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378), spheric_poses=False, val_num=1, crop=False):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.crop = crop
        self.define_transforms() #totensor

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                        # load full resolution image then resize
        if os.path.exists(os.path.join(self.root_dir,'missing_idx.npy')):
            missing_idx = np.load(os.path.join(self.root_dir,'missing_idx.npy'))
            self.image_paths = np.delete(self.image_paths, missing_idx, axis=0)

        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2) near far for each image
        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images, Nx3x5, any image, all rows, last column 
        # assert H*self.img_wh[0] == W*self.img_wh[1], \
        #     f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        # self.focal *= self.img_wh[0]/W
        self.focal_test = self.focal * (self.img_wh[0]/W)
        #### crop image based on mask
        principal_point = np.zeros((1,2))
        if self.crop == True:
            # pdb.set_trace()
            boxes_xyxy = np.load(os.path.join(self.root_dir, 'boxes_xyxy.npy')) # (N_images, 6)=> x_min,ymin,x_max,y_max,W,H
            assert len(boxes_xyxy) == len(self.image_paths), \
                    'Mismatch between number of images and number of boxes!'

            self.principal_point_px = -1.0 * (principal_point - 1.0) * boxes_xyxy[:,4::]/2
            self.principal_point_px -= boxes_xyxy[:,:2]

            scale_x = self.img_wh[0]/(boxes_xyxy[:,2]-boxes_xyxy[:,0])
            scale_y = self.img_wh[1]/(boxes_xyxy[:,3]-boxes_xyxy[:,1])
            scale = np.vstack((scale_x, scale_y)).min(axis=0)

        else:
            boxes_xyxy = np.tile(np.array([0,0,W,H,W,H])[None,:], (len(self.image_paths),1))
            self.principal_point_px = -1.0 * (principal_point - 1.0) * boxes_xyxy[:,4::]/2
            self.principal_point_px -= boxes_xyxy[:,:2]
            scale_x = self.img_wh[0]/W
            scale_y = self.img_wh[1]/H
            scale = np.vstack((scale_x, scale_y)).min(axis=0)

        self.boxes_xyxy = boxes_xyxy

        self.principal_point_px[:,0] *= scale_x
        self.principal_point_px[:,1] *= scale_y
        self.focal_x = self.focal * scale_x
        self.focal_y = self.focal * scale_y
        if np.ndim(scale_x) == 0:
            self.focal = np.tile(np.array([self.focal_x, self.focal_y])[None,:], (len(self.image_paths),1))
        else:
            self.focal = np.hstack((self.focal_x[:,None], self.focal_y[:,None]))

        # self.principal_point_px = self.principal_point_px.mean(axis=0)
        # self.focal = self.focal_length_per_image.mean()
        # pdb.set_trace()
        #############################

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        
        self.poses, self.pose_avg = center_poses(poses)
        # for i,pose in enumerate(self.poses):
        #     np.save('/home/ischakra/data/silica/normalized_poses/%05d'%i,pose)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)

        # hacks for debugging
        # self.image_paths = self.image_paths[0:10]
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        # near_original = self.poses[...,3].min() # doesn't work
        scale_factor = near_original*0.95 # 0.75 is the default parameter
                                          # the nearest depth self.bounds.min() is at 1/0.75=1.33

        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor
        # pdb.set_trace()
        # ray directions for all pixels, same for all images (same H, W, focal)
        # Pixel coordinates to camera coordinates. 
        # The ray from center pixel to camera has direction = [0,0,-1]

        # self.directions, W_index, H_index = \
        #     get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal, principal_point=self.principal_point_px) # (H, W, 3)
        # W_index = W_index.reshape(-1)
        # H_index = H_index.reshape(-1)
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                # assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                #     f'''{image_path} has different aspect ratio than img_wh, 
                #         please check your data!'''

                if self.crop == True:
                    xmin, ymin, xmax, ymax, _, _ = boxes_xyxy[i,:]
                    # pdb.set_trace()
                    img = img.crop((xmin, ymin, xmax, ymax))

                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                # rays_o is the camera pose (origin of rays in world coordinates) repeated pixel times
                # rays_d are the camera to image ray directions in world coordinates
                
                directions, W_index, H_index = \
                    get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal[i,:], principal_point=self.principal_point_px[i,:]) # (H, W, 3)
                    # get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal, principal_point=self.principal_point_px[i,:]) # (H, W, 3)
                    
                    
                W_index = W_index.reshape(-1)
                H_index = H_index.reshape(-1)
                # pdb.set_trace()
                rays_o, rays_d = get_rays(directions, c2w) # both (hxwx3, 3x4)
                
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min() #1.33
                    # far = min(8 * near, self.bounds.max()) # focus on central object only, 8*1.33
                    far = self.bounds.max()
                
                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 9)
                
                # self.all_rays += [torch.cat([i*torch.ones_like(rays_o[:, :1]),
                #                              rays_o, rays_d, 
                #                              near*torch.ones_like(rays_o[:, :1]),
                #                              far*torch.ones_like(rays_o[:, :1])],
                #                              1)] # (h*w, 9)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)

            vis=False
            if vis == True:
                tgt_image_id = 0
                # pdb.set_trace()
                tgt_rays = self.all_rays[(self.all_rays[:, 0] == tgt_image_id).nonzero().squeeze(1)]
                selected_id = random.randint(int(tgt_rays.size(0)/4),int(3*tgt_rays.size(0)/4))
                # selected_id = 1101144 
                # selected_id = 63128
                
                # selected_id = 889524
                # selected_id = 29788
                # selected_id = 44047
                print(self.img_wh, W, H, selected_id, W_index[selected_id], H_index[selected_id])
                
                tgt_image = Image.open(self.image_paths[tgt_image_id])

                if self.crop == True:
                    xmin, ymin, xmax, ymax, _, _ = boxes_xyxy[tgt_image_id,:]
                    tgt_image = tgt_image.crop((xmin, ymin, xmax, ymax))
                    
                tgt_image = np.array(tgt_image.resize(self.img_wh, Image.LANCZOS))                
                # try:
                #     cv2.circle(tgt_image, tuple(np.array([W_index[selected_id], H_index[selected_id]])), 10, (255,0,255), 2)
                # except:
                #     pdb.set_trace()                    
                tgt_rays = tgt_rays[selected_id,:].unsqueeze(0)
                
                src_image_id = 1
                # if src and tgt are same, tgt_rays[...,1:4] == poses[src_image][:,3]
                
                coords_1, coords_2 = project_rays(tgt_rays[...,1:4], tgt_rays[...,4:7], torch.FloatTensor(self.poses[src_image_id]), self.focal[src_image_id,:], self.img_wh[1], self.img_wh[0], self.principal_point_px[src_image_id,:])
                # coords_1, coords_2 = project_rays(tgt_rays[...,1:4], tgt_rays[...,4:7], torch.FloatTensor(self.poses[src_image_id]), self.focal, self.img_wh[1], self.img_wh[0], self.principal_point_px[src_image_id,:])

                src_image = Image.open(self.image_paths[src_image_id])

                if self.crop == True:
                    xmin, ymin, xmax, ymax, _, _ = boxes_xyxy[src_image_id,:]
                    src_image = src_image.crop((xmin, ymin, xmax, ymax))
                    
                src_image = np.array(src_image.resize(self.img_wh, Image.LANCZOS))

                # for coord in coords_1:
                #     cv2.circle(src_image, tuple(np.array(coord)), 2, (255,0,0), 2)
                # for coord in coords_2:
                #     cv2.circle(src_image, tuple(np.array(coord)), 2, (0,0,255), 2)

                # cv2.imwrite("tgt_image.jpg", tgt_image)
                # cv2.imwrite("src_image.jpg", src_image)
                # pdb.set_trace()

            
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]


        else: # for testing, create a parametric rendering path
            
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
                # pdb.set_trace()
            elif not self.spheric_poses:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                # here
                # radius = 1.1 * self.bounds.min()
                radius = 2 * self.bounds.min()
                # radius = .1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                # pdb.set_trace()
                c2w = torch.FloatTensor(self.poses_test[idx])
            directions, W_index, H_index = \
                    get_ray_directions(self.img_wh[1], self.img_wh[0], np.array([self.focal_test, self.focal_test]), principal_point=np.array([self.img_wh[0]/2, self.img_wh[1]/2])) # (H, W, 3)
            # directions, W_index, H_index = \
                    # get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal[idx,:], principal_point=self.principal_point_px[idx,:]) # (H, W, 3)

            rays_o, rays_d = get_rays(directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                # far = min(8 * near, self.bounds.max())
                far = self.bounds.max()

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            
            sample = {'rays': rays,
                      'c2w': c2w}
            
            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img

        return sample

if __name__ == '__main__':
    # img_wh = (1440, 1920)
    # img_wh = (4032, 3024)
    # img_wh = (1051, 1869)
    # img_wh = (350, 623)
    img_wh = (256, 256)
    # img_wh = (4512, 3008)
    # dataset = LLFFDataset('/home/ischakra/data/objectron-cup/example_0/', 'val',spheric_poses=True, img_wh=img_wh)
    # train loads all images, and all rays in a single tensor. 
    # dataset = LLFFDataset('/data/synthetic/nerf_real_360/veena_player/', 'train',spheric_poses=True, img_wh=img_wh)
    # dataset = LLFFDataset('/data/ischakra/synthetic/banana', 'train',spheric_poses=True, img_wh=img_wh)
    dataset = LLFFDataset('/data/ischakra/co3d/categories/cup/30_1129_3289', 'train',spheric_poses=True, img_wh=img_wh, crop=True)
    for idx in range(0, len(dataset)):
        sample = dataset[idx]

    # dataset.read_meta()
    # project_rays(directions, c2w_tgt, c2w_src)
    # project_rays(src=0,tgt=1)