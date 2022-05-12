from pathlib import Path
import json
import imageio
import torch
from torch.utils.data import Dataset
import pdb
from .ray_utils import *
# from ray_utils import *

class ShapenetDataset(Dataset):
    """
    returns the images, poses and instrinsics of a partiucular scene
    """
    def __init__(self, root_dir, split='train', num_views=25):
        """
        Args:
            all_folders (list): list of folder paths. each folder contains indiviual scene info
            num_views (int): number of views to return for each scene
        """
        super().__init__()

        root_path = Path(root_dir)
        splits_path = Path("/data/ischakra/synthetic/shapenet-20211122T194556Z-001/shapenet/car_splits_exists.json")
        with open(splits_path, "r") as splits_file:
            splits = json.load(splits_file)
        all_folders = [root_path.joinpath(foldername) for foldername in sorted(splits[split])]
        self.all_folders = all_folders
        self.num_views = num_views

    def __getitem__(self, idx):
        folderpath = self.all_folders[idx]
        meta_path = folderpath.joinpath("transforms.json")
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)
        
        all_imgs = []
        all_poses = []
        all_rays = []
        # get 25 views
        for frame_idx in range(self.num_views):
            frame = meta_data["frames"][frame_idx]

            img_name = f"{Path(frame['file_path']).stem}.png"
            img_path = folderpath.joinpath(img_name)
            img = imageio.imread(img_path)
            all_imgs.append(torch.as_tensor(img, dtype=torch.float))

            pose = frame["transform_matrix"]
            all_poses.append(torch.as_tensor(pose, dtype=torch.float))

        all_imgs = torch.stack(all_imgs, dim=0) / 255.
        # composite the images to a white background
        all_imgs = all_imgs[...,:3] * all_imgs[...,-1:] + 1-all_imgs[...,-1:]

        all_poses = torch.stack(all_poses, dim=0)

        # all images of a scene has the same camera intrinsics
        H, W = all_imgs[0].shape[:2]
        camera_angle_x = meta_data["camera_angle_x"]
        camera_angle_x = torch.as_tensor(camera_angle_x, dtype=torch.float)
        
        # camera angle equation: tan(angle/2) = (W/2)/focal
        focal = 0.5 * W / torch.tan(0.5 * camera_angle_x)
        hwf = torch.as_tensor([H, W, focal], dtype=torch.float)

        # all shapenet scenes are bounded between 2. and 6.
        near = 2.
        far = 6.
        bound = torch.as_tensor([near, far], dtype=torch.float)

        pixels = all_imgs.reshape(-1, 3)
        rays_o, rays_d = get_rays_shapenet(hwf, all_poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

        all_rays = torch.cat([rays_o, rays_d, 
                                near*torch.ones_like(rays_o[:, :1]),
                                far*torch.ones_like(rays_o[:, :1])],
                                1) # (h*w, 8)
        
        sample = {'rays': all_rays, 'rgbs': pixels}
        return sample
    
    def __len__(self):
        return len(self.all_folders)


def build_shapenet(image_set, dataset_root, splits_path, num_views, split):
    """
    Args:
        image_set: specifies whether to return "train", "val" or "test" dataset
        dataset_root: root path of the dataset
        splits_path: file path that specifies train, val and test split
        num_views: num of views to return from a single scene
    """
    root_path = Path(dataset_root)
    splits_path = Path(splits_path)
    with open(splits_path, "r") as splits_file:
        splits = json.load(splits_file)
    all_folders = [root_path.joinpath(foldername) for foldername in sorted(splits[image_set])]
    dataset = ShapenetDataset(all_folders, num_views)

    return dataset

from torch.utils.data import DataLoader
if __name__ == '__main__':
    train_dataset = ShapenetDataset(split='train', root_dir='/data/ischakra/synthetic/shapenet-20211122T194556Z-001/shapenet/cars/02958343/')
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=25)

    for data in train_dataloader:
        pdb.set_trace()

    