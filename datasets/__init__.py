from .blender import BlenderDataset
from .llff import LLFFDataset
from .shapenet import ShapenetDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'shapenet': ShapenetDataset}