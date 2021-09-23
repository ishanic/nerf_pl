import torch
from kornia import create_meshgrid
import pdb
import numpy as np
def get_ray_directions(H, W, focal, principal_point):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    inv(K)*x = RTX
    Applies the intrinsic matrix plus some other coordinate system related stuff: 
    since the y value indexes from top to bottom, we flip it, 
    and since the camera looks along the negative z axis, we negative it. 
    In practice we didn't find the calibration precise enough for the +0.5 to matter.
    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    # (-W/2 to W/2)/focal.
    # the direction of the center pixel will be [0,0,-1]
    # if sum(principal_point) == 0:
        # directions = \
            # torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
    # else:
    directions = \
        torch.stack([(i-principal_point[0])/focal[0], -(j-principal_point[1])/focal[1], -torch.ones_like(i)], -1) # (H, W, 3)
    # directions = \
        # torch.stack([(i-principal_point[0])/focal, -(j-principal_point[1])/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions, i, j


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays (camera position) in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def project_rays(rays_o, rays_d, c2w_src, focal, H, W, principal_point):
    # project rays from target samples onto source samples. See NerFORMER, figure 5
    # pdb.set_trace()
    N_rays = rays_o.size(0)
    N_samples = 128
    near = 1; far = 4
    K = torch.tensor([[focal[0], 0, principal_point[0]],[0, focal[1], principal_point[1]],[0, 0, 1]], dtype=torch.float32)
    # K = torch.tensor([[focal, 0, principal_point[0]],[0, focal, principal_point[1]],[0, 0, 1]], dtype=torch.float32)
    z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device) # (N_samples)
    z_vals = near * (1-z_steps) + far * z_steps
    z_vals = z_vals.expand(N_rays, N_samples)
    xyz_tgt = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
    xyz_tgt = xyz_tgt[0]

    c2w_src_homo = torch.eye(4,4)
    c2w_src_homo[:3,:] = c2w_src
    xyz_tgt_homo = torch.ones(xyz_tgt.size(0),4)
    xyz_tgt_homo[:,:3] = xyz_tgt
    # xyz_cam = xyz_tgt_homo @ torch.inverse(c2w_src_homo)
    xyz_cam = (torch.inverse(c2w_src_homo) @ xyz_tgt_homo.T).T
    
    # flip y and z
    xyz_cam[:,1] = -1*xyz_cam[:,1]
    xyz_cam[:,2] = -1*xyz_cam[:,2]
    
    xyz_img = (K @ xyz_cam[...,:3].T).T
    xyz_img[:,0] = xyz_img[:,0]/xyz_img[:,2]
    xyz_img[:,1] = xyz_img[:,1]/xyz_img[:,2]
    
    xyz_tgt = xyz_tgt.numpy()
    xyz_tgt_homo = xyz_tgt_homo.numpy()
    P_c2w = np.concatenate([c2w_src.numpy(), np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
    ## project vertices from world coordinate to camera coordinate
    vertices_cam = (P_w2c @ xyz_tgt_homo.T) # (3, N) in "right up back"
    vertices_cam[1:] *= -1 # (3, N) in "right down forward"
    ## project vertices from camera coordinate to pixel coordinate
    vertices_image = (K.numpy() @ vertices_cam).T # (N, 3)
    depth = vertices_image[:, -1:]+1e-5 # the depth of the vertices, used as far plane
    vertices_image = vertices_image[:, :2]/depth
    vertices_image = vertices_image.astype(np.float32)
    vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W-1)
    vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H-1)

    # pdb.set_trace()
    return xyz_img[...,:2].int(), vertices_image
    
    

def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d