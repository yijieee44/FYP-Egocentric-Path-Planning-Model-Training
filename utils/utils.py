import cv2
import numpy as np
import torch
import warnings
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array([
  [FOCAL,   0.,   W/2.],
  [  0.,  FOCAL,  H/2.],
  [  0.,    0.,     1.]])

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T


def draw_path_from_device_path(device_path, img, width=1, height=1.2, fill_color=(1,1,1), line_color=(1, 1, 1)):
    img_copy = img.copy()
    device_path_l = device_path + np.array([0, 0, height])
    device_path_r = device_path + np.array([0, 0, height])
    device_path_l[:, 1] -= width
    device_path_r[:, 1] += width

    img_points_norm_l = img_from_device(device_path_l)
    img_points_norm_r = img_from_device(device_path_r)
    img_pts_l = denormalize(img_points_norm_l, img.shape)
    img_pts_r = denormalize(img_points_norm_r, img.shape)

    # filter out things rejected along the way
    valid = np.logical_and(np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1))
    img_pts_l = img_pts_l[valid].astype(int)
    img_pts_r = img_pts_r[valid].astype(int)

    for i in range(1, len(img_pts_l)):
        u1,v1,u2,v2 = np.append(img_pts_l[i-1], img_pts_r[i-1])
        u3,v3,u4,v4 = np.append(img_pts_l[i], img_pts_r[i])
        pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
        cv2.fillPoly(img_copy,[pts],fill_color)
        cv2.polylines(img_copy,[pts],True,line_color)
    
    return img_copy


def img_from_device(pt_device):
    input_shape = pt_device.shape
    pt_device = np.atleast_2d(pt_device)
    pt_view = np.einsum('jk,ik->ij', view_frame_from_device_frame, pt_device)

    pt_view[pt_view[:,2] < 0] = np.nan

    pt_img = pt_view/pt_view[:,2:3]
    return pt_img.reshape(input_shape)[:,:2]


def denormalize(img_pts, img_shape):
    # denormalizes image coordinates
    # accepts single pt or array of pts
    img_pts = np.array(img_pts)
    input_shape = img_pts.shape
    img_pts = np.atleast_2d(img_pts)
    img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))

    scaling_matrix = get_scale(img_shape)

    img_pts_denormalized = scaling_matrix.dot(img_pts.T).T
    # img_pts_denormalized[img_pts_denormalized[:,0] > W] = np.nan
    # img_pts_denormalized[img_pts_denormalized[:,0] < 0] = np.nan
    # img_pts_denormalized[img_pts_denormalized[:,1] > H] = np.nan
    # img_pts_denormalized[img_pts_denormalized[:,1] < 0] = np.nan
    return img_pts_denormalized[:,:2].reshape(input_shape)


def get_scale(img_shape):
    temp = eon_intrinsics.copy()

    temp[0] = temp[0] * (img_shape[1] / FULL_FRAME_SIZE[0])
    temp[1] = temp[1] * (img_shape[0] / FULL_FRAME_SIZE[1])

    return temp


def target_denormalize_mean_std_cuda(x_norm):
    mean = torch.tensor([35.4279,  0.7769, -1.7467]).cuda()
    std = torch.tensor([22.9786,  1.4706,  1.3428]).cuda()
    x = (x_norm * std) + mean
    return x


def target_denormalize_mean_std_tensor(x_norm):
    mean = torch.tensor([35.4279,  0.7769, -1.7467])
    std = torch.tensor([22.9786,  1.4706,  1.3428])
    x = (x_norm * std) + mean
    return x


def target_denormalize_mean_std_np(x_norm):
    mean = np.array([35.4279,  0.7769, -1.7467])
    std = np.array([22.9786,  1.4706,  1.3428])
    x = (x_norm * std) + mean
    return x
