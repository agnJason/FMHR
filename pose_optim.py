import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam

def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # c2w
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    # convert to w2c
    pose = np.linalg.inv(pose)

    return intrinsics, pose

def get_data(data_path, scan_id, num=16, res=(1280, 1024)):
    camera_dict = np.load(join(data_path, '%d/camera/param.npz' % scan_id))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(num)]

    w2cs = []
    projs = []
    Pall = []
    # scale_mat = torch.from_numpy(scale_mats[0].astype(np.float32)).cuda()
    for i in range(num):
        P = world_mats[i] @ scale_mats[i]
        Pall.append(torch.from_numpy(P[:3]).float().cuda())
        proj, w2c = load_K_Rt_from_P(P[:3])

        proj[0, 0] = proj[0, 0] / (res[0] / 2.)
        proj[0, 2] = proj[0, 2] / (res[0] / 2.) - 1.
        proj[1, 1] = proj[1, 1] / (res[1] / 2.)
        proj[1, 2] = proj[1, 2] / (res[1] / 2.) - 1.
        proj[2, 2] = 0.
        proj[2, 3] = -0.1
        proj[3, 2] = 1.
        proj[3, 3] = 0.

        projs.append(proj.astype(np.float32))
        w2cs.append(w2c.astype(np.float32))

    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()
    Pall = torch.stack(Pall, dim=0)

    poses = []
    weights = np.ones((num, 42))
    for i in range(num):
        f = open(join(data_path, '%d/pose/%02d.json' % (scan_id, i)))
        pose_f = json.load(f)
        f.close()
        if len(pose_f['Left']) == 0:
            pose_f['Left'] = [-1.0] * 63
            weights[i, :21] = 0
        if len(pose_f['Right']) == 0:
            pose_f['Right'] = [-1.0] * 63
            weights[i, 21:] = 0
        pose_t = np.array(pose_f['Left'] + pose_f['Right']).reshape(-1, 3)
        assert pose_t.shape[0] == 42
        pose_t = torch.from_numpy(pose_t).float().cuda()
        # pose_t[:, 0] = (pose_t[:, 0] * -0.5 + 0.5) * res[0]
        # pose_t[:, 1] = (pose_t[:, 1] * 0.5 + 0.5) * res[1]
        poses.append(pose_t)
    poses = torch.stack(poses, dim=0)
    weights = torch.from_numpy(np.stack(weights)).cuda()
    return w2cs, projs, poses, Pall, weights

def pose_optimize(batch, epoch, w2cs, projs, poses, weights, init_interhand=False):
    num = w2cs.shape[0]
    if init_interhand:
        pred_poses = torch.cat([torch.zeros_like(poses[0, :, :2]), torch.ones_like(poses[0, :, :1])], 1)
    else:
        pred_poses = torch.cat([poses[0, :, :2], torch.zeros_like(poses[0, :, :2]).cuda()], 1)
        # pred_poses[poses[0, :, 2] < 0.5] = 0
        c2w = torch.inverse(w2cs[:1])
        proj_ = torch.inverse(projs[:1])
        pred_poses = torch.einsum('ijk,ikl->ijl', pred_poses[None, :], c2w)[0, :, :3]
    pred_poses.requires_grad_(True)
    optimizer = Adam([pred_poses], lr=0.005)
    pbar = tqdm(range(epoch))
    weights = weights.repeat(batch, 1)[:, :, None]
    for i in pbar:
        n = num * batch
        w2c = w2cs.repeat(batch, 1, 1)
        proj = projs.repeat(batch, 1, 1)
        pose = torch.clone(poses).repeat(batch, 1, 1)

        points = torch.cat([pred_poses, torch.ones_like(pred_poses[:, 0:1])], axis=1).unsqueeze(0).expand(n, -1, -1)
        rot_points = torch.einsum('ijk,ikl->ijl', points, w2c)
        proj_points = torch.einsum('ijk,ikl->ijl', rot_points, proj)
        proj_uv = proj_points[:, :, :2] / (proj_points[:, :, 3:4])
        # confidence = pose[:, :, 2]
        pose_tmp = pose[:, :, :2]

        # pose_tmp[confidence < threshold] = proj_uv[confidence < threshold]
        loss = F.l1_loss(proj_uv * weights, pose_tmp * weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        des = 'loss:%.4f' % loss.item()
        pbar.set_description(des)
    return pred_poses.detach()

def main(scan_id, data_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    out_mesh_dire = out_path + '/keypoints3d'
    os.makedirs(out_mesh_dire, exist_ok=True)

    num = 16
    batch = 200
    epoch = 500
    w2cs, projs, poses, Pall, weights = get_data(data_path, scan_id, num)
    pred_poses = pose_optimize(batch, epoch, w2cs, projs, poses, weights)

    np.savetxt(join(out_mesh_dire, 'keypoints_3d_%d.xyz' % scan_id), pred_poses.cpu().numpy())
    # pred_poses[:67] + [84:135]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_id', type=int, default=6)
    parser.add_argument('--range', type=int)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    args = parser.parse_args()
    if args.range is not None:
        for i in range(args.range):
            main(i + 1, args.data_path, args.out_path)
    else:
        main(args.scan_id, args.data_path, args.out_path)