import os
import json
import time

import trimesh
from tqdm import tqdm
from os.path import join
import numpy as np
import argparse
import smplx
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.gcn import GCNDecoder
from models.smplx import batch_rodrigues
from models.utils import get_normals
mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True).cuda(), 'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False).cuda()}

class MANODataset(Dataset):

    def __init__(self, data_path, split='train', hand_type='left'):
        self.split = split
        self.hand_type = hand_type
        with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
            self.mano_params = json.load(f)
        self.data_list = []
        for cap_idx in self.mano_params.keys():
            frame_idxs = self.mano_params[cap_idx].keys()
            for frame_idx in frame_idxs:
                if self.mano_params[cap_idx][frame_idx][self.hand_type] is not None:
                    self.data_list.append([cap_idx, frame_idx])

    def get_item(self, cap_idx, frame_idx):
        mano_params = self.mano_params[cap_idx][frame_idx][self.hand_type]
        mano_pose = torch.FloatTensor(mano_params['pose']).view(-1)
        shape = torch.FloatTensor(mano_params['shape']).view(-1)
        trans = torch.FloatTensor(mano_params['trans']).view(-1)
        return mano_pose, shape, trans

    def __getitem__(self, index):
        return self.get_item(self.data_list[index][0], self.data_list[index][1])

    def __len__(self):
        return len(self.data_list)

def mano_forward(pose, shape, trans=None, type='left'):
    output = mano_layer[type](global_orient=pose[:, :3],
                                    hand_pose=pose[:, 3:],
                                    betas=shape, transl=trans)
    if type == 'right':
        tips = output.vertices[:, [745, 317, 444, 556, 673]]
    else:
        tips = output.vertices[:, [745, 317, 445, 556, 673]]
    joints = torch.cat([output.joints, tips], 1)
    # Reorder joints to match visualization utilities
    joints = joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

    return output.vertices, joints, output.joints

def compute_both_err(pred_mesh, target_mesh, pred_joint, target_joint):
    # human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
    # root align joint
    pred_mesh, target_mesh = pred_mesh - pred_joint[:, :1, :], target_mesh - target_joint[:, :1, :]
    pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

    pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
    pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

    # pred_joint, target_joint = pred_joint[:, human36_eval_joint, :], target_joint[:, human36_eval_joint,
    #                                                                       :]
    mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
    joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

    return joint_mean_error, mesh_mean_error

def get_trans_scale(joints):
    '''
    mean-> (0,0,0) joints[0]<->joints[1] -> 0.01
    :param joints: b * n * 3
    :return:
    '''
    trans = joints.mean(1, keepdim=True)  # b, 3
    scale = 0.5 / torch.sqrt(((joints[:, 1:2] - joints[:, 0:1]) ** 2).sum(2, keepdim=True))

    return trans, scale


def train(batch_size, data_path, hand_type):
    num_epoch = 50
    if hand_type == 'left':
        gcn_net = GCNDecoder('./mano/TEMPLATE_LEFT.obj', 21 * 3).cuda()
    elif hand_type == 'right':
        gcn_net = GCNDecoder('./mano/TEMPLATE_RIGHT.obj', 21 * 3).cuda()
    else:
        assert hand_type in ['left', 'right']
    traindataset = MANODataset(data_path, hand_type=hand_type)
    train_data_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=20)
    print('train data size: ', len(train_data_loader))
    valdataset = MANODataset(data_path, hand_type=hand_type, split='val')
    val_data_loader = DataLoader(valdataset, batch_size=batch_size, shuffle=False, num_workers=20)
    print('test data size: ', len(val_data_loader))

    optimizer = torch.optim.Adam(gcn_net.parameters(), lr=0.001)

    faces = torch.from_numpy(mano_layer[hand_type].faces.astype(np.int32)).long().cuda()

    for epoch in range(num_epoch):
        gcn_net.train()
        pbar = tqdm(train_data_loader)
        for train_data in pbar:
            # retrieve the data
            mano_pose = train_data[0].cuda()
            shape = train_data[1].cuda()
            mano_params = torch.cat([mano_pose, shape], 1)[:, 3:]
            trans = train_data[2].cuda()
            with torch.no_grad():
                ori_verts, joints, ori_j = mano_forward(mano_pose, shape, trans, hand_type)
                rot = torch.randn((trans.shape[0], 3)).cuda() * 4 - 4
                rot_mat = batch_rodrigues(rot)
                ntrans, nscale = get_trans_scale(joints)
                randscale = 1.2 - torch.randn(nscale.shape).cuda() * 0.4

                joints = torch.einsum('bij,bkj->bki', rot_mat, joints - ntrans) * nscale * randscale
                verts = torch.einsum('bij,bkj->bki', rot_mat, ori_verts - ntrans) * nscale * randscale
                ori_j = torch.einsum('bij,bkj->bki', rot_mat, ori_j - ntrans) * nscale * randscale
                normals = get_normals(verts, faces)

                a = verts[:, faces[:, 0].long()]
                b = verts[:, faces[:, 1].long()]
                c = verts[:, faces[:, 2].long()]
                edge_length = torch.cat(
                    [((a - b) ** 2).sum(2), ((c - b) ** 2).sum(2), ((a - c) ** 2).sum(2)], 1)

            pred_verts, preds_mano, _ = gcn_net(joints.reshape(-1, 63))
            pred_joints = torch.einsum('bij,ki->bkj', pred_verts, mano_layer[hand_type].J_regressor)

            mesh_loss = F.l1_loss(pred_verts, verts) * 2

            joints_loss = F.l1_loss(pred_joints, ori_j) * 2
            normal_loss = F.l1_loss(get_normals(pred_verts, faces), normals)

            a = pred_verts[:, faces[:, 0].long()]
            b = pred_verts[:, faces[:, 1].long()]
            c = pred_verts[:, faces[:, 2].long()]
            pred_edge = torch.cat(
                [((a - b) ** 2).sum(2), ((c - b) ** 2).sum(2), ((a - c) ** 2).sum(2)], 1)
            edge_loss = F.l1_loss(pred_edge, edge_length) * 5

            mano_loss = F.l1_loss(mano_params, preds_mano)
            p_mano_verts, _, _ = mano_forward(torch.cat([mano_pose[:, :3], preds_mano[:, :-10]], 1),
                                              preds_mano[:, -10:], trans, hand_type)
            union_loss = F.l1_loss(p_mano_verts, ori_verts) * 2
            loss = mesh_loss + joints_loss + normal_loss + edge_loss + mano_loss + union_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description('epoch: %d, mesh: %f, joints: %f, normal: %f, edge: %f, mano: %f, union: %f' % (
                epoch, mesh_loss.item(), joints_loss.item(), normal_loss.item(), edge_loss.item(), mano_loss.item(),
                union_loss.item()))

        with torch.no_grad():
            surface_error = 0.0
            joint_error = 0.0
            for val_data in val_data_loader:
                # retrieve the data
                mano_pose = val_data[0].cuda()
                shape = val_data[1].cuda()
                trans = val_data[2].cuda()

                verts, joints, ori_j = mano_forward(mano_pose, shape, trans, hand_type)
                rot = torch.randn((trans.shape[0], 3)).cuda() * 4 - 4
                rot_mat = batch_rodrigues(rot)
                ntrans, nscale = get_trans_scale(joints)

                joints = torch.einsum('bij,bkj->bki', rot_mat, joints - ntrans) * nscale
                verts = torch.einsum('bij,bkj->bki', rot_mat, verts - ntrans) * nscale
                ori_j = torch.einsum('bij,bkj->bki', rot_mat, ori_j - ntrans) * nscale

                pred_verts, mano_param, _ = gcn_net(joints.reshape(-1, 63))
                mesh = trimesh.Trimesh(vertices=pred_verts[0].detach().cpu().numpy(), faces=gcn_net.faces, process=False,
                                       maintain_order=True)
                mesh.export('gcn_out/test.obj')
                # loss = F.l1_loss(pred_verts, verts)
                joint_mean_error, mesh_mean_error = compute_both_err(pred_verts / nscale, verts / nscale,
                                 torch.einsum('bij,ki->bkj', pred_verts / nscale, mano_layer[hand_type].J_regressor), ori_j / nscale)
                surface_error += mesh_mean_error * 1000
                joint_error += joint_mean_error * 1000
            print('val MPVPE: %f, MPJPE: %f' % (surface_error / len(val_data_loader), (joint_error / len(val_data_loader))))


    torch.save(gcn_net.state_dict(), 'mano/gcn_%s_union.pth' % (hand_type))

def infer(net, joints, hand_type):
    batch = joints.shape[0]
    ntrans, nscale = get_trans_scale(joints)
    joints = (joints - ntrans) * nscale
    verts, mano, mid = net[hand_type](joints.reshape(batch, 63))
    verts = verts / nscale + ntrans
    for i in range(len(mid)):
        mid[i] = mid[i] / nscale + ntrans

    joints = torch.einsum('bij,ki->bkj', verts, mano_layer[hand_type].J_regressor)
    outputs = mano_layer[hand_type](global_orient=torch.zeros(batch, 3).cuda(), hand_pose=mano[:, :-10],
                                 betas=mano[:, -10:])

    mano_vertices = outputs.vertices
    mano_joints = outputs.joints

    scale = (torch.sqrt(((joints[:, 1:2] - joints[:, 0:1]) ** 2).sum(2, keepdim=True)) / \
            torch.sqrt(((mano_joints[:, 1:2] - mano_joints[:, 0:1]) ** 2).sum(2, keepdim=True)))
    # mano_vertices = mano_vertices * scale

    A = torch.cat([mano_vertices, torch.ones(batch, mano_vertices.shape[1], 1).cuda()], 2)
    B = torch.cat([verts, torch.ones(batch, verts.shape[1], 1).cuda()], 2)

    Rt = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(A.permute(0, 2, 1), A)), A.permute(0, 2, 1)), B)

    R = Rt[:, :3, :3]
    # rot = torch.zeros(1,3).cuda()
    rot = torch.diag(R[0]).detach().unsqueeze(0)
    scale = scale.detach().clone()
    scale.requires_grad_(True)
    rot.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': rot, 'lr': 1}])

    for i in range(100):
        loss = F.l1_loss(batch_rodrigues(rot) * scale, R.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Rt[:, :3, :3] = batch_rodrigues(rot.detach()) * scale.detach()
    mano_verts = torch.matmul(torch.cat([mano_vertices, torch.ones(batch, mano_vertices.shape[1], 1).cuda()], 2), Rt)

    return verts, mano_verts[:, :, :3], mano, Rt, mid

def eval(keypoints_path):
    joints = np.loadtxt(keypoints_path)

    left_joints = torch.from_numpy(joints[:21]).float().cuda().unsqueeze(0)
    right_joints = torch.from_numpy(joints[21:]).float().cuda().unsqueeze(0)

    gcn_net_left = GCNDecoder('./mano/TEMPLATE_LEFT.obj', 21 * 3).cuda()
    gcn_net_right = GCNDecoder('./mano/TEMPLATE_RIGHT.obj', 21 * 3).cuda()
    gcn_net_left.load_state_dict(torch.load("./mano/gcn_left.pth"))
    gcn_net_right.load_state_dict(torch.load("./mano/gcn_right.pth"))
    gcn_net = {'left': gcn_net_left, 'right':gcn_net_right}
    # with torch.no_grad():
    left_verts, mano_lverts, _, _, mid_l = infer(gcn_net, left_joints, 'left')

    right_verts, mano_rverts, _, _, mid_r = infer(gcn_net, right_joints, 'right')

    verts = torch.cat([left_verts, right_verts], 1)[0].detach().cpu().numpy()
    mano_verts = torch.cat([mano_lverts, mano_rverts], 1)[0].detach().cpu().numpy()
    faces = np.concatenate([gcn_net['left'].faces, gcn_net['right'].faces + left_verts.shape[1]], 0)

    for i in range(3):
        np.savetxt('gcn_out/mid%d.xyz' % i, torch.cat([mid_l[i][0, :, :], mid_r[i][0, :, :]]).detach().cpu().numpy())
    trimesh.Trimesh(vertices=verts, faces=faces, process=False,
                           maintain_order=True).export('gcn_out/mesh.obj')

    trimesh.Trimesh(vertices=mano_verts, faces=faces,
                    process=False, maintain_order=True).export('gcn_out/mano.obj')


def run_infer(net, data_path, scan_id):
    keypoints_path = '%s/keypoints3d/keypoints_3d_%d.xyz' % (data_path, scan_id)

    joints = np.loadtxt(keypoints_path)
    if joints.shape[0] != 42:
        import pdb
        pdb.set_trace()
    left_joints = torch.from_numpy(joints[:21]).float().cuda().unsqueeze(0)
    right_joints = torch.from_numpy(joints[21:]).float().cuda().unsqueeze(0)

    # with torch.no_grad():
    left_verts, mano_lverts, mano_lparam, left_Rt, mid = infer(net, left_joints, 'left')

    right_verts, mano_rverts, mano_rparam, right_Rt, mid = infer(net, right_joints, 'right')
    drop_left = left_joints.sum() == 0
    drop_right = right_joints.sum() == 0

    if drop_left and drop_right:
        print('failed')
        return
    elif drop_right:
        verts = left_verts[0].detach().cpu().numpy()
        mano_verts = mano_lverts[0].detach().cpu().numpy()
        faces = net['left'].faces
    elif drop_left:
        verts = right_verts[0].detach().cpu().numpy()
        mano_verts = mano_rverts[0].detach().cpu().numpy()
        faces = net['right'].faces
    else:
        verts = torch.cat([left_verts, right_verts], 1)[0].detach().cpu().numpy()
        mano_verts = torch.cat([mano_lverts, mano_rverts], 1)[0].detach().cpu().numpy()
        faces = np.concatenate([net['left'].faces, net['right'].faces + mano_lverts.shape[1]], 0)

    os.makedirs('%s/gcn_out'% data_path, exist_ok=True)

    trimesh.Trimesh(vertices=mano_verts, faces=faces,
                    process=False, maintain_order=True).export('%s/gcn_out/%d.obj' % (data_path, scan_id))
    trimesh.Trimesh(vertices=verts, faces=faces,
                    process=False, maintain_order=True).export('%s/gcn_out/ori_%d.obj' % (data_path, scan_id))
    params_left, params_right = {'type': 'left'}, {'type': 'right'}
    params_left["pose"] = torch.cat([torch.zeros(1, 3).cuda(), mano_lparam[:, :-10]], 1).detach().cpu()
    params_left["shape"] = mano_lparam[:, -10:].detach().cpu()
    params_left["Rt"] = left_Rt.detach().cpu()
    params_right["pose"] = torch.cat([torch.zeros(1, 3).cuda(), mano_rparam[:, :-10]], 1).detach().cpu()
    params_right["shape"] = mano_rparam[:, -10:].detach().cpu()
    params_right["Rt"] = right_Rt.detach().cpu()
    if drop_right:
        torch.save([params_left], '%s/gcn_out/%d.pt' % (data_path, scan_id))
    elif drop_left:
        torch.save([params_right], '%s/gcn_out/%d.pt' % (data_path, scan_id))
    else:
        torch.save([params_left, params_right], '%s/gcn_out/%d.pt' % (data_path, scan_id))
