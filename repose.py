import os
import numpy as np
import trimesh
import argparse
import torch
import smplx
from pyhocon import ConfigFactory
from models.smplx import SMPLX, batch_rodrigues, batch_global_rigid_transformation
import pickle
from get_data import get_demo_data, get_interhand_data, mano_layer

# mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True), 'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False)}

def subdivide_weight(weights, faces):
    weight_sub = np.zeros((faces.max()+1, weights.shape[1]))
    weight_sub[:weights.shape[0]] = weights
    for i in range(int(faces.shape[0]/4)):
        tmp = faces[i*4: i*4+4]
        if weight_sub[tmp[0][1]].sum() != 0:
            pass
        weight_sub[tmp[0][1]] = (weight_sub[tmp[0][0]] + weight_sub[tmp[1][1]]) / 2
        weight_sub[tmp[0][2]] = (weight_sub[tmp[0][0]] + weight_sub[tmp[2][2]]) / 2
        weight_sub[tmp[1][2]] = (weight_sub[tmp[1][1]] + weight_sub[tmp[2][2]]) / 2
    return weight_sub

def subdivide_weight_loop(weights, vertices, faces, iterations=3):
    for i in range(iterations):
        vertices, faces = trimesh.remesh.subdivide_loop(vertices, faces, iterations=1)
        weights = subdivide_weight(weights, faces)
    return vertices, faces, weights

def save_sub_weights():
    out = {}
    for hand_type in ['left', 'right']:
        _, faces_tmp, new_weights = subdivide_weight_loop(mano_layer[hand_type].lbs_weights.numpy(),
                                                          mano_layer[hand_type].v_template.numpy(),
                                                          mano_layer[hand_type].faces.astype(np.int64),
                                                          iterations=3)
        out.update({hand_type:{'faces': faces_tmp, 'weights': new_weights}})
    with open('mano/mano_weight_sub3.pkl', 'wb') as f:
        pickle.dump(out, f)

def lbs(pose, shape, weights, verts_tpose, hand_type='right'):
    mano = mano_layer[hand_type]
    b = pose.shape[0]
    device = pose.device
    dtype = shape.dtype
    v_template = mano.v_template
    v_shaped = torch.einsum('bl,mkl->bmk', [shape, mano.shapedirs]) + v_template
    J = torch.einsum('bik,ji->bjk', [v_shaped, mano.J_regressor])

    num_joints = 16

    pose += mano.pose_mean

    R = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, -1, 3, 3)

    # lrotmin = (R[:, 1:, :] - self.e3).reshape(b, -1)
    # v_posed = torch.matmul(lrotmin, smplx.posedirs).reshape(b, smplx.size[0], smplx.size[1]) + v_shaped  # smpl_v_posed

    J_transformed, A = batch_global_rigid_transformation(R, J, mano.parents)

    weights = weights.expand(b, -1, -1)    # [b, num_v, num_j]
    T = torch.matmul(weights, A.reshape(b, num_joints, 16)).reshape(b, -1, 4, 4)  # [b, num_v, 4, 4]

    verts_homo = torch.cat([verts_tpose, torch.ones(b, verts_tpose.shape[1], 1, device=device)], dim=2)
    verts = torch.matmul(T, verts_homo.unsqueeze(-1))

    verts = verts[:, :, :3, 0]
    return verts

def lbs_tpose(pose, shape, weights, verts, hand_type='right'):
    mano = mano_layer[hand_type]
    b = pose.shape[0]
    device = pose.device
    dtype = shape.dtype
    v_template = mano.v_template
    v_shaped = torch.einsum('bl,mkl->bmk', [shape, mano.shapedirs]) + v_template
    J = torch.einsum('bik,ji->bjk', [v_shaped, mano.J_regressor])

    num_joints = 16

    pose += mano.pose_mean

    R = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, -1, 3, 3)

    # lrotmin = (R[:, 1:, :] - self.e3).reshape(b, -1)
    # v_posed = torch.matmul(lrotmin, smplx.posedirs).reshape(b, smplx.size[0], smplx.size[1]) + v_shaped  # smpl_v_posed

    J_transformed, A = batch_global_rigid_transformation(R, J, mano.parents)

    weights = weights.expand(b, -1, -1)  # [b, num_v, num_j]
    T = torch.matmul(weights, A.reshape(b, num_joints, 16)).reshape(b, -1, 4, 4)  # [b, num_v, 4, 4]

    verts_homo = torch.cat([verts, torch.ones(b, verts.shape[1], 1, device=device)], dim=2)
    verts_tpose = torch.matmul(torch.linalg.inv(T), verts_homo.unsqueeze(-1))

    verts_tpose = verts_tpose[:, :, :3, 0]
    return verts_tpose

def forward_mano_with_Rt(hand_type, mano_pose, mano_shape, mano_Rt):
    batch = mano_pose.shape[0]

    if mano_pose.shape[1] == 48:
        global_orient = mano_pose[:, :3]
        hand_pose = mano_pose[:, 3:]
    else:
        global_orient = torch.zeros(batch, 3).to(mano_pose.device)
        hand_pose = mano_pose
    outputs = mano_layer[hand_type](global_orient=global_orient,
                                              hand_pose=hand_pose,
                                              betas=mano_shape.to(mano_pose.device))
    mano_vertices = outputs.vertices
    mano_verts = torch.matmul(
        torch.cat([mano_vertices, torch.ones(batch, mano_vertices.shape[1], 1).to(mano_pose.device)], 2), mano_Rt.to(mano_pose.device))[:, :, :3]
    vertices = mano_verts.to(mano_pose.device)
    faces = torch.from_numpy(mano_layer[hand_type].faces.astype(np.int32)).to(mano_pose.device)
    return vertices, faces

def forward_mano_with_trans(hand_type, mano_pose, mano_shape, trans, scale):
    batch = mano_pose.shape[0]
    if mano_pose.shape[1] == 51:
        global_orient = mano_pose[:, :3]
        hand_pose = mano_pose[:, 3:]
    else:
        global_orient = torch.zeros(batch, 3).to(mano_pose.device)
        hand_pose = mano_pose
    outputs = mano_layer[hand_type](global_orient=global_orient,
                                              hand_pose=hand_pose,
                                              betas=mano_shape.to(mano_pose.device))
    mano_vertices = outputs.vertices
    mano_verts = mano_vertices * scale + trans
    vertices = mano_verts.to(mano_pose.device)
    faces = torch.from_numpy(mano_layer[hand_type].faces.astype(np.int32)).to(mano_pose.device)
    return vertices, faces

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/demo_sfs.conf')
    parser.add_argument('--scan_id', type=int, default=6)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.conf)
    scan_id = args.scan_id
    data_path = args.data_path
    data_type = conf.get_string('data_type')
    if data_path is not None:
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        out_mesh_dire = out_path + '/' + conf.get_string('out_mesh_dire')
        input_mesh_dire = out_path + '/' + conf.get_string('input_mesh_dire')
        os.makedirs(out_mesh_dire, exist_ok=True)
    else:
        data_path = conf.get_string('data_path')
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        out_mesh_dire = './%s_out/%s_%s' % (data_type, capture_name, data_name)
        os.makedirs('%s_out' % data_type, exist_ok=True)
        os.makedirs(out_mesh_dire, exist_ok=True)
        drop_cam = conf.get_string('drop_cam').split(',')

    if data_type == 'demo':
        imgs, grayimgs, masks, w2cs, projs = get_demo_data(data_path, scan_id, num, (w,h))
        mesh = trimesh.load(join(input_mesh_dire, '%d.obj' % scan_id), process=False, maintain_order=True)
        vertices, faces = trimesh.remesh.subdivide_loop(mesh.vertices, mesh.faces, iterations=3)
        vertices = torch.from_numpy(vertices.astype(np.float32))
        faces = torch.from_numpy(faces.astype(np.int32))
    elif data_type == 'interhand':
        imgs, grayimgs, masks, w2cs, projs, vertices, faces, mano_out = get_interhand_data(
            '/hdd/data1/gqj/datasets/InterHand2.6M/InterHand2.6M_5fps_batch1', scan_id, res=(334, 512),
            data_name=data_name, capture_name=capture_name, drop_cam=drop_cam)
        detailed_mesh = trimesh.load('%s/%d.obj' % (out_mesh_dire, scan_id), process=False, maintain_order=True)
        vertices, faces = detailed_mesh.vertices, detailed_mesh.faces
    if not os.path.exists('mano/mano_weight_sub3.pkl'):
        save_sub_weights()
    with open('mano/mano_weight_sub3.pkl', 'rb') as f:
        pkl = pickle.load(f)

    num = len(mano_out)
    vertices_length = int(vertices.shape[0]/num)

    for i, mano_para in enumerate(mano_out):
        hand_type = mano_para['type']
        pose = mano_para['pose'].view(1, -1)
        shape = mano_para['shape'].view(1, -1)
        trans = mano_para['trans'].view(1, -1)
        vertices_tmp = vertices[vertices_length*i: vertices_length*(i+1)]

        faces_tmp = pkl[hand_type]['faces']
        new_weights = pkl[hand_type]['weights']

        mesh = trimesh.Trimesh(vertices=vertices_tmp, faces=faces_tmp, process=False, maintain_order=True)
        mesh.export(os.path.join(out_mesh_dire, '%s_oo.obj' % hand_type))

        vertices_tmp = torch.from_numpy(vertices_tmp.astype(np.float32)).unsqueeze(0)
        vertices_tmp = vertices_tmp - trans
        faces_tmp = torch.from_numpy(faces_tmp.astype(np.int32))
        new_weights = torch.from_numpy(new_weights.astype(np.float32))
        verts_t = lbs_tpose(pose.clone(), shape, new_weights, vertices_tmp, hand_type=hand_type)
        # verts_new = lbs(pose_new.clone(), shape_new, new_weights, verts_t, hand_type=hand_type)
        verts_t = verts_t + trans

        mesh = trimesh.Trimesh(vertices=verts_t.cpu().numpy()[0], faces=faces_tmp,process=False, maintain_order=True)
        mesh.export(os.path.join(out_mesh_dire, '%s_tpose.obj' % hand_type))