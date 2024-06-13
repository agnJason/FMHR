import cv2
import argparse
import os.path
from os.path import join
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import nvdiffrast.torch as dr
import smplx
mano_layer = {'right': smplx.create('./', 'mano', use_pca=True, is_rhand=True).cuda(), 'left': smplx.create('./', 'mano', use_pca=True, is_rhand=False).cuda()}
from models.utils import load_K_Rt_from_P

def get_edges(verts, faces):
    V = verts.shape[0]
    f = faces.shape[0]
    device = verts.device
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
    edges, _ = edges.sort(dim=1)
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool, device=device)
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    edges = torch.stack([u // V, u % V], dim=1)

    faces_to_edges = inverse_idxs.reshape(3, f).t()

    return edges, faces_to_edges


def laplacian_cot(verts, faces):
    '''
    verts n 3
    faces f 3
    '''
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]  # f 3 3
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot = cot / 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]

    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    L = L + L.t()

    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas


def compute_laplacian(verts, faces):
    # first compute edges

    V = verts.shape[0]
    device = verts.device

    edges, faces_to_edges = get_edges(verts, faces)

    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def laplacian_smoothing(verts, faces, method="uniform"):
    weights = 1.0 / verts.shape[0]

    with torch.no_grad():
        if method == "uniform":
            L = compute_laplacian(verts, faces)
        else:
            L, inv_areas = laplacian_cot(verts, faces)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas

    if method == "uniform":
        loss = L.mm(verts)
    elif method == "cot":
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = (L.mm(verts) - L_sum * verts) * norm_w

    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum()

# Select number of principal components for pose space
ncomps = 6

def create_param(params, batch_size, k3d):
    params["pose"] = torch.zeros(batch_size, ncomps + 3).cuda()
    params["shape"] = torch.zeros(batch_size, 10).cuda()
    params["trans"] = k3d[:, 0] - torch.tensor([[-0.0680,  0.0061,  0.0052]]).cuda()
    params["scale"] = torch.ones(1).cuda() * 2
    params["pose"].requires_grad = True
    params["shape"].requires_grad = True
    params["trans"].requires_grad = True
    params["scale"].requires_grad = True
    return params

def get_demo_data(data_path, scan_id, num, res=(1280,1024)):
    camera_dict = np.load(join(data_path, '%d/camera/param.npz' % scan_id))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    imgs = []
    grayimgs = []
    masks = []
    w2cs = []
    projs = []
    Pall = []
    # scale_mat = torch.from_numpy(scale_mats[0].astype(np.float32)).cuda()
    for i in range(num):
        P = world_mats[i] @ scale_mats[i]
        Pall.append(torch.from_numpy(P[:3]).float().cuda())
        proj, w2c = load_K_Rt_from_P(P[:3])

        proj[0,0] = proj[0,0] / (res[0] / 2.)
        proj[0,2] = proj[0,2] / (res[0] / 2.) - 1.
        proj[1,1] = proj[1,1] / (res[1] / 2.)
        proj[1,2] = proj[1,2] / (res[1] / 2.) - 1.
        proj[2,2] = 0.
        proj[2,3] = -0.1
        proj[3,2] = 1.
        proj[3,3] = 0.

        projs.append(proj.astype(np.float32))
        w2cs.append(w2c.astype(np.float32))
        img = cv2.imread(join(data_path, '%d/img/%02d.png'%(scan_id, i)))
        mask = cv2.imread(join(data_path, '%d/mask/%02d.png'%(scan_id, i)))[:,:,0]
        mask = (mask>127.5).astype(np.float32)
        img[mask==0] = 0
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, res)
        grayimg = cv2.resize(grayimg, res)
        mask = cv2.resize(mask, res, interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy((img/255.)).float().cuda()
        grayimg = torch.from_numpy((grayimg/255.)).float().cuda()
        mask = torch.from_numpy(mask).float().cuda()

        imgs.append(img)
        grayimgs.append(grayimg)
        masks.append(mask)

    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0,2,1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0,2,1).cuda()
    imgs = torch.stack(imgs, dim=0)
    grayimgs = torch.stack(grayimgs, dim=0)
    masks = torch.stack(masks, dim=0)

    return imgs, grayimgs, masks, w2cs, projs

def mano_forward(pose, shape, trans, scale, type):
    output = mano_layer[type](global_orient=pose[:, :3],
                                    hand_pose=pose[:, 3:],
                                    betas=shape)

    if type == 'right':
        tips = output.vertices[:, [745, 317, 444, 556, 673]]
    else:
        tips = output.vertices[:, [745, 317, 445, 556, 673]]

    joints = torch.cat([output.joints, tips], 1)

    # Reorder joints to match visualization utilities
    joints = joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    joints = joints * scale + trans.unsqueeze(1)
    verts = output.vertices * scale + trans.unsqueeze(1)

    return verts, joints

def mano_optimize(k3d):
    batch_size = 1

    params_right = {'type': 'right'}
    params_right = create_param(params_right, batch_size, k3d[:, 21:])
    params_left = {'type': 'left'}
    params_left = create_param(params_left, batch_size, k3d[:, :21])

    optim_params = [{'params': params_right["pose"], 'lr': 0.05},
                    {'params': params_right["shape"], 'lr': 0.01},
                    {'params': params_right["trans"], 'lr': 0.01},
                    {'params': params_right["scale"], 'lr': 0.01},
                    {'params': params_left["pose"], 'lr': 0.05},
                    {'params': params_left["shape"], 'lr': 0.01},
                    {'params': params_left["trans"], 'lr': 0.01},
                    {'params': params_left["scale"], 'lr': 0.01},
                    ]

    # set_zeros = torch.ones_like(params["pose_params"])
    # set_zeros[:, 21:27] = 0
    # set_zeros[:, 22] = 1
    # set_zeros[:, 25] = 1
    # set_zeros[:, 30:36] = 0

    optimizer = Adam(optim_params)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    epoch = 250
    pbar = tqdm(range(epoch))
    for i in pbar:
        verts_r, joints_r = mano_forward(params_right["pose"], params_right["shape"],
                                         params_right["trans"], params_right["scale"], 'right')
        verts_l, joints_l = mano_forward(params_left["pose"], params_left["shape"],
                                         params_left["trans"], params_left["scale"], 'left')

        joints = torch.cat([joints_l, joints_r], 1)

        loss = 40 * F.mse_loss(joints, k3d) + F.l1_loss(params_left["shape"], params_right["shape"]) * 0.1
        if False:  # ！！not useful
            n = num
            w2c = w2cs
            proj = projs
            mask = masks
            verts = torch.cat([verts_l, verts_r], 1)
            vertsw = torch.cat([verts, torch.ones_like(verts[:, :, 0:1])], axis=2).expand(n, -1, -1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.ones_like(vertsw[:, :, :1])
            feat, _ = dr.interpolate(feat, rast_out, faces)
            pred_mask = feat[:, :, :, :1].contiguous()
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)
            mask_loss = F.mse_loss(pred_mask, mask)
        else:
            mask_loss = torch.tensor(0).cuda()
        loss_total = loss + mask_loss
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        # scheduler.step()
        des = 'loss:%.4f; mask:%.4f' % (loss.item(), mask_loss.item())
        pbar.set_description(des)
    return params_left, params_right, verts_l, verts_r

def main(scan_id, data_path, out_path, use_last_param=False):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path + '/mano_out', exist_ok=True)

    # num = conf.get_int('num')
    # w = conf.get_int('w')
    # h = conf.get_int('h')
    # resolution = (h, w)
    num = 16
    w = 1280
    h = 1024
    resolution = (h, w)
    # imgs, grayimgs, masks, w2cs, projs = get_demo_data(data_path, scan_id, num, (w,h))
    # glctx = dr.RasterizeGLContext()

    k3d = np.loadtxt('%s/keypoints3d/keypoints_3d_%d.xyz' % (out_path, scan_id))
    k3d = torch.from_numpy(k3d).float().unsqueeze(0).cuda()
    # k3d = torch.cat([k3d[:, :19], k3d[:, 20:21], k3d[:, 23:24], k3d[:, 25:]], 1)

    # gt_mesh = trimesh.load('%s/demo_sfs/%d.obj'% (input_dire, scan_id), process=False, maintain_order=True)
    # gt_verts = torch.from_numpy(gt_mesh.vertices.astype(np.float32)).unsqueeze(0).cuda()
    # gt_verts[:, :, :2] = gt_verts[:, :, :2] * -1
    # gt_faces = torch.from_numpy(gt_mesh.faces.astype(np.int64)).cuda().long()
    #
    # face_vertices = index_vertices_by_faces(gt_verts, gt_faces)
    faces = torch.cat([torch.from_numpy(mano_layer['left'].faces.astype(np.int32)).int(),
                       torch.from_numpy(mano_layer['right'].faces.astype(np.int32)).int() +
                       mano_layer['right'].v_template.shape[0]], 0).int().cuda()

    params_left, params_right, verts_l, verts_r = mano_optimize(k3d)

    with torch.no_grad():
        # params["pose_params"] = params["pose_params"].detach()
        # params["pose_params"][:, 21:27] = 0
        # params["pose_params"][:, 30:36] = 0
        output_r = mano_layer['right'](global_orient=params_right["pose"][:, :3],
                                  hand_pose=params_right["pose"][:, 3:],
                                  betas=params_right["shape"])
        output_l = mano_layer['left'](global_orient=params_left["pose"][:, :3],
                                  hand_pose=params_left["pose"][:, 3:],
                                  betas=params_left["shape"])

    mano_verts = torch.cat([verts_l, verts_r], 1)[0].detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=mano_verts, faces=faces.detach().cpu().numpy())
    mesh.export('%s/mano_out/%d.obj' % (out_path, scan_id))
    params_right["pose"] = torch.cat([output_r.global_orient, output_r.hand_pose], 1).detach().cpu()
    params_right["shape"] = output_r.betas.detach().cpu()
    params_right["trans"] = params_right["trans"].detach().cpu()
    params_right["scale"] = params_right["scale"].detach().cpu()
    params_left["pose"] = torch.cat([output_l.global_orient, output_l.hand_pose], 1).detach().cpu()
    params_left["shape"] = output_l.betas.detach().cpu()
    params_left["trans"] = params_left["trans"].detach().cpu()
    params_left["scale"] = params_left["scale"].detach().cpu()
    torch.save([params_left, params_right], '%s/mano_out/%d.pt' % (out_path, scan_id))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_id', type=int, default=6)
    parser.add_argument('--range', type=int)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    args = parser.parse_args()

    if args.range is not None:
        for i in range(1, args.range + 1):
             main(i, args.data_path, args.out_path, True)
    else:
        main(args.scan_id, args.data_path, args.out_path)