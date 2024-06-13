import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from tqdm import tqdm
import argparse
import pickle
from pyhocon import ConfigFactory
import numpy as np
import cv2
import trimesh
import torch
import torch.nn.functional as F
from torch.optim import Adam
import nvdiffrast.torch as dr
from models.utils import get_normals, get_radiance, get_matrix, laplacian_smoothing
from get_data import get_demo_data, get_interhand_data
from repose import lbs_tpose

def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()

def main(conf, scan_id, data_path=None):
    conf = ConfigFactory.parse_file(conf)
    type = conf.get_string('data_type')
    if data_path is not None:
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        out_mesh_dire = out_path + '/' + conf.get_string('out_mesh_dire') + '/%d' % scan_id
        input_mesh_dire = out_path + '/' + conf.get_string('input_mesh_dire')
        os.makedirs(out_mesh_dire, exist_ok=True)
    else:
        data_path = conf.get_string('data_path')
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        out_mesh_dire = './%s_out/%s_%s' % (type, capture_name, data_name)
        input_mesh_dire = out_mesh_dire + '/' + conf.get_string('input_mesh_dire')
        os.makedirs('%s_out' % type, exist_ok=True)
        os.makedirs(out_mesh_dire, exist_ok=True)
        drop_cam = conf.get_string('drop_cam').split(',')

    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    epoch_albedo = conf.get_int('epoch_albedo')
    epoch_sfs = conf.get_int('epoch_sfs')
    resolution = (h, w)
    sfs_weight = conf.get_float('sfs_weight')
    lap_weight = conf.get_float('lap_weight')
    albedo_weight = conf.get_float('albedo_weight')
    mask_weight = conf.get_float('mask_weight')
    edge_weight = conf.get_float('edge_weight')
    delta_weight = conf.get_float('delta_weight')
    degree = conf.get_int('degree')
    batch = conf.get_int('batch')
    lr = conf.get_float('lr')
    albedo_lr = conf.get_float('albedo_lr')
    sh_lr = conf.get_float('sh_lr')

    # vgg_loss = VGGLoss()

    if type == 'demo':
        imgs, grayimgs, masks, w2cs, projs = get_demo_data(data_path, scan_id, num, (w,h))

        mesh = trimesh.load(join(input_mesh_dire, '%d.obj' % scan_id), process=False, maintain_order=True)

        mano_out = torch.load(join(input_mesh_dire, '%d.pt' % scan_id))
        len_v = int(mesh.vertices.shape[0] / len(mano_out))
        len_f = int(mesh.faces.shape[0] / len(mano_out))
        vertices, faces = [], []
        for i in range(len(mano_out)):
            verts_tmp = mesh.vertices[i * len_v: (i + 1) * len_v]
            faces_tmp = mesh.faces[i * len_f: (i + 1) * len_f] - i * len_v

            v, f = trimesh.remesh.subdivide_loop(verts_tmp, faces_tmp, iterations=3)
            f = f + i * v.shape[0]
            vertices.append(torch.from_numpy(v.astype(np.float32)).cuda())
            faces.append(torch.from_numpy(f.astype(np.int32)).cuda())

        vertices = torch.cat(vertices, 0)
        faces = torch.cat(faces, 0)

    elif type == 'interhand':
        split = conf.get_string('split')
        imgs, grayimgs, masks, w2cs, projs, vertices, faces, mano_out = get_interhand_data(
            data_path, scan_id, res=(334, 512), data_name=data_name, capture_name=capture_name, drop_cam=drop_cam,
            split=split)
        num = imgs.shape[0]

        mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy(),
                               process=False, maintain_order=True)
        len_v = int(mesh.vertices.shape[0] / len(mano_out))
        len_f = int(mesh.faces.shape[0] / len(mano_out))
        vertices, faces = [], []
        for i in range(len(mano_out)):
            verts_tmp = mesh.vertices[i * len_v: (i + 1) * len_v]
            faces_tmp = mesh.faces[i * len_f: (i + 1) * len_f] - i * len_v

            v, f = trimesh.remesh.subdivide_loop(verts_tmp, faces_tmp, iterations=3)
            f = f + i * v.shape[0]
            vertices.append(torch.from_numpy(v.astype(np.float32)).cuda())
            faces.append(torch.from_numpy(f.astype(np.int32)).cuda())

        vertices = torch.cat(vertices, 0)
        faces = torch.cat(faces, 0)

    np_verts = vertices.squeeze().detach().cpu().numpy()
    np_faces = faces.squeeze().detach().cpu().numpy()

    mesh = trimesh.Trimesh(np_verts, np_faces, process=False, maintain_order=True)
    mesh.export(join(out_mesh_dire, 'ori_%d.obj' % scan_id))

    glctx = dr.RasterizeGLContext()

    num = imgs.shape[0]

    with torch.no_grad():
        valid_normals = []
        valid_grayimgs = []
        valid_masks = []
        valid_imgs = []
        sh_coeffs = []
        for k in range(0, num, 1):
            n = min(num, k + 1) - k
            w2c = w2cs[k:k + 1]
            proj = projs[k:k + 1]
            mask = masks[k:k + 1]
            img = imgs[k:k + 1]
            grayimg = grayimgs[k:k + 1]

            vertsw = torch.cat([vertices, torch.ones_like(vertices[:, 0:1])], axis=1).unsqueeze(0).expand(n, -1, -1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:, :, :3], faces.long())
            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat, _ = dr.interpolate(torch.cat([normals, torch.ones_like(vertsw[:, :, :1])], 2), rast_out, faces)
            pred_normals = feat[:, :, :, :3].contiguous()
            pred_mask = feat[:, :, :, 3:4].contiguous()
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, faces)
            pred_normals = F.normalize(pred_normals, p=2, dim=3)

            valid_idx = (mask > 0) & (rast_out[:, :, :, 3] > 0)
            valid_normal = pred_normals[valid_idx].detach().cpu().numpy()
            valid_grayimg = grayimg[valid_idx].detach().cpu().numpy()
            matrix = get_matrix(valid_normal, degree)
            sh_coeff = np.linalg.lstsq(matrix, valid_grayimg, rcond=None)[0]
            valid_normals.append(valid_normal)
            valid_imgs.append(img[valid_idx])
            valid_grayimgs.append(grayimg[valid_idx].detach().cpu().numpy())
            valid_masks.append(pred_mask)
            sh_coeffs.append(torch.from_numpy(sh_coeff.astype(np.float32)).cuda().unsqueeze(0))
    valid_normals = np.concatenate(valid_normals, axis=0)
    valid_imgs = torch.cat(valid_imgs, 0)
    valid_grayimgs = np.concatenate(valid_grayimgs, axis=0)
    valid_masks = torch.cat(valid_masks, 0)

    matrix = get_matrix(valid_normals, degree)
    sh_coeff = np.linalg.lstsq(matrix, valid_grayimgs, rcond=None)[0]

    sh_coeff = torch.from_numpy(sh_coeff.astype(np.float32)).cuda()
    sh_coeffs = torch.cat(sh_coeffs, 0)
    tmp_sh_coeffs = sh_coeffs.clone()
    sh_coeffs.requires_grad_(True)

    radiance = get_radiance(sh_coeff, torch.from_numpy(valid_normals).cuda(), degree).unsqueeze(-1)
    albedo_mean = (valid_imgs / radiance).mean(0).detach()

    albedo = (torch.zeros_like(vertices) + albedo_mean).unsqueeze(0)
    albedo.requires_grad_(True)

    delta = torch.zeros_like(vertices)
    delta.requires_grad_(True)

    vertices_tmp = torch.clone(vertices).detach()

    a = vertices[faces[:, 0].long()]
    b = vertices[faces[:, 1].long()]
    c = vertices[faces[:, 2].long()]

    edge_length_mean = torch.cat([((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)]).mean()


    vertices = (vertices_tmp + delta).detach()
    # compute sphere harmonic coefficient as initialization
    optimizer = Adam([{'params': albedo, 'lr': albedo_lr}, {'params': sh_coeffs, 'lr': sh_lr}])

    pbar = tqdm(range(epoch_albedo))
    delta.requires_grad_(False)
    for i in pbar:
        perm = torch.randperm(num).cuda()
        for k in range(0, num, batch):
            n = min(num, k+batch) - k
            w2c = w2cs[perm[k:k+batch]]
            proj = projs[perm[k:k+batch]]
            img = imgs[perm[k:k+batch]]
            mask = masks[perm[k:k+batch]]
            sh_coeff = sh_coeffs[perm[k:k+batch]]

            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:,:,:3], faces.long())

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.cat([normals, albedo.expand(n,-1,-1)], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, faces)
            pred_normals = feat[:,:,:,:3].contiguous()
            rast_albedo = feat[:,:,:,3:6].contiguous()
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, faces)
            pred_normals = F.normalize(pred_normals,p=2,dim=3)
            rast_albedo = dr.antialias(rast_albedo, rast_out, proj_verts, faces)

            valid_idx = torch.where((mask > 0) & (rast_out[:,:,:,3] > 0))
            valid_normals = pred_normals[valid_idx]
            valid_shcoeff = sh_coeff[valid_idx[0]]
            valid_albedo = rast_albedo[valid_idx]

            valid_img = img[valid_idx]
            radiance = get_radiance(valid_shcoeff, valid_normals, degree).unsqueeze(-1)
            pred_img = radiance * valid_albedo

            sfs_loss = sfs_weight * F.l1_loss(pred_img, valid_img)
            albedo_loss = albedo_weight * laplacian_smoothing(albedo.squeeze(0), faces.long(), method="uniform")

            loss = sfs_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            des = 'sfs:%.4f'%sfs_loss.item() + ' albedo:%.4f'%albedo_loss.item()
            pbar.set_description(des)

    delta.requires_grad_(True)
    sh_coeffs.requires_grad_(False)
    optimizer = Adam([{'params': delta, 'lr': lr}, {'params': albedo, 'lr': albedo_lr}, {'params': sh_coeffs, 'lr': sh_lr}])

    pbar = tqdm(range(epoch_sfs))
    rendered_img = []
    for i in pbar:
        perm = torch.randperm(num).cuda()
        if i == epoch_sfs//2:
            albedo_weight = albedo_weight / 10000
        for k in range(0, num, batch):
            vertices = vertices_tmp + delta
            n = min(num, k+batch) - k
            w2c = w2cs[perm[k:k+batch]]
            proj = projs[perm[k:k+batch]]
            img = imgs[perm[k:k+batch]]
            mask = masks[perm[k:k+batch]]
            valid_mask = valid_masks[perm[k:k+batch]]
            sh_coeff = sh_coeffs[perm[k:k+batch]]

            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:,:,:3], faces.long())

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.cat([normals, albedo.expand(n,-1,-1), torch.ones_like(vertsw[:,:,:1])], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, faces)
            pred_normals = feat[:,:,:,:3].contiguous()
            rast_albedo = feat[:,:,:,3:6].contiguous()
            pred_mask = feat[:,:,:,6:7].contiguous()
            pred_normals = F.normalize(pred_normals,p=2,dim=3)
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)

            valid_idx = torch.where((mask > 0) & (rast_out[:,:,:,3] > 0))
            valid_normals = pred_normals[valid_idx]
            valid_shcoeff = sh_coeff[valid_idx[0]]
            valid_albedo = rast_albedo[valid_idx]

            valid_img = img[valid_idx]
            radiance = get_radiance(valid_shcoeff, valid_normals, degree).unsqueeze(-1)
            pred_img = radiance * valid_albedo

            tmp_img = torch.zeros_like(img)
            tmp_img[valid_idx] = pred_img
            tmp_img = dr.antialias(tmp_img, rast_out, proj_verts, faces)

            sfs_loss = sfs_weight * (F.l1_loss(tmp_img[valid_idx], valid_img))


            lap_loss  = lap_weight * laplacian_smoothing(vertices, faces.long(), method="uniform")
            albedo_loss = albedo_weight * laplacian_smoothing(albedo.squeeze(0), faces.long(), method="uniform")

            mask_loss = mask_weight * F.mse_loss(pred_mask, valid_mask)
            a = vertices[faces[:, 0].long()]
            b = vertices[faces[:, 1].long()]
            c = vertices[faces[:, 2].long()]
            edge_length = torch.cat(
                [((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])

            edge_loss = torch.clip(edge_length-edge_length_mean, 0, 1).mean() * edge_weight

            delta_loss = (delta ** 2).sum(1).mean() * delta_weight

            loss = sfs_loss + lap_loss + albedo_loss + mask_loss + delta_loss + edge_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            des = 'sfs:%.4f'%sfs_loss.item() + ' lap:%.4f'%lap_loss.item() + ' albedo:%.4f'%albedo_loss.item() +\
                 ' mask:%.4f'%mask_loss.item() + ' edge:%.4f' % edge_loss.item() + ' delta:%.4f' % delta_loss.item()
            pbar.set_description(des)
            if i == epoch_sfs - 1:
                rendered_img.append(tmp_img)
                perm_last = perm.cpu().numpy()

    del optimizer

    torch.save({'sh_coeff': sh_coeffs, 'albedo': albedo}, join(out_mesh_dire, '%d.pt'%scan_id))

    np_verts = vertices.squeeze().detach().cpu().numpy()
    np_faces = faces.squeeze().detach().cpu().numpy()

    mesh = trimesh.Trimesh(np_verts, np_faces, process=False, maintain_order=True)
    mesh.export(join(out_mesh_dire, '%d.obj' % scan_id))

    normals = get_normals(vertices[None, :, :3], faces.long())[0]
    radiance = get_radiance(sh_coeffs[-1], normals, degree).unsqueeze(-1)

    color = torch.clamp(0.5 * albedo, 0, 1)

    save_obj_mesh_with_color(join(out_mesh_dire, '%d_c.obj' % scan_id), np_verts,
                             np_faces, (color.detach().cpu().numpy()[0])[:, 2::-1])

    os.makedirs(join(out_mesh_dire, 'rerender'), exist_ok=True)

    # save for rerender
    rendered_img = torch.cat(rendered_img, 0)

    for i, idx in enumerate(perm_last):
        cv2.imwrite(join(out_mesh_dire, 'rerender', 'mesh_%02d.png' % idx), (rendered_img[i].detach().cpu().numpy() * 255).astype(np.int32))

    # save for tpose
    with open('mano/mano_weight_sub3.pkl', 'rb') as f:
        pkl = pickle.load(f)

    num = len(mano_out)
    vertices_length = int(np_verts.shape[0]/num)

    for i, mano_para in enumerate(mano_out):
        hand_type = mano_para['type']
        pose = mano_para['pose'].view(1, -1)
        shape = mano_para['shape'].view(1, -1)
        if 'Rt' in mano_para.keys():
            Rt = mano_para['Rt'].view(1, 4, 4)
        else:
            Rt = None
            trans = mano_para['trans'].view(1, -1)
            if 'scale' in mano_para.keys():
                scale = mano_para['scale']
            else:
                scale = 1

        vertices_tmp = np_verts[vertices_length*i: vertices_length*(i+1)]

        faces_tmp = pkl[hand_type]['faces']
        new_weights = pkl[hand_type]['weights']

        vertices_tmp = torch.from_numpy(vertices_tmp.astype(np.float32)).unsqueeze(0)
        if Rt is not None:
            vertices_tmp = torch.matmul(
                torch.cat([vertices_tmp, torch.ones(1, vertices_tmp.shape[1], 1)], 2), torch.linalg.inv(Rt))[:, :, :3]
        else:
            vertices_tmp = (vertices_tmp - trans) / scale

        new_weights = torch.from_numpy(new_weights.astype(np.float32))
        verts_t = lbs_tpose(pose.clone(), shape, new_weights, vertices_tmp, hand_type=hand_type)
        # verts_new = lbs(pose_new.clone(), shape_new, new_weights, verts_t, hand_type=hand_type)
        if Rt is not None:
            verts_t = torch.matmul(
                torch.cat([verts_t, torch.ones(1, verts_t.shape[1], 1)], 2), Rt)[:, :, :3]
        else:
            verts_t = verts_t * scale + trans

        mesh = trimesh.Trimesh(vertices=verts_t.cpu().numpy()[0], faces=faces_tmp, process=False, maintain_order=True)
        mesh.export(os.path.join(out_mesh_dire, '%d_%s_tpose.obj' % (scan_id, hand_type)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/demo_sfs.conf')
    parser.add_argument('--scan_id', type=int, default=6)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    main(args.conf, args.scan_id, args.data_path)