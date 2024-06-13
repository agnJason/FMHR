import os
import time
import math
from os.path import join
import json
from tqdm import tqdm
import argparse
import pickle
from pyhocon import ConfigFactory
import numpy as np
import cv2
import trimesh
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import lpips
from skimage.metrics import structural_similarity as SSIM
from models.utils import load_K_Rt_from_P, laplacian_smoothing
import torchvision
from torchvision import transforms
from models.unet import UNet
from models.PostionalEncoding import PostionalEncoding
from models.get_rays import get_ray_directions, get_rays
from get_data import get_demo_data, get_interhand_data
from train_mlp import mlp_forward, MLP
from train_unet import unet_forward
from repose import lbs_tpose, lbs, forward_mano_with_Rt, forward_mano_with_trans

loss_fn_alex = lpips.LPIPS(net='alex', version=0.1).cuda()
transf = transforms.ToTensor()

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [8， 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)*(img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 9:
        return float('inf')
    return 28 * math.log10(1.0 / math.sqrt(mse))

def train(conf, scan_id, data_path=None, continue_train=False, net_type='mlp', split_data=False, data_name=None):
    conf = ConfigFactory.parse_file(conf)
    type = conf.get_string('data_type')
    if data_path is not None:
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        input_mesh_dire = out_path + '/' + conf.get_string('input_mesh_dire')
    else:
        data_path = conf.get_string('data_path')
        if data_name is None:
            data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        input_mesh_dire = './%s_out/%s_%s' % (type, capture_name, data_name)
        os.makedirs('%s_out' % type, exist_ok=True)
        drop_cam = conf.get_string('drop_cam').split(',')
    os.makedirs(join(input_mesh_dire, 'checkpoints'), exist_ok=True)
    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)


    if net_type == 'mlp':
        net = MLP(284, 3).cuda()
        net_g = MLP(12, 3).cuda()
        forward = mlp_forward
    elif net_type == 'unet':
        net = UNet(284, 3, 2, 0).cuda()
        net_g = UNet(12, 3, 2, 0).cuda()
        forward = unet_forward

    pe = PostionalEncoding(min_deg=0, max_deg=1, scale=0.1)
    glctx = dr.RasterizeGLContext()

    if type == 'demo':
        batch = 2
        imgs, grayimgs, gt_masks, w2cs, projs, rays = get_demo_data(data_path, scan_id, num, (w, h), return_ray=True, use_mask=True)
        num_epoch = 200
    elif type == 'interhand':
        batch = 8
        num_epoch = 100
        split_data = False
        split = conf.get_string('split')
        imgs, grayimgs, gt_masks, w2cs, projs, vertices, faces, mano_out, rays = get_interhand_data(
            data_path, scan_id, res=(334, 512), data_name=data_name, use_sam=False,
            capture_name=capture_name, drop_cam=drop_cam, return_ray=True, split=split, test_num=10)
    imgs = imgs.flip(3)

    print('load obj:', join(input_mesh_dire, '%d.obj' % scan_id))
    mesh = trimesh.load(join(input_mesh_dire, '%d.obj' % scan_id), process=False, maintain_order=True)
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces.astype(np.int32)).int().cuda()
    pt = torch.load(join(input_mesh_dire, '%d.pt' % scan_id))
    # sh_coeffs = pt['sh_coeff'].cuda()

    albedo = pt['albedo'].cuda()
    num = imgs.shape[0]
    print(f'use {num} views')

    if continue_train:
        pt = torch.load('checkpoints/%s_20.pth' % net_type)
        vertex_feat = pt['vertex_feat'].cuda()
        if vertices.shape[0] > vertex_feat.shape[0]:
            vertex_feat = torch.cat([vertex_feat, vertex_feat], 0)
        net.load_state_dict(pt['model'])
    else:
        vertex_feat = torch.zeros(vertices.shape[0], 20).cuda()
    # vertex_feat = torch.zeros(vertices.shape[0], 20).cuda()
    vertex_feat.requires_grad_(True)
    vertices.requires_grad_(False)
    albedo.requires_grad_(False)
    vertices_tmp = vertices.clone().detach()
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.0001},
                                  {'params': vertices, 'lr': 0.0005},
                                  {'params': albedo, 'lr': 0.0005},
                                  {'params': vertex_feat, 'lr': 0.0001}])

    a = vertices[faces[:, 0].long()]
    b = vertices[faces[:, 1].long()]
    c = vertices[faces[:, 2].long()]

    edge_length_mean = torch.cat([((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)]).mean().detach()

    if split_data:
        val_num = 10
        num = num - val_num
        val_w2cs = w2cs[-val_num:]
        val_projs = projs[-val_num:]
        val_imgs = imgs[-val_num:]
        val_rays = rays[-val_num:]
        w2cs = w2cs[:-val_num]
        projs = projs[:-val_num]
        imgs = imgs[:-val_num]
        rays = rays[:-val_num]

    pbar = tqdm(range(num_epoch))
    for epoch in pbar:
        net.train()
        perm = torch.randperm(num).cuda()
        rerenders = []
        maskss = []
        for k in range(0, num, batch):
            n = min(num, k + batch) - k
            w2c = w2cs[perm[k:k + batch]]
            proj = projs[perm[k:k + batch]]
            img = imgs[perm[k:k + batch]]
            gt_mask = gt_masks[perm[k:k + batch]]
            ray = rays[perm[k:k + batch]]
            render_imgs, masks = forward(net, pe, glctx, [ray, w2c, proj, vertices.unsqueeze(0).expand(n, -1, -1),
                                                          faces, albedo.expand(n, -1, -1), vertex_feat], resolution)
            valid_index = (masks[:, :, :, 0] > 0) & (gt_mask > 0)
            img_loss = 100 * F.smooth_l1_loss(render_imgs[valid_index], img[valid_index])

            loss = img_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description('epoch: %d, loss: %f' % (epoch, img_loss.item()))
            rerenders.append(render_imgs)
            maskss.append(masks)
        torch.cuda.empty_cache()
        rerenders = torch.cat(rerenders, 0)
        maskss = torch.cat(maskss, 0)
    torch.save({'model': net.state_dict(), 'vertex_feature': vertex_feat.detach().cpu()},
               join(input_mesh_dire, 'checkpoints', '%s_%d_%d.pth' % (net_type, scan_id, epoch + 1)))
    vertex_feat.requires_grad_(True)
    vertices.requires_grad_(True)
    albedo.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.0001},
                                  {'params': net_g.parameters(), 'lr': 0.0005},
                                  {'params': vertices, 'lr': 0.0005},
                                  {'params': albedo, 'lr': 0.0005},
                                  {'params': vertex_feat, 'lr': 0.0001}])

    pbar = tqdm(range(num_epoch))
    for epoch in pbar:
        net_g.train()
        perm = torch.randperm(num).cuda()
        rerenders = []
        maskss = []
        for k in range(0, num, batch):
            n = min(num, k + batch) - k
            w2c = w2cs[perm[k:k + batch]]
            proj = projs[perm[k:k + batch]]
            img = imgs[perm[k:k + batch]]
            gt_mask = gt_masks[perm[k:k + batch]]
            ray = rays[perm[k:k + batch]]
            render_z, masks = forward(net, pe, glctx, [ray, w2c, proj, vertices.unsqueeze(0).expand(n, -1, -1),
                                                          faces, albedo.expand(n, -1, -1), vertex_feat], resolution)

            render_imgs, masks = forward(net_g, pe, glctx, [ray, w2c, proj, vertices.unsqueeze(0).expand(n, -1, -1),
                                                          faces, albedo.expand(n, -1, -1), render_z.detach()], resolution, True)
            valid_index = (masks[:, :, :, 0] > 0) & (gt_mask > 0)
            img_loss = F.l1_loss(render_imgs[valid_index], img[valid_index])
            imgz_loss = F.l1_loss(render_z[valid_index], img[valid_index])
            lap_loss = 100 * laplacian_smoothing(vertices, faces.long(), method="uniform")
            mask_loss = F.l1_loss(masks[:, :, :, 0], gt_mask) * 0

            a = vertices[faces[:, 0].long()]
            b = vertices[faces[:, 1].long()]
            c = vertices[faces[:, 2].long()]
            edge_length = torch.cat(
                [((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])

            edge_loss = torch.clip(edge_length - edge_length_mean, 0, 1).mean() * 150000
            delta_loss = ((vertices_tmp - vertices) ** 2).sum(1).mean() * 50000
            loss = img_loss + imgz_loss + lap_loss + mask_loss + edge_loss + delta_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description('epoch: %d, loss: %f, lossz: %f, lap: %f, mask: %f, edge：%f, delta: %f' % (
            epoch, img_loss.item(), imgz_loss.item(), lap_loss.item(), mask_loss.item(), edge_loss.item(), delta_loss.item()))
            rerenders.append(render_imgs)
            maskss.append(masks)

    torch.save({'model': net.state_dict(), 'vertex_feature': vertex_feat.detach().cpu()},
               join(input_mesh_dire, 'checkpoints', '%s_%d_%d.pth' % (net_type, scan_id, epoch + 1)))
    trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=mesh.faces, process=False, maintain_order=True).\
        export(join(input_mesh_dire, 'final_%d.obj' % scan_id))

def eval(conf, scan_id, data_path=None, net_type='mlp'):
    conf = ConfigFactory.parse_file(conf)
    type = conf.get_string('data_type')
    if data_path is not None:
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        input_mesh_dire = out_path + '/' + conf.get_string('input_mesh_dire') + '/' + str(scan_id)
    else:
        data_path = conf.get_string('data_path')
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        input_mesh_dire = './%s_out/%s_%s' % (type, capture_name, data_name)
        os.makedirs('%s_out' % type, exist_ok=True)
        drop_cam = conf.get_string('drop_cam').split(',')
    os.makedirs(join(input_mesh_dire, 'checkpoints'), exist_ok=True)
    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)
    batch = 32

    if net_type == 'mlp':
        net = MLP(284, 3).cuda()
        forward = mlp_forward
    elif net_type == 'unet':
        net = UNet(284, 3, 2, 0).cuda()
        forward = unet_forward

    pe = PostionalEncoding(min_deg=0, max_deg=1, scale=0.1)
    glctx = dr.RasterizeGLContext()

    if type == 'demo':
        imgs, grayimgs, masks, w2cs, projs, rays = get_demo_data(data_path, scan_id, num, (w, h), return_ray=True, use_mask=False)
        img_name = str(scan_id).zfill(4) + '0000'
        camera_names = list(range(num))
        num_epoch = 200
    elif type == 'interhand':
        split = conf.get_string('split')
        imgs, w2cs, projs, rays, camera_names, img_name = get_valinterhand_data(
            data_path, scan_id, res=(334, 512), data_name=data_name, split=split,
            capture_name=capture_name, drop_cam=drop_cam, return_ray=True)
        num_epoch = 100
        # imgs, grayimgs, masks, w2cs, projs, vertices, faces, mano_out, rays = get_interhand_data(
        #     data_path, scan_id, res=(334, 512), data_name=data_name,
        #     capture_name=capture_name, drop_cam=drop_cam, return_ray=True)
    imgs = imgs.flip(3)
    print('load obj:', join(input_mesh_dire, '%d.obj' % scan_id))
    mesh = trimesh.load(join(input_mesh_dire, '%d.obj' % scan_id), process=False, maintain_order=True)

    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces.astype(np.int32)).int().cuda()
    pt = torch.load(join(input_mesh_dire, '%d.pt' % scan_id))
    albedo = pt['albedo'].cuda()

    num = w2cs.shape[0]

    pth = torch.load(join(input_mesh_dire, 'checkpoints', '%s_%d_%d.pth' % (net_type, scan_id, num_epoch)))
    net.load_state_dict(pth['model'])
    vertex_feat = pth['vertex_feature'].cuda()
    batch = 1
    infer_time = 0
    psnr_sum = 0
    ssim_sum = 0
    lpips = 0

    os.makedirs(input_mesh_dire + '/rerender', exist_ok=True)
    with torch.no_grad():
        for k in range(0, num, batch):
            n = min(num, k + batch) - k
            img = imgs[k:k + batch]
            w2c = w2cs[k:k + batch]
            proj = projs[k:k + batch]
            ray = rays[k:k + batch]
            start_time = time.time()
            render_imgs, masks = forward(net, pe, glctx, [ray, w2c, proj, vertices.unsqueeze(0).expand(n, -1, -1),
                                                      faces, albedo.expand(n, -1, -1), vertex_feat], resolution)
            infer_time += (time.time() - start_time)

            render_imgs = render_imgs * masks
            idx = 0
            gt = (img[idx] * masks[idx]).detach().cpu().numpy()
            pred = render_imgs[idx].detach().cpu().numpy()
            psnr = calculate_psnr(gt, pred, masks[idx].detach().cpu().numpy())  # PSNR(gt, pred)

            ssim = SSIM(gt, pred, channel_axis=2, data_range=1)

            lpips_loss = loss_fn_alex(render_imgs[idx].unsqueeze(0).permute(0, 3, 1, 2), (img[idx] * masks[idx]).unsqueeze(0).permute(0, 3, 1, 2))
            psnr_sum += psnr
            ssim_sum += ssim
            lpips += lpips_loss.item()

            total_psnr.append(psnr)
            total_ssim.append(ssim)
            total_LPIPS.append(lpips_loss.item())

            torchvision.utils.save_image(render_imgs[idx].permute(2, 0, 1), input_mesh_dire + '/rerender/nr_%s_%s.png' % (img_name[:-4], camera_names[k]))
    print("render ave: ", f"PSNR value is {psnr_sum / num} dB", f"SSIM value is {ssim_sum / num}", f"LPIPS value is {lpips / num}")
    print('render fps:', 1 / (infer_time/num))


def eval_repose(conf, scan_id, data_path=None, net_type='mlp'):
    conf = ConfigFactory.parse_file(conf)
    type = conf.get_string('data_type')
    if data_path is not None:
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        input_pt_dire = out_path + '/' + conf.get_string('input_pt_dire')
        input_mesh_dire = out_path + '/' + conf.get_string('input_mesh_dire') + '/' + str(scan_id)
    else:
        data_path = conf.get_string('data_path')
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        input_mesh_dire = './%s_out/%s_%s' % (type, capture_name, data_name)
        os.makedirs('%s_out' % type, exist_ok=True)
        drop_cam = conf.get_string('drop_cam').split(',')
    os.makedirs(join(input_mesh_dire, 'repose'), exist_ok=True)
    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)

    batch = 32

    if net_type == 'mlp':
        net = MLP(284, 3).cuda()
        forward = mlp_forward
    elif net_type == 'unet':
        net = UNet(284, 3, 2, 0).cuda()
        forward = unet_forward

    pe = PostionalEncoding(min_deg=0, max_deg=1, scale=0.1)
    glctx = dr.RasterizeGLContext()

    test_scan_id = 50
    if type == 'demo':
        imgs, grayimgs, masks, w2cs, projs, rays = get_demo_data(data_path, scan_id, num, (w, h), return_ray=True,
                                                                 use_mask=False)
        mano_out_last = torch.load(join(input_pt_dire, '%d_refined.pt' % scan_id))
        ori_name = str(scan_id)
        num_epoch = 200
    elif type == 'dhm':
        imgs, grayimgs, masks, w2cs, projs, rays = get_dhm_data(data_path, scan_id, (w, h),
                                                                data_name=data_name, capture_name=capture_name,
                                                                return_ray=True)
    elif type == 'interhand':
        split = conf.get_string('split')
        _, _, _, _, _, _, _, mano_out_last, _ = get_interhand_data(
            data_path, scan_id, res=(334, 512), data_name=data_name, use_sam=False,
            capture_name=capture_name, drop_cam=drop_cam, return_ray=True, split=split, check_gcn=False, test_num=10, load_refined=True)
        ori_name = sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name, 'cam400262')))[scan_id][:-4]
        num_epoch = 100
        print('now name:', ori_name)
    print('load obj:', join(input_mesh_dire, '%d.obj' % scan_id))
    mesh = trimesh.load(join(input_mesh_dire, '%d.obj' % scan_id), process=False, maintain_order=True)

    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces.astype(np.int32)).int().cuda()
    pt = torch.load(join(input_mesh_dire, '%d.pt' % scan_id))
    albedo = pt['albedo'].cuda()
    pth = torch.load(join(input_mesh_dire, 'checkpoints', '%s_%d_%d.pth' % (net_type, scan_id, num_epoch)))
    net.load_state_dict(pth['model'])
    vertex_feat = pth['vertex_feature'].cuda()
    infer_time = 0
    psnr_sum = 0
    ssim_sum = 0
    lpips = 0
    for scan_id in range(test_scan_id):
        if type == 'demo':
            imgs, grayimgs, gt_masks, w2cs, projs, rays = get_demo_data(data_path, scan_id, num, (w, h),
                                                                     return_ray=True,
                                                                     use_mask=False)
            mano_out = torch.load(join(input_pt_dire, '%d_refined.pt' % scan_id))
            img_name = str(scan_id)
        elif type == 'interhand':
            imgs, grayimgs, gt_masks, w2cs, projs, _, _, mano_out, rays = get_interhand_data(
            data_path, scan_id, res=(334, 512), data_name=data_name, use_sam=False,
            capture_name=capture_name, drop_cam=drop_cam, return_ray=True, split=split, use_mask=False, check_gcn=False,
            test_num=10, load_refined=True)
            img_name = sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name, 'cam400262')))[scan_id][:-4]

        imgs = imgs.flip(3)
        # print('repose to name:', img_name)

        with open('mano/mano_weight_sub3.pkl', 'rb') as f:
            pkl = pickle.load(f)


        batch = 1
        vertices = []
        num_hand = len(mano_out_last)
        vertices_length = int(mesh.vertices.shape[0] / num_hand)

        cam_idx = 5
        img = imgs[cam_idx:cam_idx+1]
        w2c = w2cs[cam_idx:cam_idx+1]
        proj = projs[cam_idx:cam_idx+1]
        ray = rays[cam_idx:cam_idx+1]

        for i, mano_para in enumerate(mano_out_last):
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

            vertices_tmp = mesh.vertices[vertices_length*i: vertices_length*(i+1)]

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
            verts_new = lbs(mano_out[i]['pose'].view(1, -1).clone(), shape, new_weights, verts_t, hand_type=hand_type)
            if 'Rt' in mano_out[i].keys():
                Rt = mano_out[i]['Rt'].view(1, 4, 4)
            else:
                Rt = None

            if Rt is not None:
                verts_new = torch.matmul(
                    torch.cat([verts_new, torch.ones(1, verts_new.shape[1], 1)], 2), Rt)[:, :, :3]
                vertices.append(verts_new[0])
            else:
                if 'scale' in mano_out[i].keys():
                    scale_new = mano_out[i]['scale']
                else:
                    scale_new = 1
                vertices.append(verts_new[0] * scale_new + mano_out[i]['trans'].view(1, -1))
        vertices = torch.cat(vertices).float().cuda()

        trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=mesh.faces).export(input_mesh_dire+'/repose/%s_%s.obj' % (ori_name, img_name))
        with torch.no_grad():
            for k in range(0, 1):
                n = 1
                start_time = time.time()

                render_imgs, masks = forward(net, pe, glctx, [ray, w2c, proj, vertices.unsqueeze(0).expand(n, -1, -1),
                                                              faces, albedo.expand(n, -1, -1), vertex_feat], resolution)

                infer_time += (time.time() - start_time)

                render_imgs = render_imgs * masks
                idx = 0
                gt = (img[idx] * masks[idx]).detach().cpu().numpy()
                pred = render_imgs[idx].detach().cpu().numpy()
                psnr = calculate_psnr(gt, pred, masks[idx].detach().cpu().numpy())  # PSNR(gt, pred)
                ssim = SSIM(gt, pred, multichannel=True)

                lpips_loss = loss_fn_alex(render_imgs[idx].unsqueeze(0).permute(0, 3, 1, 2),
                                          (img[idx] * masks[idx]).unsqueeze(0).permute(0, 3, 1, 2))
                psnr_sum += psnr
                ssim_sum += ssim
                lpips += lpips_loss.item()

                total_rp_psnr.append(psnr)
                total_rp_ssim.append(ssim)
                total_rp_LPIPS.append(lpips_loss.item())

                torchvision.utils.save_image(render_imgs[idx].permute(2, 0, 1), input_mesh_dire + '/repose/%s_%s.png' % (ori_name, img_name))
                torchvision.utils.save_image((img[idx] * masks[idx]).permute(2, 0, 1),
                                             input_mesh_dire + '/repose/gt_%s_%s.png' % (ori_name, img_name))
                # print(scan_id, psnr, ssim, lpips_loss.item())

    print("repose ave: ", f"PSNR value is {psnr_sum / (test_scan_id)} dB", f"SSIM value is {ssim_sum / (test_scan_id) }",
          f"LPIPS value is {lpips / (test_scan_id)}")
    print('repose render fps:', 1 / (infer_time / (test_scan_id)))

def write2video(path, outpath):
    import glob

    img_list = sorted(glob.glob(path))
    video_writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1280, 1024))
    for img in img_list:
        video_writer.write(cv2.imread(img))
    video_writer.release()




def get_valinterhand_data(data_path, scan_id, res=(334,512), data_name='0003_fake_gun',
                       capture_name='Capture9', drop_cam=[], split='train', return_ray=False):
    capture_idx = capture_name.replace('Capture', '')

    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_camera.json' % split)) as f:
        cam_params = json.load(f)

    cam_param = cam_params[capture_idx]
    camera_names = [i for i in sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name)))
                    if i not in drop_cam and '400' in i]
    imgs = []
    w2cs = []
    projs = []
    img_name = sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name, camera_names[0])))[
        scan_id]

    for i, cam_name in enumerate(camera_names):
        cam_idx = cam_name.replace('cam', '')
        img = cv2.imread(join(data_path, 'images/%s' % split, capture_name, data_name, 'cam' + cam_idx, img_name))
        img = cv2.resize(img, res)
        img = torch.from_numpy((img / 255.)).float().cuda()
        imgs.append(img)

        cam_idx = cam_name.replace('cam', '')
        t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(
            cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3, 3)
        scale_mats = np.eye(4)
        scale_mats[:3, :3] = R
        cam_t = -np.dot(R, t.reshape(3, 1)).reshape(3) / 1000
        scale_mats[:3, 3] = cam_t

        focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
        princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
        cameraIn = np.array([[focal[0], 0, princpt[0]],
                             [0, focal[1], princpt[1]],
                             [0, 0, 1]])

        P = cameraIn @ scale_mats[:3]
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

    imgs = torch.stack(imgs, dim=0)
    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()

    if return_ray:
        ray_directions = []
        c2ws = torch.inverse(w2cs.permute(0, 2, 1))
        for i, cam_name in enumerate(camera_names):
            cam_idx = cam_name.replace('cam', '')
            cam_ray_direction = get_ray_directions(res[1], res[0], cam_param['focal'][cam_idx][0],
                                       cam_param['focal'][cam_idx][1],
                                       cam_param['princpt'][cam_idx][0],
                                       cam_param['princpt'][cam_idx][1],).cuda()

            tmp_ray_direction, _ = get_rays(cam_ray_direction, c2ws[i])

            ray_direction = tmp_ray_direction.reshape(res[1], res[0], 3)
            ray_directions.append(ray_direction)
        ray_directions = torch.stack(ray_directions)

    return imgs, w2cs, projs, ray_directions, camera_names, img_name



if __name__ == "__main__":
    # dataset = InterhandDataset('/mnt/ssd/InterHand2.6M_5fps_batch1')
    # start_t = time.time()
    # data = dataset[2000]
    # print(time.time() - start_t)
    # exit()
    # train('conf/ih_nr.conf', 10, continue_train=True, net_type='unet')
    # eval('conf/ih_nr.conf', 10, net_type='unet')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='conf/ih_sfs.conf')
    parser.add_argument('--scan_id', type=int, default=6)
    parser.add_argument('--range', type=int, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--net_type', type=str, default='mlp')
    args = parser.parse_args()

    if args.range is not None:
        start = 1
        end = args.range + 1
    else:
        start = args.scan_id
        end = args.scan_id + 1

    total_psnr, total_ssim, total_LPIPS, total_rp_psnr, total_rp_ssim, total_rp_LPIPS = [], [], [], [], [], []

    for i in range(start, end):
        train(args.conf, i, data_path=args.data_path, continue_train=False, net_type=args.net_type)
        eval(args.conf, i, data_path=args.data_path, net_type=args.net_type)
        # eval_repose(args.conf, i, data_path=args.data_path, net_type=args.net_type)