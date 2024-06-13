import os
from os.path import join
import glob
import json
import numpy as np
import cv2
import trimesh
import torch
import smplx
import nvdiffrast.torch as dr
from models.utils import load_K_Rt_from_P
from models.get_rays import get_ray_directions, get_rays
mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True), 'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False)}

import sys

sys.path.append("/home/gqj/segment_anything")
from segment_anything import sam_model_registry, SamPredictor
import torchvision

def img_contrast_bright(img,a,b,g):
    h,w,c=img.shape
    blank=np.zeros([h,w,c],img.dtype)
    dst=cv2.addWeighted(img,a,blank,b,g)
    return dst

sam = None
def sam_hand(imgs, masks, level=0):
    global sam
    if sam is None:
        sam = sam_model_registry['vit_h'](checkpoint='/home/gqj/playground/models/sam_vit_h_4b8939.pth')
        sam.to(device='cuda')
    predictor = SamPredictor(sam)
    boxes = torchvision.ops.masks_to_boxes(masks)
    new_masks = []

    for box, image in zip(boxes, imgs):
        predictor.set_image((image.cpu().numpy() * 255).astype(np.uint8))
        new_mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box.cpu().numpy()[None, :],
            multimask_output=True,
        )
        new_masks.append(torch.from_numpy(new_mask[level:level+1]))

    return torch.cat(new_masks, 0).float().cuda()

def get_demo_data(data_path, scan_id, num, res=(1280,1024), return_ray=False, with_mask=True, use_mask=False):
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
        if with_mask:
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
    if return_ray:
        ray_directions = []
        c2ws = torch.inverse(w2cs.permute(0, 2, 1))
        for i in range(num):
            cam_ray_direction = get_ray_directions(res[1], res[0], camera_dict['int_%d' % i][0, 0],
                                       camera_dict['int_%d' % i][1, 1],
                                       camera_dict['int_%d' % i][0, 2],
                                       camera_dict['int_%d' % i][1, 2],).cuda()

            tmp_ray_direction, _ = get_rays(cam_ray_direction, c2ws[i])

            ray_direction = tmp_ray_direction.reshape(res[1], res[0], 3)
            ray_directions.append(ray_direction)
        ray_directions = torch.stack(ray_directions)

        return imgs, grayimgs, masks, w2cs, projs, ray_directions

    return imgs, grayimgs, masks, w2cs, projs

def get_interhand_data(data_path, scan_id, res=(334, 512), data_name='0003_fake_gun', use_mask=True, check_gcn=True,
                       capture_name='Capture9', drop_cam=[], split='train', save_mask=True, return_ray=False,
                       test_num=20, use_sam=True, load_refined=False, use_ori_gcn=True):
    mano_layer['right'] = mano_layer['right'].cpu()
    mano_layer['left'] = mano_layer['left'].cpu()
    capture_idx = capture_name.replace('Capture', '')

    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_camera.json' % split)) as f:
        cam_params = json.load(f)
    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
        mano_params = json.load(f)
    cam_param = cam_params[capture_idx]
    camera_names = [i for i in sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name)))
                    if i not in drop_cam and '400' in i]
    #if split == 'test':
        #camera_names = camera_names[:test_num]
    num = len(camera_names)
    print('use cameras:', camera_names, 'num:', num)
    img_name = sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name, camera_names[0])))[
        scan_id]
    print('use image name:', img_name)
    if os.path.exists(
            join('interhand_out/%s_%s/gcn_out' % (capture_name, data_name), '%d.obj' % scan_id)) and check_gcn:
        if load_refined and os.path.exists(
                join('interhand_out/%s_%s/gcn_out' % (capture_name, data_name), '%d_refined.obj' % scan_id)):
            print('use mano from refined gcn_out')
            mesh = trimesh.load(join('interhand_out/%s_%s/gcn_out' % (capture_name, data_name),
                                     '%d_refined.obj' % scan_id), process=False, maintain_order=True)
            mano_out = torch.load(
                join('interhand_out/%s_%s/gcn_out' % (capture_name, data_name), '%d_refined.pt' % scan_id))
        else:
            print('use mano from gcn_out')
            if use_ori_gcn:
                mesh = trimesh.load(join('interhand_out/%s_%s/gcn_out' % (capture_name, data_name),
                                         'ori_%d.obj' % scan_id), process=False, maintain_order=True)
            else:
                mesh = trimesh.load(join('interhand_out/%s_%s/gcn_out' % (capture_name, data_name),
                                         '%d.obj' % scan_id), process=False, maintain_order=True)
            mano_out = torch.load(join('interhand_out/%s_%s/gcn_out' % (capture_name, data_name), '%d.pt' % scan_id))
        vertices = torch.from_numpy(mesh.vertices).float().cuda().unsqueeze(0)
        faces = mesh.faces

    else:
        print('use mano from interhand')

        mano_param = mano_params[capture_idx][str(int(img_name[5:-4]))]
        vertices = []
        faces = []
        mano_out = []
        for hand_type in ['left', 'right']:
            if mano_param[hand_type] is not None:
                mano_pose = torch.FloatTensor(mano_param[hand_type]['pose']).view(-1, 3)
                root_pose = mano_pose[0].view(1, 3)
                hand_pose = mano_pose[1:, :].view(1, -1)
                shape = torch.FloatTensor(mano_param[hand_type]['shape']).view(1, -1)
                trans = torch.FloatTensor(mano_param[hand_type]['trans']).view(1, 3)
                output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)

                vertices.append(output.vertices)
                mano_out.append({'type': hand_type, 'pose': mano_pose, 'shape': shape, 'trans': trans})
                if len(faces) == 0:
                    faces.append(mano_layer[hand_type].faces)
                else:
                    faces.append(mano_layer[hand_type].faces + output.vertices.shape[1])

        vertices = torch.cat(vertices, 1).cuda()
        faces = np.concatenate(faces, 0)
        mesh = trimesh.Trimesh(vertices=vertices[0].detach().cpu().numpy(), faces=faces)
        os.makedirs('interhand_out/%s_%s/mano_out' % (capture_name, data_name), exist_ok=True)
        mesh.export(join('interhand_out/%s_%s/mano_out' % (capture_name, data_name),
                                         '%d.obj' % scan_id))
    # mesh = trimesh.Trimesh(vertices=vertices[0].detach().cpu().numpy(), faces=faces)
    # mesh.export('test.obj')
    faces = torch.from_numpy(faces.astype(np.int32)).int().cuda()

    w2cs = []
    projs = []
    imgs = []
    grayimgs = []

    for i, cam_name in enumerate(camera_names):
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

        img = cv2.imread(join(data_path, 'images/%s' % split, capture_name, data_name, 'cam' + cam_idx, img_name))
        # img = img_contrast_bright(img, 1.2, -0.2, 30)

        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, res)
        grayimg = cv2.resize(grayimg, res)

        img = torch.from_numpy((img / 255.)).float().cuda()
        grayimg = torch.from_numpy((grayimg / 255.)).float().cuda()

        imgs.append(img)
        grayimgs.append(grayimg)

    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()

    glctx = dr.RasterizeGLContext()
    vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1])], axis=2).expand(num, -1, -1)
    rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2cs)
    proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, projs)

    rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=(res[1], res[0]))
    feat = torch.ones_like(vertsw[:, :, :1])
    feat, _ = dr.interpolate(feat, rast_out, faces)
    masks = feat[:, :, :, :1].contiguous().squeeze(-1)
    # masks = dr.antialias(masks, rast_out, proj_verts, faces).squeeze(-1)

    imgs = torch.stack(imgs, dim=0)
    grayimgs = torch.stack(grayimgs, dim=0)

    if use_sam:
        masks = sam_hand(imgs, masks)

    if use_mask:
        imgs[masks == 0] = 0
        grayimgs[masks == 0] = 0
    if save_mask:
        os.makedirs('interhand_out/%s_%s/masks' % (capture_name, data_name), exist_ok=True)
        for cam_name, mask in zip(camera_names, masks):
            cv2.imwrite('interhand_out/%s_%s/masks/%s.png' % (capture_name, data_name, cam_name),
                        mask.cpu().numpy() * 255)
    if return_ray:
        ray_directions = []
        c2ws = torch.inverse(w2cs.permute(0, 2, 1))
        for i, cam_name in enumerate(camera_names):
            cam_idx = cam_name.replace('cam', '')
            cam_ray_direction = get_ray_directions(res[1], res[0], cam_param['focal'][cam_idx][0],
                                                   cam_param['focal'][cam_idx][1],
                                                   cam_param['princpt'][cam_idx][0],
                                                   cam_param['princpt'][cam_idx][1], ).cuda()

            tmp_ray_direction, _ = get_rays(cam_ray_direction, c2ws[i])

            ray_direction = tmp_ray_direction.reshape(res[1], res[0], 3)
            ray_directions.append(ray_direction)
        ray_directions = torch.stack(ray_directions)

        return imgs, grayimgs, masks, w2cs, projs, vertices[0], faces, mano_out, ray_directions
    return imgs, grayimgs, masks, w2cs, projs, vertices[0], faces, mano_out