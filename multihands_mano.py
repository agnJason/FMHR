import os
from os.path import join
import cv2
import numpy as np
import json
import argparse
import trimesh
from pyhocon import ConfigFactory
from mediapipe_hands import joints_est
from pose_optim import pose_optimize
from mano_optim import mano_optimize, mano_layer
import torch
from models.utils import load_K_Rt_from_P
from train_gcn import run_infer as gcn_mano
from models.gcn import GCNDecoder


def get_demo_data(data_path, scan_id, num, res=(1280,1024)):
    camera_dict = np.load(join(data_path, '%d/camera/param.npz' % scan_id))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    imgs = []
    grayimgs = []
    w2cs = []
    projs = []
    Pall = []
    img_names = []
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
        img_name = join(data_path, '%d/img/%02d.png'%(scan_id, i))
        img_names.append(img_name)
        img = cv2.imread(img_name)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, res)
        grayimg = cv2.resize(grayimg, res)

        img = torch.from_numpy((img/255.)).float().cuda()
        grayimg = torch.from_numpy((grayimg/255.)).float().cuda()

        imgs.append(img)
        grayimgs.append(grayimg)

    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0,2,1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0,2,1).cuda()
    imgs = torch.stack(imgs, dim=0)
    grayimgs = torch.stack(grayimgs, dim=0)

    return imgs, grayimgs, w2cs, projs, img_names

def get_interhand_data(data_path, scan_id, res=(334,512), data_name='0003_fake_gun',
                       capture_name='Capture9', drop_cam=[], split='train'):
    capture_idx = capture_name.replace('Capture', '')

    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_camera.json' % split)) as f:
        cam_params = json.load(f)
    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
        mano_params = json.load(f)


    cam_param = cam_params[capture_idx]
    camera_names = [i for i in sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name)))
                    if i not in drop_cam and '400' in i]
    num = len(camera_names)

    img_name = sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name, camera_names[0])))[scan_id]
    img_names = [join(data_path, 'images/%s' % split, capture_name, data_name, camera_name, img_name) for camera_name in camera_names]
    mano_param = mano_params[capture_idx][str(int(img_name[5:-4]))]
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

        img = cv2.imread(join(data_path, 'images/%s' % split, capture_name, data_name, cam_name, img_name))
        # img = img_contrast_bright(img, 1.2, -0.2, 30)

        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, res)
        grayimg = cv2.resize(grayimg, res)

        img = torch.from_numpy((img / 255.)).float().cuda()
        grayimg = torch.from_numpy((grayimg / 255.)).float().cuda()

        imgs.append(img)
        grayimgs.append(grayimg)

    imgs = torch.stack(imgs, dim=0)
    grayimgs = torch.stack(grayimgs, dim=0)
    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()

    return imgs, grayimgs, w2cs, projs, img_names, mano_param

def get_dhm_data(data_path, scan_id, res=(334,512), data_name='0003_fake_gun',
                       capture_name='subject_4'):
    with open('/hdd/data1/gqj/datasets/InterHand2.6M/InterHand2.6M_5fps_batch1/annotations/val/InterHand2.6M_val_camera.json') as f:
        cam_params = json.load(f)
    cam_param = cam_params["0"]

    camera_names = [i for i in sorted(os.listdir(join(data_path, 'images', capture_name, data_name)))]
    num = len(camera_names)

    img_name = sorted(os.listdir(join(data_path, 'images', capture_name, data_name, camera_names[0])))[scan_id]
    img_names = [join(data_path, 'images', capture_name, data_name, camera_name, img_name) for camera_name in
                 camera_names]
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

        img = cv2.imread(join(data_path, 'images', capture_name, data_name, cam_name, img_name))
        # img = img_contrast_bright(img, 1.2, -0.2, 30)

        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, res)
        grayimg = cv2.resize(grayimg, res)

        img = torch.from_numpy((img / 255.)).float().cuda()
        grayimg = torch.from_numpy((grayimg / 255.)).float().cuda()

        imgs.append(img)
        grayimgs.append(grayimg)

    imgs = torch.stack(imgs, dim=0)
    grayimgs = torch.stack(grayimgs, dim=0)
    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()

    return imgs, grayimgs, w2cs, projs, img_names

def mano_from_mvimages(conf, scan_id, data_path=None, use_optimmano=False):
    conf = ConfigFactory.parse_file(conf)
    type = conf.get_string('data_type')
    if data_path is None:
        data_path = conf.get_string('data_path')

    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)

    if type == 'demo':
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        out_mesh_dire = out_path
        os.makedirs(out_mesh_dire, exist_ok=True)
        imgs, grayimgs, w2cs, projs, img_names = get_demo_data(data_path, scan_id, num, res=(w,h))
    elif type == 'interhand':
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        split = conf.get_string('split')
        out_mesh_dire = './%s_out/%s_%s' % (type, capture_name, data_name)
        os.makedirs('%s_out' % type, exist_ok=True)
        os.makedirs(out_mesh_dire, exist_ok=True)
        drop_cam = conf.get_string('drop_cam').split(',')
        imgs, grayimgs, w2cs, projs, img_names, mano_param = get_interhand_data(data_path, scan_id, res=(334, 512), data_name=data_name,
                           capture_name=capture_name, drop_cam=drop_cam, split=split)
        num = len(img_names)
    elif type == 'dhm':
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        out_mesh_dire = './%s_out/%s_%s' % (type, capture_name, data_name)
        os.makedirs('%s_out' % type, exist_ok=True)
        os.makedirs(out_mesh_dire, exist_ok=True)
        imgs, grayimgs, w2cs, projs, img_names = get_dhm_data(data_path, scan_id, res=(334, 512), data_name=data_name,
                           capture_name=capture_name)
        num = len(img_names)

    os.makedirs('%s/keypoints3d' % out_mesh_dire, exist_ok=True)
    os.makedirs('%s/mano_out' % out_mesh_dire, exist_ok=True)

    poses = []
    drop_left = 0
    drop_right = 0
    weights = np.ones((num, 42))
    for i in range(num):
        results, pose_f = joints_est(img_names[i], 0.1)
        if False:
            if results.multi_handedness is not None:
                print('Handedness:', results.multi_handedness)
                image = cv2.flip(cv2.imread(img_names[i]), 1)
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                os.makedirs('%s/mano_out' % (out_mesh_dire) + '/annotated_image', exist_ok=True)
                cv2.imwrite(
                    '%s/mano_out' % (out_mesh_dire) + '/annotated_image/' + str(i) + '.png', cv2.flip(annotated_image, 1))

        if len(pose_f['Left']) == 0:
            pose_f['Left'] = [-1.0] * 63
            weights[i, :21] = 0
            drop_left += 1
        if len(pose_f['Right']) == 0:
            pose_f['Right'] = [-1.0] * 63
            weights[i, 21:] = 0
            drop_right += 1
        pose_t = np.array(pose_f['Left'] + pose_f['Right']).reshape(-1, 3)
        assert pose_t.shape[0] == 42
        pose_t = torch.from_numpy(pose_t).float().cuda()
        # pose_t[:, 0] = (pose_t[:, 0] * -0.5 + 0.5) * res[0]
        # pose_t[:, 1] = (pose_t[:, 1] * 0.5 + 0.5) * res[1]
        poses.append(pose_t)

    drop_left = (drop_left / num) > 0.8
    drop_right = (drop_right / num) > 0.8

    poses = torch.stack(poses, dim=0)
    weights = torch.from_numpy(np.stack(weights)).cuda()

    batch_pose = 200
    epoch_pose = 250

    pred_poses = pose_optimize(batch_pose, epoch_pose, w2cs, projs, poses, weights, init_interhand= type == 'interhand' or type == 'dhm')
    if drop_left:
        pred_poses[:21] = 0
    if drop_right:
        pred_poses[21:] = 0

    np.savetxt('%s/keypoints3d/keypoints_3d_%d.xyz' % (out_mesh_dire, scan_id), pred_poses.cpu().numpy())
    if use_optimmano:
        params_left, params_right, verts_l, verts_r = mano_optimize(pred_poses.unsqueeze(0))


        with torch.no_grad():
            output_r = mano_layer['right'](global_orient=params_right["pose"][:, :3],
                                      hand_pose=params_right["pose"][:, 3:],
                                      betas=params_right["shape"])
            output_l = mano_layer['left'](global_orient=params_left["pose"][:, :3],
                                      hand_pose=params_left["pose"][:, 3:],
                                      betas=params_left["shape"])

        params_right["pose"] = torch.cat([output_r.global_orient, output_r.hand_pose], 1).detach().cpu()
        params_right["shape"] = output_r.betas.detach().cpu()
        params_right["trans"] = params_right["trans"].detach().cpu()
        params_right["scale"] = params_right["scale"].detach().cpu()
        params_left["pose"] = torch.cat([output_l.global_orient, output_l.hand_pose], 1).detach().cpu()
        params_left["shape"] = output_l.betas.detach().cpu()
        params_left["trans"] = params_left["trans"].detach().cpu()
        params_left["scale"] = params_left["scale"].detach().cpu()
        if drop_left and drop_right:
            print('failed!')
        elif drop_left:
            mano_verts = verts_r[0].detach().cpu().numpy()
            faces = mano_layer['right'].faces
            torch.save([params_right], '%s/mano_out/%d.pt' % (out_mesh_dire, scan_id))
        elif drop_right:
            mano_verts = verts_l[0].detach().cpu().numpy()
            faces = mano_layer['left'].faces
            torch.save([params_left], '%s/mano_out/%d.pt' % (out_mesh_dire, scan_id))
        else:
            mano_verts = torch.cat([verts_l, verts_r], 1)[0].detach().cpu().numpy()
            faces = torch.cat([torch.from_numpy(mano_layer['left'].faces.astype(np.int32)).int(),
                               torch.from_numpy(mano_layer['right'].faces.astype(np.int32)).int() +
                               mano_layer['right'].v_template.shape[0]], 0).int().numpy()
            torch.save([params_left, params_right], '%s/mano_out/%d.pt' % (out_mesh_dire, scan_id))

        mesh = trimesh.Trimesh(vertices=mano_verts, faces=faces)

        mesh.export('%s/mano_out/%d.obj' % (out_mesh_dire, scan_id))
    else:
        gcn_net_left = GCNDecoder('./mano/TEMPLATE_LEFT.obj', 21 * 3).cuda()
        gcn_net_right = GCNDecoder('./mano/TEMPLATE_RIGHT.obj', 21 * 3).cuda()
        gcn_net_left.load_state_dict(torch.load("./mano/gcn_left.pth"))
        gcn_net_right.load_state_dict(torch.load("./mano/gcn_right.pth"))
        gcn_net = {'left': gcn_net_left, 'right': gcn_net_right}
        gcn_mano(gcn_net, out_mesh_dire, scan_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./conf/demo_sfs.conf')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--if_stream', type=bool, default=False)
    parser.add_argument('--scan_id', type=int, default=2)
    parser.add_argument('--range', type=int, default=None)
    args = parser.parse_args()
    if args.range is not None:
        start = 1
        end = args.range + 1
    else:
        start = args.scan_id
        end = args.scan_id + 1
    for i in range(start, end):
        mano_from_mvimages(args.conf, i, args.data_path)
