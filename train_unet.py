import os
import time
from os.path import join
import json
from tqdm import tqdm
from pyhocon import ConfigFactory
import numpy as np
import math
from PIL import Image
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from models.utils import get_normals
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.utils import load_K_Rt_from_P
import torchvision
from torchvision import transforms
from models.unet import UNet
from models.PostionalEncoding import PostionalEncoding
from models.get_rays import get_ray_directions, get_rays
from skimage.metrics import structural_similarity as SSIM


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [8， 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)*(img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 9:
        return float('inf')
    return 28 * math.log10(1.0 / math.sqrt(mse))

class InterhandDataset(Dataset):

    def __init__(self, data_path, split='train', drop_cam=[], res=(334, 512), hand_type='left'):
        self.split = split
        self.data_path = data_path
        self.drop_cam = drop_cam
        self.res = res
        self.hand_type = hand_type
        with open(join(self.data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
            self.mano_params = json.load(f)
        with open(join(self.data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_camera.json' % split)) as f:
            self.cam_params = json.load(f)

        for cap_idx in self.cam_params:
            self.cam_params[cap_idx]['cam_ray_direction'] = {}
            for cam_idx in self.cam_params[cap_idx]['focal']:
                self.cam_params[cap_idx]['cam_ray_direction'][cam_idx] = \
                    get_ray_directions(self.res[1], self.res[0], self.cam_params[cap_idx]['focal'][cam_idx][0],
                                       self.cam_params[cap_idx]['focal'][cam_idx][1],
                                       self.cam_params[cap_idx]['princpt'][cam_idx][0],
                                       self.cam_params[cap_idx]['princpt'][cam_idx][1],)



        self.data_list = []
        capture_names = os.listdir(join(self.data_path, 'images', self.split))
        for capture_name in capture_names:
            capture_idx = capture_name.replace('Capture', '')
            data_names = [i for i in sorted(os.listdir(join(self.data_path, 'images', self.split, capture_name))) if 'dh' not in i]
            for data_name in data_names:
                cam_names = sorted(os.listdir(join(self.data_path, 'images', self.split, capture_name, data_name)))
                img_names = sorted(os.listdir(join(self.data_path, 'images', self.split, capture_name, data_name, cam_names[0])))
                for scan_id, img_name in enumerate(img_names):
                    img_name = img_name[5:-4]
                    if str(int(img_name)) in self.mano_params[capture_idx].keys() and \
                            os.path.exists(join(self.data_path, 'features', self.split, capture_name, data_name, '%s.npy' % img_name)): # and \
                        # self.mano_params[capture_idx][str(int(img_name))][self.hand_type] is not None:
                        camera_names = [i for i in sorted(
                            os.listdir(join(self.data_path, 'images', self.split, capture_name, data_name)))
                                        if i not in self.drop_cam and '400' in i]
                        for camera_name in camera_names:
                            self.data_list.append([capture_name, data_name, img_name, camera_name])
        self.transform = transforms.Compose([
            transforms.Resize((self.res[1], self.res[0])),
            transforms.ToTensor(),

        ])

    def get_item(self, capture_name, data_name, img_name, cam_name):
        time_n = time.time()
        capture_idx = capture_name.replace('Capture', '')
        cam_param = self.cam_params[capture_idx]
        hand_type = self.hand_type

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

        proj[0, 0] = proj[0, 0] / (self.res[0] / 2.)
        proj[0, 2] = proj[0, 2] / (self.res[0] / 2.) - 1.
        proj[1, 1] = proj[1, 1] / (self.res[1] / 2.)
        proj[1, 2] = proj[1, 2] / (self.res[1] / 2.) - 1.
        proj[2, 2] = 0.
        proj[2, 3] = -0.1
        proj[3, 2] = 1.
        proj[3, 3] = 0.

        # img = cv2.imread(join(self.data_path, 'images', self.split, capture_name, data_name, cam_name, 'image%s.jpg' % img_name))
        img_pil = Image.open(join(self.data_path, 'images', self.split, capture_name, data_name, cam_name, 'image%s.jpg' % img_name))
        # img = cv2.resize(img, self.res)
        img = self.transform(img_pil).permute(1, 2, 0)
        img_pil.close()

        with torch.no_grad():
            cam_ray_direction = cam_param['cam_ray_direction'][cam_idx]  # get_ray_directions(self.res[1], self.res[0], focal[0], focal[1], princpt[0], princpt[1])

            c2w = torch.inverse(torch.from_numpy(w2c))
            tmp_ray_direction, _ = get_rays(cam_ray_direction, c2w)

            ray_direction = tmp_ray_direction.reshape(self.res[1], self.res[0], 3)

            w2c = w2c.astype(np.float32).T
            proj = proj.astype(np.float32).T

            features = np.load(join(self.data_path, 'features', self.split, capture_name, data_name,
                                    '%s.npy' % img_name), allow_pickle=True).item()

            albedo = features['albedo'][0]
            feature = features['feature'][0]
            if 'vertices' not in features:
                print(join(self.data_path, 'features', self.split, capture_name, data_name, '%s.npy' % img_name))
            vertices = features['vertices']
            faces = features['faces']
            # if vertices.shape[0] != 49281:
            albedo = albedo[:49281]
            feature = feature[:49281]
            vertices = vertices[:49281]
            faces = faces[:98432]

        return img, ray_direction, w2c, proj, vertices, faces, albedo, feature

    def __getitem__(self, index):
        return self.get_item(self.data_list[index][0], self.data_list[index][1],
                             self.data_list[index][2], self.data_list[index][3])

    def __len__(self):
        return len(self.data_list)

def unet_forward(u_net, pe, glctx, inputs, resolution, if_geo=False):
    if not if_geo:
        ray, w2cs, projs, vertices, faces, albedo, vertex_feat = inputs
    else:
        ray, w2cs, projs, vertices, faces, albedo, img_z = inputs
    batch = w2cs.shape[0]

    uni_vertices = vertices.clone().uniform_(0, 1)

    vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1])], axis=2)
    rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2cs)
    proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, projs)
    normals = get_normals(rot_verts[:, :, :3], faces.long())

    rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
    if not if_geo:
        feat = torch.cat([torch.ones_like(vertsw[:, :, :1]), normals, albedo, uni_vertices,
                      vertex_feat.unsqueeze(0).expand(batch, -1, -1)], 2)
    else:
        feat = torch.cat([torch.ones_like(vertsw[:, :, :1]), normals, albedo, uni_vertices], 2)
    feat, _ = dr.interpolate(feat, rast_out, faces)
    masks = feat[:, :, :, :1].contiguous()  # [B, 1, H, W]


    if not if_geo:
        normal_map = pe(feat[:, :, :, 1:4].contiguous())
        albedo_map = pe(feat[:, :, :, 4:7].contiguous())  # [B, H, W, 87]
        pos = pe(feat[:, :, :, 7:10].contiguous())  # [B, H, W, 87]
        vertex_f = feat[:, :, :, 10:30].contiguous()  # [B, H, W, 20]
        input_f = torch.cat([normal_map, albedo_map, pos, ray, vertex_f], 3).permute(0, 3, 1, 2)
    else:
        normal_map = feat[:, :, :, 1:4].contiguous()
        albedo_map = feat[:, :, :, 4:7].contiguous()  # [B, H, W, 3]
        pos = feat[:, :, :, 7:10].contiguous()  # [B, H, W, 3]
        vertex_f = img_z.contiguous()
        input_f = torch.cat([normal_map, albedo_map, pos, vertex_f], 3).permute(0, 3, 1, 2)

    if input_f.shape[-1] % 8 != 0:
        input_f = torch.cat([torch.zeros_like(input_f[:, :, :, :1]), input_f, torch.zeros_like(input_f[:, :, :, :1])], 3)
        render_imgs = u_net(input_f)[:, :, :, 1:-1].permute(0, 2, 3, 1)
    else:
        render_imgs = u_net(input_f).permute(0, 2, 3, 1)

    return render_imgs, masks

def train(conf, continue_train=False):
    conf = ConfigFactory.parse_file(conf)

    data_path = conf.get_string('data_path')
    drop_cam = conf.get_string('drop_cam').split(',')

    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)
    degree = conf.get_int('degree')
    batch = 16
    num_epoch = 50
    train_dataset = InterhandDataset(data_path, split='train', drop_cam=drop_cam, res=(w, h))
    train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=8,
                                   pin_memory=False, drop_last=True)

    u_net = UNet(284, 3, 2, 0).cuda()
    pe = PostionalEncoding(min_deg=0, max_deg=1, scale=0.1)
    glctx = dr.RasterizeGLContext()

    if continue_train:
        pt = torch.load('checkpoints/unet_0.pth')
        vertex_feat = pt['vertex_feat'].cuda()
        u_net.load_state_dict(pt['model'])
    else:
        vertex_feat = torch.zeros(49281, 20).cuda()
    vertex_feat.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': u_net.parameters(), 'lr': 0.00001}, {'params': vertex_feat, 'lr': 0.00001}])

    for epoch in range(num_epoch):
        u_net.train()
        pbar = tqdm(train_data_loader)
        for idx, train_data in enumerate(pbar):
            # retrieve the data
            imgs = train_data[0].cuda()
            ray = train_data[1].cuda()
            w2cs = train_data[2].cuda()
            projs = train_data[3].cuda()
            vertices = train_data[4].cuda()
            faces = train_data[5].cuda()[0]
            albedo = train_data[6].cuda()
            feature = train_data[7].cuda()

            render_imgs, masks = unet_forward(u_net, pe, glctx,
                                              [ray, w2cs, projs, vertices, faces, albedo, vertex_feat], resolution)

            valid_index = masks[:, :, :, 0] > 0
            loss = F.smooth_l1_loss(render_imgs[valid_index], imgs[valid_index])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description('epoch: %d, loss: %f' % (epoch, loss.item()))

        if epoch % 5 == 0:
            torch.save({'model': u_net.state_dict(), 'vertex_feat': vertex_feat.detach().cpu()},
                       'checkpoints/unet_%d.pth' % epoch)

def eval(conf):
    conf = ConfigFactory.parse_file(conf)

    data_path = conf.get_string('data_path')
    drop_cam = conf.get_string('drop_cam').split(',')

    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)
    degree = conf.get_int('degree')

    u_net = UNet(284, 3, 2, 0).cuda()
    pe = PostionalEncoding(min_deg=0, max_deg=1, scale=0.1)
    glctx = dr.RasterizeGLContext()

    train_dataset = InterhandDataset(data_path, split='train', drop_cam=drop_cam, res=(w, h))
    os.makedirs('eval_unet', exist_ok=True)

    pt = torch.load('checkpoints/unet_45.pth')
    vertex_feat = pt['vertex_feat'].cuda()
    u_net.load_state_dict(pt['model'])

    psnr_sum = 0
    ssim_sum = 0
    for i in range(10):
        index = np.random.randint(0, len(train_dataset))
        img, ray_direction, w2c, proj, vertices, faces, albedo, feature = train_dataset[index]
        img = img.cuda()
        ray = ray_direction.cuda().unsqueeze(0)
        w2cs = torch.from_numpy(w2c).cuda().unsqueeze(0)
        projs = torch.from_numpy(proj).cuda().unsqueeze(0)
        vertices = torch.from_numpy(vertices).cuda().unsqueeze(0)
        faces = torch.from_numpy(faces).cuda()
        albedo = torch.from_numpy(albedo).cuda().unsqueeze(0)

        with torch.no_grad():
            render_imgs, masks = unet_forward(u_net, pe, glctx,
                                              [ray, w2cs, projs, vertices, faces, albedo, vertex_feat], resolution)

            render_imgs[masks[:, :, :, 0] == 0] = 0

            idx = 0
            gt = (img * masks[idx]).detach().cpu().numpy()
            pred = render_imgs[idx].detach().cpu().numpy()
            psnr = calculate_psnr(gt, pred, masks[idx].detach().cpu().numpy()) #PSNR(gt, pred)
            ssim = SSIM(gt, pred, multichannel=True)
            psnr_sum += psnr
            ssim_sum += ssim
            print(f"PSNR value is {psnr} dB")
            print(f"SSIM value is {ssim}")
        torchvision.utils.save_image(render_imgs.permute(0, 3, 1, 2)[0], 'eval_unet/%d.png' % i)
    print("ave: ", psnr_sum / 10, ssim_sum / 10)



if __name__ == "__main__":
    # dataset = InterhandDataset('/mnt/ssd/InterHand2.6M_5fps_batch1')
    # start_t = time.time()
    # data = dataset[2000]
    # print(time.time() - start_t)
    # exit()
    # train('conf/ih_sfs.conf')
    eval('conf/ih_sfs.conf')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf', type=str, default='conf/ih_sfs.conf')
    # parser.add_argument('--scan_id', type=int, default=6)
    # parser.add_argument('--data_path', type=str, default=None)
    # args = parser.parse_args()
    # main(args.conf, args.data_path)
