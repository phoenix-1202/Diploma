from pathlib import Path

import torch
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import save_npy_img, image_manifold_size, write_json

from testing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoint/last_state.pth"
BEST_CHECKPOINT_PATH = "./checkpoint/best_last_state.pth"

WIDTH = 108
HEIGHT = 108

OBJ = 9
LEN_CLASSES = 5
P = 4

BATCH = 64


def layout_bbox(final_pred, output_width, output_height):
    OO = final_pred.shape[1]
    LL = OO - P

    bbox_reg = final_pred[:, :, 0:P].to(device)
    cls_prob = final_pred[:, :, P:P + LL].to(device)

    bbox_reg = torch.reshape(bbox_reg, (BATCH, OO, P))

    x_c = bbox_reg[:, :, 0:1] * output_width
    y_c = bbox_reg[:, :, 1:2] * output_height
    w = bbox_reg[:, :, 2:3] * output_width
    h = bbox_reg[:, :, 3:4] * output_height

    x1 = x_c - 0.5 * w
    x2 = x_c + 0.5 * w
    y1 = y_c - 0.5 * h
    y2 = y_c + 0.5 * h

    xt = torch.reshape(torch.arange(0.0, output_width, 1.0), (1, 1, 1, -1)).to(device)
    xt = torch.reshape(torch.tile(xt, (BATCH, OO, output_height, 1)), (BATCH, OO, -1)).to(
        device)

    yt = torch.reshape(torch.arange(0.0, output_height, 1.0), (1, 1, -1, 1)).to(device)
    yt = torch.reshape(torch.tile(yt, (BATCH, OO, 1, output_width)), (BATCH, OO, -1)).to(
        device)

    x1_diff = torch.reshape(xt - x1, (BATCH, OO, output_height, output_width, 1))
    y1_diff = torch.reshape(yt - y1, (BATCH, OO, output_height, output_width, 1))
    x2_diff = torch.reshape(x2 - xt, (BATCH, OO, output_height, output_width, 1))
    y2_diff = torch.reshape(y2 - yt, (BATCH, OO, output_height, output_width, 1))

    f = torch.nn.ReLU()
    x1_line = f(1.0 - torch.abs(x1_diff)) * \
              torch.minimum(f(y1_diff), torch.ones_like(y1_diff)) * \
              torch.minimum(f(y2_diff), torch.ones_like(y2_diff))
    x2_line = f(1.0 - torch.abs(x2_diff)) * \
              torch.minimum(f(y1_diff), torch.ones_like(y1_diff)) * \
              torch.minimum(f(y2_diff), torch.ones_like(y2_diff))
    y1_line = f(1.0 - torch.abs(y1_diff)) * \
              torch.minimum(f(x1_diff), torch.ones_like(x1_diff)) * \
              torch.minimum(f(x2_diff), torch.ones_like(x2_diff))
    y2_line = f(1.0 - torch.abs(y2_diff)) * \
              torch.minimum(f(x1_diff), torch.ones_like(x1_diff)) * \
              torch.minimum(f(x2_diff), torch.ones_like(x2_diff))

    xy = torch.cat((x1_line, x2_line, y1_line, y2_line), dim=-1)
    xy_max = torch.amax(xy, dim=-1, keepdim=True)

    spatial_prob = torch.multiply(torch.tile(xy_max, (1, 1, 1, 1, LL)),
                                  torch.reshape(cls_prob, (BATCH, OO, 1, 1, LL)))
    spatial_prob_max = torch.amax(spatial_prob, dim=1, keepdim=False)

    return spatial_prob_max


class RelationNonLocal(torch.nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels

        self.cv0 = torch.nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.cv1 = torch.nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.cv2 = torch.nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.cv3 = torch.nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(self, inputs):
        N, C, H, W = inputs.shape
        assert C == self.channels, (C, self.channels)

        output_dim, d_k, d_g = C, C, C

        # NCHW -> NHWC
        f_v = self.cv0(inputs).permute(0, 2, 3, 1).contiguous()
        f_k = self.cv1(inputs).permute(0, 2, 3, 1).contiguous()
        f_q = self.cv2(inputs).permute(0, 2, 3, 1).contiguous()

        f_k = torch.reshape(f_k, (N, H * W, d_k))
        f_q = torch.reshape(f_q, (N, H * W, d_k))
        f_q = f_q.permute(0, 2, 1).contiguous()
        f_v = torch.reshape(f_v, (N, H * W, output_dim))

        # (N, H*W, d_k) * (N, d_k, H*W) -> (N, H*W, H*W)
        w = torch.matmul(f_k, f_q) / (H * W)

        # (N, H*W, H*W) * (N, H*W, output_dim) -> (N, H*W, output_dim)
        f_r = torch.matmul(w.permute(0, 2, 1).contiguous(), f_v)
        f_r = torch.reshape(f_r, (N, H, W, output_dim))

        # NHWC -> NCHW
        f_r = f_r.permute(0, 3, 1, 2).contiguous()
        f_r = self.cv3(f_r)

        return f_r


def initialize_layer(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bn0_0 = torch.nn.BatchNorm2d(256)
        self.bn0_1 = torch.nn.BatchNorm2d(64)
        self.bn0_2 = torch.nn.BatchNorm2d(64)
        self.bn0_3 = torch.nn.BatchNorm2d(256)

        self.cv0_0 = torch.nn.Conv2d(P + LEN_CLASSES, 256, kernel_size=(1, 1))
        self.cv0_1 = torch.nn.Conv2d(P + LEN_CLASSES, 64, kernel_size=(1, 1))
        self.cv0_2 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1))
        self.cv0_3 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1))

        self.bn1_0 = torch.nn.BatchNorm2d(1024)
        self.bn1_1 = torch.nn.BatchNorm2d(256)
        self.bn1_2 = torch.nn.BatchNorm2d(256)
        self.bn1_3 = torch.nn.BatchNorm2d(1024)

        self.cv1_0 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1))
        self.cv1_1 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1))
        self.cv1_2 = torch.nn.Conv2d(256, 256, kernel_size=(1, 1))
        self.cv1_3 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1))

        self.g_bn_x0 = torch.nn.BatchNorm2d(256)
        self.g_bn_x1 = torch.nn.BatchNorm2d(256)
        self.g_bn_x2 = torch.nn.BatchNorm2d(256)
        self.g_bn_x3 = torch.nn.BatchNorm2d(256)

        self.rel0 = RelationNonLocal(256)
        self.rel1 = RelationNonLocal(256)

        self.g_bn_x4 = torch.nn.BatchNorm2d(1024)
        self.g_bn_x5 = torch.nn.BatchNorm2d(1024)
        self.g_bn_x6 = torch.nn.BatchNorm2d(1024)
        self.g_bn_x7 = torch.nn.BatchNorm2d(1024)

        self.rel2 = RelationNonLocal(1024)
        self.rel3 = RelationNonLocal(1024)

        self.cv_bbox = torch.nn.Conv2d(1024, P, kernel_size=(1, 1))
        self.cv_cls = torch.nn.Conv2d(1024, LEN_CLASSES, kernel_size=(1, 1))

        self.relu = torch.nn.ReLU()

        self.rel0.apply(initialize_layer)
        self.rel1.apply(initialize_layer)
        self.rel2.apply(initialize_layer)
        self.rel3.apply(initialize_layer)

    def forward(self, z):
        gnet = torch.permute(z, (0, 2, 1))
        gnet = torch.reshape(gnet, (BATCH, P + LEN_CLASSES, OBJ, 1))

        # gnet -> h0_0
        #  └─> h0_1 -> h0_2 -> h0_3
        h0_0 = self.bn0_0(self.cv0_0(gnet))
        h0_1 = self.relu(self.bn0_1(self.cv0_1(gnet)))
        h0_2 = self.relu(self.bn0_2(self.cv0_2(h0_1)))
        h0_3 = self.bn0_3(self.cv0_3(h0_2))

        # gnet: (-1, 256, NUM, 1)
        gnet = self.relu(torch.add(h0_0, h0_3))

        gnet = self.relu(self.g_bn_x1(torch.add(gnet, self.g_bn_x0(self.rel0(gnet)))))
        gnet = self.relu(self.g_bn_x3(torch.add(gnet, self.g_bn_x2(self.rel1(gnet)))))

        # gnet -> h1_0 -> h1_1 -> h1_2 -> h1_3
        h1_0 = self.bn1_0(self.cv1_0(gnet))
        h1_1 = self.relu(self.bn1_1(self.cv1_1(h1_0)))
        h1_2 = self.relu(self.bn1_2(self.cv1_2(h1_1)))
        h1_3 = self.bn1_3(self.cv1_3(h1_2))
        # gnet: (-1, 256, NUM, 1)
        gnet = self.relu(torch.add(h1_0, h1_3))

        gnet = self.relu(self.g_bn_x5(torch.add(gnet, self.g_bn_x4(self.rel2(gnet)))))
        gnet = self.relu(self.g_bn_x7(torch.add(gnet, self.g_bn_x6(self.rel3(gnet)))))

        bbox_pred = self.cv_bbox(gnet)
        bbox_pred = torch.reshape(bbox_pred, (-1, P, OBJ))
        bbox_pred = torch.sigmoid(bbox_pred)  # : 0 ~ 1

        cls_score = self.cv_cls(gnet)
        cls_score = torch.reshape(cls_score, (-1, LEN_CLASSES, OBJ))
        cls_prob = torch.sigmoid(cls_score)  # : 0 ~ 1

        # (-1, DIM+CLS, NUM)
        final_pred = torch.cat([bbox_pred, cls_prob], dim=1)
        final_pred = torch.permute(final_pred, (0, 2, 1))

        return final_pred


class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=LEN_CLASSES, out_channels=32, kernel_size=(5, 5), stride=(2, 2),
                            padding='valid'),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding='valid'),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=5376, out_features=512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=512, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, image):
        layout = layout_bbox(image, WIDTH, HEIGHT)
        layout = torch.permute(layout, (0, 3, 1, 2))
        output = self.model(layout)
        return output


def gen_noise():
    batch_z_bbox = np.random.normal(0.5, 0.15, (BATCH, OBJ, P))
    batch_z_cls = np.identity(LEN_CLASSES)[np.random.randint(LEN_CLASSES, size=(BATCH, OBJ))]
    batch_z = np.concatenate([batch_z_bbox, batch_z_cls], axis=-1)
    noise = torch.tensor(batch_z, dtype=torch.float).to(device)
    return noise


class LayoutGAN:
    def __init__(self):
        self.gen = Generator().to(device)
        self.disc = Discriminator().to(device)

    def train(self):
        n_epoch = 101
        start_epoch = 0

        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=5e-5)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=5e-5)

        criterion = torch.nn.BCELoss()

        row_data = np.load('./data/doc_train.npy')
        dataloader = DataLoader(TensorDataset(torch.Tensor(row_data)), batch_size=BATCH, shuffle=True, drop_last=True)

        checkpoint_file = Path(CHECKPOINT_PATH)
        if checkpoint_file.exists():
            checkpoint = torch.load(CHECKPOINT_PATH)
            start_epoch = checkpoint['start_epoch']
            self.gen.load_state_dict(checkpoint['generator'])
            self.disc.load_state_dict(checkpoint['discriminator'])
            gen_opt.load_state_dict(checkpoint['gen_opt'])
            disc_opt.load_state_dict(checkpoint['disc_opt'])

        def get_fake_pred(should_detach):
            noise = gen_noise()
            if should_detach:
                with torch.no_grad():
                    fake_tensor = self.gen(noise)
            else:
                fake_tensor = self.gen(noise)
            fake_pred = self.disc(fake_tensor)
            return fake_tensor, fake_pred

        my_sample = gen_noise()
        counter = 0
        for epoch in range(start_epoch, n_epoch):
            print("Epoch ", epoch)
            idx = 0
            for data in tqdm(dataloader):
                real_layout_tensor = data[0].to(device)

                # Update D network
                # Train with all-real batch
                disc_opt.zero_grad()
                disc_real_pred = self.disc(real_layout_tensor)
                label = torch.ones_like(disc_real_pred)
                disc_loss_real = criterion(disc_real_pred, label)

                # Train with all-fake batch
                _, disc_fake_pred = get_fake_pred(should_detach=True)
                label = torch.zeros_like(disc_fake_pred)
                disc_loss_fake = criterion(disc_fake_pred, label)

                disc_loss = disc_loss_real + disc_loss_fake
                disc_loss.backward()
                disc_opt.step()

                # Update G network
                gen_opt.zero_grad()
                _, disc_fake_pred = get_fake_pred(should_detach=False)
                label = torch.ones_like(disc_fake_pred)
                gen_loss = criterion(disc_fake_pred, label)
                gen_loss.backward()
                gen_opt.step()

                if counter % 500 == 0:
                    with torch.no_grad():
                        res_tensor = self.gen(my_sample)
                    image = layout_bbox(res_tensor, WIDTH, HEIGHT)
                    write_json(res_tensor, '{:03d}_{:04d}'.format(epoch, idx))
                    size = image_manifold_size(list(image.size())[0])
                    path = './samples/train_{:03d}_{:04d}.jpg'.format(epoch, idx)
                    save_npy_img(image.detach().cpu().numpy(), size, path)

                idx += 1
                counter += 1

            if epoch % 1 == 0:
                torch.save({'start_epoch': epoch + 1,
                            'generator': self.gen.state_dict(),
                            'discriminator': self.disc.state_dict(),
                            'gen_opt': gen_opt.state_dict(),
                            'disc_opt': disc_opt.state_dict()
                            }, CHECKPOINT_PATH)
                torch.save({'start_epoch': epoch + 1,
                            'generator': self.gen.state_dict(),
                            'discriminator': self.disc.state_dict(),
                            'gen_opt': gen_opt.state_dict(),
                            'disc_opt': disc_opt.state_dict()
                            }, "./checkpoint/state_{:03d}".format(epoch) + ".pth")

    def test(self, cur_checkpoint):
        checkpoint_file = Path(cur_checkpoint)
        if checkpoint_file.exists():
            checkpoint = torch.load(cur_checkpoint, map_location=device)
            start_epoch = checkpoint['start_epoch']
            print(start_epoch)
            self.gen.load_state_dict(checkpoint['generator'])
            self.disc.load_state_dict(checkpoint['discriminator'])

            for idx in range(5):
                tensor = gen_noise()
                with torch.no_grad():
                    res_tensor = self.gen(tensor)
                image = layout_bbox(res_tensor, WIDTH, HEIGHT)
                # write_json(res_tensor, '{:03d}_{:04d}'.format(start_epoch, idx))
                size = image_manifold_size(list(image.size())[0])
                path = './samples/test_{:03d}_{:04d}.jpg'.format(start_epoch, idx)
                save_npy_img(image.detach().cpu().numpy(), size, path)

                new_layout = from_doc_to_ad(res_tensor)
                image = layout_bbox(new_layout, WIDTH, HEIGHT)
                write_json(new_layout, '{:02d}'.format(idx))
                size = image_manifold_size(list(image.size())[0])
                path = './samples/ans_{:02d}.jpg'.format(idx)
                save_npy_img(image.detach().cpu().numpy(), size, path)
