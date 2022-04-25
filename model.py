from pathlib import Path

import torch
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import save_npy_img, image_manifold_size, write_json, get_text_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoint/last_state.pth"
BEST_CHECKPOINT_PATH = "./checkpoint/best_last_state.pth"

WIDTH = 108
HEIGHT = 108

OBJ = 11
LEN_CLASSES = 7
P = 4

FROM = 0.5
BATCH = 64


def count_non_zero(tensor):
    x = torch.count_nonzero(tensor, dim=2)
    x = torch.count_nonzero(x, dim=1)
    x = x.type(torch.FloatTensor).to(device)
    z = torch.zeros_like(x)
    x = torch.where(3 <= x, x, (x - 3) / 3)
    x = torch.where(x <= 8, x, (8 - x) / 3)
    x = torch.where(x < 0, -x, z)
    return x


def modify_tensor(cl, batch_z):
    z = torch.zeros_like(cl).to(device)
    o = torch.ones_like(cl).to(device)

    new_cl = torch.where(cl > FROM, o, z).to(device)
    one = torch.ones(7).to(device)
    res = torch.matmul(new_cl, one).to(device)

    res = torch.diag_embed(res).to(device)
    z = torch.zeros_like(res).to(device)
    o = torch.ones_like(res).to(device)

    new_res = torch.where(res >= float(1.), o, z).to(device)
    return torch.matmul(new_res, batch_z)


def layout_bbox(final_pred, output_height, output_width):
    batch_size = final_pred.shape[0]
    objects_cnt = final_pred.shape[1]
    vector_size = final_pred.shape[2]
    properties_cnt = vector_size - LEN_CLASSES

    bbox_reg = final_pred[:, :, 0:properties_cnt].to(device)
    cls_prob = final_pred[:, :, properties_cnt:vector_size].to(device)

    bbox_reg = torch.reshape(bbox_reg, (batch_size, objects_cnt, properties_cnt))

    x_c = bbox_reg[:, :, 0:1] * output_width
    y_c = bbox_reg[:, :, 1:2] * output_height
    w = bbox_reg[:, :, 2:3] * output_width
    h = bbox_reg[:, :, 3:4] * output_height

    x1 = x_c - 0.5 * w
    x2 = x_c + 0.5 * w
    y1 = y_c - 0.5 * h
    y2 = y_c + 0.5 * h

    xt = torch.reshape(torch.arange(0.0, output_width, 1.0), (1, 1, 1, -1)).to(device)
    xt = torch.reshape(torch.tile(xt, (batch_size, objects_cnt, output_height, 1)), (batch_size, objects_cnt, -1)).to(
        device)

    yt = torch.reshape(torch.arange(0.0, output_height, 1.0), (1, 1, -1, 1)).to(device)
    yt = torch.reshape(torch.tile(yt, (batch_size, objects_cnt, 1, output_width)), (batch_size, objects_cnt, -1)).to(
        device)

    x1_diff = torch.reshape(xt - x1, (batch_size, objects_cnt, output_height, output_width, 1))
    y1_diff = torch.reshape(yt - y1, (batch_size, objects_cnt, output_height, output_width, 1))
    x2_diff = torch.reshape(x2 - xt, (batch_size, objects_cnt, output_height, output_width, 1))
    y2_diff = torch.reshape(y2 - yt, (batch_size, objects_cnt, output_height, output_width, 1))

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

    spatial_prob = torch.multiply(torch.tile(xy_max, (1, 1, 1, 1, LEN_CLASSES)),
                                  torch.reshape(cls_prob, (batch_size, objects_cnt, 1, 1, LEN_CLASSES)))
    spatial_prob_max = torch.amax(spatial_prob, dim=1, keepdim=False)

    return spatial_prob_max


def get_relation_non_local_block1(C):
    f_v = torch.nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(1, 1), stride=(1, 1))
    f_k = torch.nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(1, 1), stride=(1, 1))
    f_q = torch.nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(1, 1), stride=(1, 1))
    return f_v, f_k, f_q


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.h0_0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=P + LEN_CLASSES, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(256),
        )
        self.h0_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=P + LEN_CLASSES, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.h0_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(256),
        )

        self.h1_0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(1024),
        )
        self.h1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.h1_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.BatchNorm2d(1024),
        )

        self.relu = torch.nn.ReLU()

        self.bn_x0 = torch.nn.BatchNorm2d(256)
        self.bn_x1 = torch.nn.BatchNorm2d(256)
        self.bn_x2 = torch.nn.BatchNorm2d(1024)
        self.bn_x3 = torch.nn.BatchNorm2d(1024)
        self.bn_x0_relu = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.bn_x1_relu = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.bn_x2_relu = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )
        self.bn_x3_relu = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )

        self.v0, self.k0, self.q0 = get_relation_non_local_block1(256)
        self.v1, self.k1, self.q1 = get_relation_non_local_block1(256)
        self.v2, self.k2, self.q2 = get_relation_non_local_block1(1024)
        self.v3, self.k3, self.q3 = get_relation_non_local_block1(1024)

        self.r0 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        self.r1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        self.r2 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1))
        self.r3 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1))

        self.bbox0 = torch.nn.Conv2d(in_channels=1024, out_channels=P, kernel_size=(1, 1), stride=(1, 1))
        self.bbox1 = torch.nn.Sigmoid()

        self.cls0 = torch.nn.Conv2d(in_channels=1024, out_channels=LEN_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        self.cls1 = torch.nn.Sigmoid()

    def relation_non_local(self, input_, i):
        shape_org = list(input_.size())
        N, C, H, W = shape_org[0], shape_org[1], shape_org[2], shape_org[3]
        output_dim, d_k, d_g = C, C, C

        if i == 0:
            f_v, f_k, f_q = self.v0.to(device)(input_), self.k0.to(device)(input_), self.q0.to(device)(input_)
        elif i == 1:
            f_v, f_k, f_q = self.v1.to(device)(input_), self.k1.to(device)(input_), self.q1.to(device)(input_)
        elif i == 2:
            f_v, f_k, f_q = self.v2.to(device)(input_), self.k2.to(device)(input_), self.q2.to(device)(input_)
        else:
            f_v, f_k, f_q = self.v3.to(device)(input_), self.k3.to(device)(input_), self.q3.to(device)(input_)

        f_k = torch.permute(torch.reshape(f_k, (N, d_k, H * W)), (0, 2, 1)).to(device)
        f_q = torch.reshape(f_q, (N, d_k, H * W)).to(device)
        w = torch.matmul(f_k, f_q) / (H * W)

        f_v = torch.permute(torch.reshape(f_v, (N, d_k, H * W)), (0, 2, 1)).to(device)
        f_r = torch.matmul(torch.permute(w, (0, 2, 1)), f_v).to(device)
        f_r = torch.reshape(torch.permute(f_r, (0, 2, 1)), (N, output_dim, H, W)).to(device)

        if i == 0:
            f_r = self.r0.to(device)(f_r)
        elif i == 1:
            f_r = self.r1.to(device)(f_r)
        elif i == 2:
            f_r = self.r2.to(device)(f_r)
        else:
            f_r = self.r3.to(device)(f_r)

        return f_r

    def forward(self, noise):
        gnet = torch.reshape(noise, (BATCH, OBJ, 1, P + LEN_CLASSES))
        gnet = torch.permute(gnet, (0, 3, 1, 2))

        h0_0 = self.h0_0(gnet)
        h0_1 = self.h0_1(gnet)
        h0_3 = self.h0_3(h0_1)
        add = torch.add(h0_0, h0_3)
        gnet = self.relu(add)

        rl0 = self.bn_x0(self.relation_non_local(gnet, 0))
        add0 = torch.add(gnet, rl0)
        gnet = self.bn_x0_relu(add0)

        rl1 = self.bn_x1(self.relation_non_local(gnet, 1))
        add1 = torch.add(gnet, rl1)
        gnet = self.bn_x1_relu(add1)

        h1_0 = self.h1_0(gnet)
        h1_1 = self.h1_1(h1_0)
        h1_3 = self.h1_3(h1_1)
        add = torch.add(h1_0, h1_3)
        gnet = self.relu(add)

        rl0 = self.bn_x2(self.relation_non_local(gnet, 2))
        add0 = torch.add(gnet, rl0)
        gnet = self.bn_x2_relu(add0)

        rl1 = self.bn_x3(self.relation_non_local(gnet, 3))
        add1 = torch.add(gnet, rl1)
        gnet = self.bn_x3_relu(add1)

        bbox_pred = self.bbox0.to(device)(gnet)
        bbox_pred = torch.permute(bbox_pred, (0, 2, 3, 1))
        bbox_pred = self.bbox1.to(device)(torch.reshape(bbox_pred, (BATCH, OBJ, P)))

        cls_score = self.cls0.to(device)(gnet)
        cls_score = torch.permute(cls_score, (0, 2, 3, 1))
        cls_prob = self.cls1.to(device)(torch.reshape(cls_score, (BATCH, OBJ, LEN_CLASSES)))

        final_pred = torch.cat((bbox_pred, cls_prob), dim=-1)
        return modify_tensor(cls_prob, final_pred)


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
            torch.nn.Linear(in_features=36864, out_features=512),  # 5376
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


class LayoutGAN:
    def __init__(self):
        self.gen = Generator().to(device)
        self.disc = Discriminator().to(device)

    def train(self):
        n_epoch = 5001
        start_epoch = 0
        counter = 0

        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=10e-5, amsgrad=True)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=10e-5, amsgrad=True)

        main_criterion = torch.nn.BCELoss()
        count_criterion = torch.nn.BCELoss()
        text_criterion = torch.nn.BCELoss()

        row_data = np.load('./data/data.npy')
        dataloader = DataLoader(TensorDataset(torch.Tensor(row_data)), batch_size=BATCH, shuffle=True, drop_last=True)
        BEST_MODEL_FAIL_OBJECTS = 64

        checkpoint_file = Path(CHECKPOINT_PATH)
        if checkpoint_file.exists():
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            start_epoch = checkpoint['start_epoch']
            BEST_MODEL_FAIL_OBJECTS = checkpoint['best_fail']
            counter = start_epoch * 49  # batch count
            self.gen.load_state_dict(checkpoint['generator'])
            self.disc.load_state_dict(checkpoint['discriminator'])
            gen_opt.load_state_dict(checkpoint['gen_opt'])
            disc_opt.load_state_dict(checkpoint['disc_opt'])

        def gen_noise():
            batch_z_bbox = np.random.normal(0.5, 0.15, (BATCH, OBJ, P))
            batch_z_cls = np.random.uniform(0, 1, (BATCH, OBJ, LEN_CLASSES))
            batch_z = np.concatenate([batch_z_bbox, batch_z_cls], axis=-1)
            noise = torch.tensor(batch_z, dtype=torch.float).to(device)
            c = torch.tensor(batch_z_cls, dtype=torch.float).to(device)
            return modify_tensor(c, noise)

        def get_fake_pred(should_detach):
            noise = gen_noise()
            if should_detach:
                with torch.no_grad():
                    fake_tensor = self.gen(noise)
            else:
                fake_tensor = self.gen(noise)
            fake_pred = self.disc(fake_tensor)
            return fake_tensor, fake_pred

        for epoch in range(start_epoch, n_epoch):
            print("Epoch ", epoch)
            idx = 0
            for data in tqdm(dataloader):
                real_layout_tensor = data[0].to(device)

                disc_opt.zero_grad()

                d_real = self.disc(real_layout_tensor)
                generated_tensor, d_generated = get_fake_pred(should_detach=True)
                zeros = torch.zeros_like(d_generated, requires_grad=True).to(device)
                ones = torch.ones_like(d_generated, requires_grad=True).to(device)
                d_loss = main_criterion(d_generated, zeros) + main_criterion(d_real, ones)
                disc_loss = d_loss

                _, disc_fake_pred = get_fake_pred(should_detach=True)
                count_fake = count_non_zero(_)
                count_real = count_non_zero(real_layout_tensor)
                zeros = torch.zeros_like(count_fake, requires_grad=True).to(device)
                ones = torch.ones_like(count_fake, requires_grad=True).to(device)
                disc_loss_count = count_criterion(count_fake, ones) + count_criterion(count_real, zeros)
                disc_loss += disc_loss_count

                _, disc_fake_pred = get_fake_pred(should_detach=True)
                tmp = get_text_loss(_)
                text_fake = torch.tensor(tmp, requires_grad=True).to(device)
                tmp = get_text_loss(real_layout_tensor)
                text_real = torch.tensor(tmp, requires_grad=True).to(device)
                zeros = torch.zeros_like(text_fake, requires_grad=True).to(device)
                ones = torch.ones_like(text_fake, requires_grad=True).to(device)
                disc_loss_text = 3 * (text_criterion(text_fake, ones) + text_criterion(text_real, zeros))
                disc_loss += disc_loss_text

                disc_loss.backward()
                disc_opt.step()

                if (idx + 1) % 1 == 0:
                    gen_opt.zero_grad()

                    _, disc_fake_pred = get_fake_pred(should_detach=False)
                    ones = torch.ones_like(disc_fake_pred, requires_grad=True)
                    g_loss = main_criterion(disc_fake_pred, ones)
                    gen_loss = g_loss

                    _, disc_fake_pred = get_fake_pred(should_detach=False)
                    count = count_non_zero(_)
                    zeros = torch.zeros_like(count, requires_grad=True).to(device)
                    gen_loss_count = count_criterion(count, zeros)
                    gen_loss += gen_loss_count

                    _, disc_fake_pred = get_fake_pred(should_detach=False)
                    tmp = get_text_loss(_)
                    text_losses = torch.tensor(tmp, requires_grad=True).to(device)
                    zeros = torch.zeros_like(text_losses, requires_grad=True).to(device)
                    gen_loss_text = 3 * text_criterion(text_losses, zeros)
                    gen_loss += gen_loss_text

                    gen_loss.backward()
                    gen_opt.step()

                if counter % 49 == 0:
                    res_tensor, _ = get_fake_pred(should_detach=True)
                    count = count_non_zero(res_tensor)
                    cnt = torch.count_nonzero(count)
                    if cnt.item() <= 32 or cnt.item() < BEST_MODEL_FAIL_OBJECTS:
                        BEST_MODEL_FAIL_OBJECTS = min(BEST_MODEL_FAIL_OBJECTS, cnt.item())
                        torch.save({'start_epoch': epoch + 1,
                                    'best_fail': BEST_MODEL_FAIL_OBJECTS,
                                    'generator': self.gen.state_dict(),
                                    'discriminator': self.disc.state_dict(),
                                    'gen_opt': gen_opt.state_dict(),
                                    'disc_opt': disc_opt.state_dict()
                                    }, BEST_CHECKPOINT_PATH)

                    torch.save({'start_epoch': epoch + 1,
                                'best_fail': BEST_MODEL_FAIL_OBJECTS,
                                'generator': self.gen.state_dict(),
                                'discriminator': self.disc.state_dict(),
                                'gen_opt': gen_opt.state_dict(),
                                'disc_opt': disc_opt.state_dict()
                                }, CHECKPOINT_PATH)

                    if counter % 10 == 0:
                        image = layout_bbox(res_tensor, WIDTH, HEIGHT)
                        write_json(res_tensor, '{:02d}_{:04d}'.format(epoch, idx))
                        size = image_manifold_size(list(image.size())[0])
                        path = './samples/train_{:02d}_{:04d}.jpg'.format(epoch, idx)
                        save_npy_img(image.detach().cpu().numpy(), size, path)

                idx += 1
                counter += 1
