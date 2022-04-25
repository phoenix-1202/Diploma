from utils import *
# import torch
import copy

OBJECTS = {'texts', 'object', 'images', 'circles', 'rotated_rectangles', 'logo', 'rectangles', 'stripes'}
CLASSES = {'Background': 0, 'Stripe': 1, 'Logo': 2, 'Circle': 3, 'Image': 4, 'Rectangle': 5, 'Text': 6}
PARAMETERS = {'x': 0, 'y': 1, 'w': 2, 'h': 3}
MAX_CNT = 11


def generate_data_from_json(file_path):
    row_data = ""
    with open(file_path) as f:
        for i in f.readlines():
            row_data += i

    data = json.loads(row_data)
    all_data = []

    for i in range(len(data)):
        all_data.append([])
        for k in data[i].keys():
            if k == 'name' or k == 'height' or k == 'width':
                continue
            if k == 'logo' or k == 'object':
                all_data[i].append(data[i][k])
            elif k != 'rotated_rectangles':
                for x in data[i][k]:
                    all_data[i].append(x)

    print(len(all_data))

    final_data = []
    for i in range(len(all_data)):
        final_data.append([])
        cur_count = [0 for _ in range(7)]
        for model in all_data[i]:
            arr = [0 for _ in range(11)]
            for (x, y) in model.items():
                if x == 'elementClass':
                    cur_count[CLASSES[y]] += 1
                    arr[CLASSES[y] + 4] = 1
                if x in PARAMETERS.keys():
                    arr[PARAMETERS[x]] = y
            arr[0] += arr[2] / 2
            arr[1] += arr[3] / 2
            final_data[i].append(arr)
        for j in range(MAX_CNT - len(final_data[i])):
            arr = [0 for _ in range(11)]
            final_data[i].append(arr)

    dataset = np.array(final_data, np.float32)
    dataset = np.stack(dataset)
    return dataset
    # np.save('data/data_hor.npy', dataset)


ANTI_BIG_CLASSES = {0: 'images', 1: 'stripes', 2: 'logo', 3: 'circles', 4: 'object', 5: 'rectangles', 6: 'texts'}
ANTI_CLASSES = {0: 'Background', 1: 'Stripe', 2: 'Logo', 3: 'Circle', 4: 'Image', 5: 'Rectangle', 6: 'Text'}

# Background -> images
# Text       -> texts
# Circle     -> circles
# Rectangle  -> rectangles
# Logo       -> logo          1
# Stripe     -> stripes
# Image      -> object        1

GLOBAL_COUNTER = 1


def write_json(big_image):
    global GLOBAL_COUNTER
    big_final = []
    for image in big_image:
        final = {"name": GLOBAL_COUNTER, "height": 1080, "width": 1080}
        GLOBAL_COUNTER += 1
        for i in range(image.shape[0]):
            x = float(image[i][0])
            y = float(image[i][1])
            w = float(image[i][2])
            h = float(image[i][3])
            class_id = -1
            maxi = 0
            for j in range(4, image[i].shape[0]):
                if image[i][j] > maxi:
                    maxi = image[i][j]
                    class_id = j
            if class_id == -1:
                continue
            k = ANTI_BIG_CLASSES[class_id - 4]

            if k == "logo" or k == "object":
                final[k] = {"elementClass": ANTI_CLASSES[class_id - 4], "x": x - w / 2, "y": y - h / 2, "w": w, "h": h}
            else:
                if k not in final.keys():
                    final[k] = []
                if (k == "texts" and len(final[k]) < 3) or (k != "texts"):
                    final[k].append(
                        {"elementClass": ANTI_CLASSES[class_id - 4], "x": x - w / 2, "y": y - h / 2, "w": w, "h": h})
        big_final.append(final)

    out_file = open("./data.json", "w")
    out_file.truncate(0)
    json.dump(big_final, out_file)
    out_file.close()


def vertical_mirror(data):
    res_data = copy.deepcopy(data)
    for tensor in res_data:
        for t in tensor:
            x = t[0]
            t[0] = 1 - x
    return res_data


def horizontal_mirror(data):
    res_data = copy.deepcopy(data)
    for tensor in res_data:
        for t in tensor:
            y = t[1]
            t[1] = 1 - y
    return res_data


def rotate_right(data):
    res_data = copy.deepcopy(data)
    for tensor in res_data:
        for t in tensor:
            x = t[0]
            y = t[1]
            t[0] = 1 - y
            t[1] = x
            t[2], t[3] = t[3], t[2]
    return res_data


def rotate_left(data):
    return rotate_right(horizontal_mirror(vertical_mirror(data)))


def add_noise(data):
    res_data = copy.deepcopy(data)
    for tensor in res_data:
        for t in tensor:
            rnd = np.random.normal(0, 0.02, 4)
            t[0] += rnd[0]
            t[1] += rnd[1]
            t[2] += rnd[2]
            t[3] += rnd[3]
    return res_data


my_data = generate_data_from_json("./res_add_hor.json")
new_part_data_rl = rotate_left(my_data)
new_part_data_rr = rotate_right(my_data)
new_part_data_hm = horizontal_mirror(my_data)
new_part_data_vm = vertical_mirror(my_data)
new_part_data_an = add_noise(my_data)
new_part_data = []

for i in range(my_data.shape[0]):
    new_part_data.append(my_data[i])
    new_part_data.append(new_part_data_rl[i])
    new_part_data.append(new_part_data_rr[i])
    new_part_data.append(new_part_data_hm[i])
    new_part_data.append(new_part_data_vm[i])
    new_part_data.append(new_part_data_an[i])
new_part_data = np.array(new_part_data, np.float32)
new_part_data = np.stack(new_part_data)
write_json(new_part_data)


# Check correctness

# new_data = generate_data_from_json("./data.json")
# my_image = layout_bbox(torch.Tensor(new_data[0:36]), 108, 108)
# size = image_manifold_size(list(my_image.size())[0])
# path = './samples/train_try.jpg'
# save_npy_img(my_image.detach().cpu().numpy(), size, path)

# WIDTH = 108
# HEIGHT = 108
#
# OBJ = 11
# LEN_CLASSES = 7
# P = 4
#
# BATCH = 64
#
#
# def layout_bbox(final_pred, output_height, output_width):
#     batch_size = final_pred.shape[0]
#     objects_cnt = final_pred.shape[1]
#     vector_size = final_pred.shape[2]
#     properties_cnt = vector_size - LEN_CLASSES
#
#     bbox_reg = final_pred[:, :, 0:properties_cnt]
#     cls_prob = final_pred[:, :, properties_cnt:vector_size]
#
#     bbox_reg = torch.reshape(bbox_reg, (batch_size, objects_cnt, properties_cnt))
#
#     x_c = bbox_reg[:, :, 0:1] * output_width
#     y_c = bbox_reg[:, :, 1:2] * output_height
#     w = bbox_reg[:, :, 2:3] * output_width
#     h = bbox_reg[:, :, 3:4] * output_height
#
#     x1 = x_c - 0.5 * w
#     x2 = x_c + 0.5 * w
#     y1 = y_c - 0.5 * h
#     y2 = y_c + 0.5 * h
#
#     xt = torch.reshape(torch.arange(0.0, output_width, 1.0), (1, 1, 1, -1))
#     xt = torch.reshape(torch.tile(xt, (batch_size, objects_cnt, output_height, 1)), (batch_size, objects_cnt, -1))
#
#     yt = torch.reshape(torch.arange(0.0, output_height, 1.0), (1, 1, -1, 1))
#     yt = torch.reshape(torch.tile(yt, (batch_size, objects_cnt, 1, output_width)), (batch_size, objects_cnt, -1))
#
#     x1_diff = torch.reshape(xt - x1, (batch_size, objects_cnt, output_height, output_width, 1))
#     y1_diff = torch.reshape(yt - y1, (batch_size, objects_cnt, output_height, output_width, 1))
#     x2_diff = torch.reshape(x2 - xt, (batch_size, objects_cnt, output_height, output_width, 1))
#     y2_diff = torch.reshape(y2 - yt, (batch_size, objects_cnt, output_height, output_width, 1))
#
#     f = torch.nn.ReLU()
#     x1_line = f(1.0 - torch.abs(x1_diff)) * \
#               torch.minimum(f(y1_diff), torch.ones_like(y1_diff)) * \
#               torch.minimum(f(y2_diff), torch.ones_like(y2_diff))
#     x2_line = f(1.0 - torch.abs(x2_diff)) * \
#               torch.minimum(f(y1_diff), torch.ones_like(y1_diff)) * \
#               torch.minimum(f(y2_diff), torch.ones_like(y2_diff))
#     y1_line = f(1.0 - torch.abs(y1_diff)) * \
#               torch.minimum(f(x1_diff), torch.ones_like(x1_diff)) * \
#               torch.minimum(f(x2_diff), torch.ones_like(x2_diff))
#     y2_line = f(1.0 - torch.abs(y2_diff)) * \
#               torch.minimum(f(x1_diff), torch.ones_like(x1_diff)) * \
#               torch.minimum(f(x2_diff), torch.ones_like(x2_diff))
#
#     xy = torch.cat((x1_line, x2_line, y1_line, y2_line), dim=-1)
#     xy_max = torch.amax(xy, dim=-1, keepdim=True)
#
#     spatial_prob = torch.multiply(torch.tile(xy_max, (1, 1, 1, 1, LEN_CLASSES)),
#                                   torch.reshape(cls_prob, (batch_size, objects_cnt, 1, 1, LEN_CLASSES)))
#     spatial_prob_max = torch.amax(spatial_prob, dim=1, keepdim=False)
#
#     return spatial_prob_max
