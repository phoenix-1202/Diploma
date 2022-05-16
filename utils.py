import torch
import numpy as np
from PIL import Image
import imageio as imageio
import imageio.core.util
import json


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


def save_npy_img(images, size, image_path):
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [0, 0, 128],
                                 [128, 128, 0],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [0, 0, 0]], dtype='uint8').flatten()

    cls_map_all = np.zeros((images.shape[0], images.shape[1], images.shape[2], 3), dtype=np.uint8)

    for img_ind in range(images.shape[0]):
        binary_mask = images[img_ind, :, :, :]

        # Add background
        image_sum = np.sum(binary_mask, axis=-1)
        ind = np.where(image_sum == 0)
        image_bk = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float32)
        image_bk[ind] = 1.0
        image_bk = np.reshape(image_bk, (binary_mask.shape[0], binary_mask.shape[1], 1))
        binary_mask = np.concatenate((image_bk, binary_mask), axis=-1)

        cls_map = np.argmax(binary_mask, axis=2)

        cls_map_img = Image.fromarray(cls_map.astype(np.uint8))
        cls_map_img.putpalette(palette)
        cls_map_img = cls_map_img.convert('RGB')
        cls_map_all[img_ind, :, :, :] = np.array(cls_map_img)

    cls_map_all = np.ndarray.squeeze(merge(cls_map_all, size))
    return imageio.imwrite(image_path, cls_map_all)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


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


def write_json(big_image, name=""):
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

    out_file = open("./samples/json/train_" + name + ".json", "w")
    out_file.truncate(0)
    json.dump(big_final, out_file)
    out_file.close()


def get_area(rect1, rect2):
    x1, y1, w1, h1 = rect1[0], rect1[1], rect1[2], rect1[3]
    x2, y2, w2, h2 = rect2[0], rect2[1], rect2[2], rect2[3]
    ld1 = (x1 - w1 / 2, y1 - h1 / 2)
    ru1 = (x1 + w1 / 2, y1 + h1 / 2)
    ld2 = (x2 - w2 / 2, y2 - h2 / 2)
    ru2 = (x2 + w2 / 2, y2 + h2 / 2)
    left = max(ld1[0], ld2[0])
    top = min(ru1[1], ru2[1])
    right = min(ru1[0], ru2[0])
    bottom = max(ld1[1], ld2[1])

    width = right - left
    height = top - bottom
    if width < 0 or height < 0:
        return 0.
    return width * height / min(h1 * w1, h2 * w2)
