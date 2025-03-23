# import argparse
# from copy import deepcopy
# import logging
import os
# import pprint

import torch
import torchvision
import numpy as np
from PIL import Image
# from torch import nn
# import torch.backends.cudnn as cudnn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import yaml

# from dataset.semi import SemiDataset
from UnimatchV2_LULC.model.semseg.dpt import DPT
# from supervised import evaluate
# from util.classes import CLASSES
# from util.ohem import ProbOhemCrossEntropy2d
# from util.utils import count_params, init_log, AverageMeter
# from util.dist_helper import setup_distributed

# from tqdm import tqdm


device = 'cuda'

model = DPT(
    **{'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768],
       'nclass': 6})
# state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
# model.backbone.load_state_dict(state_dict)

# exp/coco/unimatch_v2/dinov2_base/bd/best_after.pth
# unimatch_path = 'exp/best_after.pth'

model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model.to(device)

unimatch_path = '/home/skeptic/web-lulc/UnimatchV2_LULC/exp/best_after.pth'
if os.path.exists(unimatch_path):
    checkpoint = torch.load(
        unimatch_path, map_location='cpu', weights_only=False)

    new_state_dict = {}
    for k, v in checkpoint['model'].items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)


def make_patches(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Splits a PIL image into patches of size (patch_size, patch_size),
    zero-padding the remaining parts if needed.

    Args:
        image (PIL.Image.Image): Input image.
        patch_size (int): Size of each patch (square).

    Returns:
        numpy.ndarray: Array of patches with shape 
        (num_patches_vertical, num_patches_horizontal, 1, patch_size, patch_size, 3)
    """
    # image = image.convert('RGB')
    image_size = image.shape[1], image.shape[0]
    # image_np = np.array(image)
    image_np = image
    h, w, c = image_np.shape

    num_patches_vertical = (h + patch_size - 1) // patch_size
    num_patches_horizontal = (w + patch_size - 1) // patch_size

    patches = np.zeros(
        (num_patches_vertical, num_patches_horizontal, 1, patch_size, patch_size, c),
        dtype=image_np.dtype
    )

    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            y_start = i * patch_size
            x_start = j * patch_size
            patch = image_np[y_start:y_start +
                             patch_size, x_start:x_start + patch_size]

            # Handle padding if needed
            padded_patch = np.zeros(
                (patch_size, patch_size, c), dtype=image_np.dtype)
            padded_patch[:patch.shape[0], :patch.shape[1], :] = patch

            patches[i, j, 0] = padded_patch

    return patches, image_size


def unpatchify(patches: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """
    Reconstructs the original image from patches.

    Args:
        patches (numpy.ndarray): Array of patches with shape 
            (num_patches_vertical, num_patches_horizontal, 1, patch_size, patch_size, 3)
        original_height (int): Height of the original image.
        original_width (int): Width of the original image.

    Returns:
        PIL.Image.Image: Reconstructed image.
    """
    original_width, original_height = original_size
    num_patches_vertical, num_patches_horizontal, _, patch_size, _, _ = patches.shape

    # Calculate full reconstructed size with padding
    full_height = num_patches_vertical * patch_size
    full_width = num_patches_horizontal * patch_size

    # Create an empty array to hold the reconstructed (possibly padded) image
    reconstructed = np.zeros((full_height, full_width, 3), dtype=patches.dtype)

    # Place each patch back into the image
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            y_start = i * patch_size
            x_start = j * patch_size
            reconstructed[y_start:y_start + patch_size,
                          x_start:x_start + patch_size] = patches[i, j, 0]

    # Remove any padding to match the original image size
    reconstructed_cropped = reconstructed[:original_height, :original_width]

    return reconstructed_cropped


def decode_segmap(image: np.ndarray, nc=6) -> np.ndarray:
    # 0=Unrecognized
    # 1=Farmland, 2=Water, 3=Forest, 4=Built-Up, 5=Meadow
    label_colors = np.array(
        # [(0, 0, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)])
        [(0, 0, 0), (0, 255, 0), (0, 0, 255),
         (0, 255, 255), (255, 0, 0), (255, 255, 0)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def predict(image: np.ndarray, patch_size: int = 518) -> tuple[Image.Image, Image.Image]:

    image = image[:, :, :3]
    original_image = image

    patch_images, image_size = make_patches(image, patch_size)

    size_y, size_x, _, p_s_1, p_s_2, channels = patch_images.shape
    patch_images = patch_images.reshape(
        size_x * size_y, p_s_1, p_s_2, channels)

    output_images = []
    count_array = np.zeros(6, dtype=np.int_)

    for index, image in enumerate(patch_images):
        model.eval()

        image = torchvision.transforms.functional.to_tensor(
            image).to(device)
        image = torchvision.transforms.functional.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).reshape(1, channels, patch_size, patch_size)

        image = image.to(dtype=torch.float32)

        output = model(image)

        output = output.detach().max(dim=1)[1].cpu().numpy().squeeze(axis=0)

        unique, counter = np.unique(output, return_counts=True)
        count_temp = np.zeros(6, dtype=np.int_)

        count_temp[unique] = counter

        count_array += count_temp

        output = decode_segmap(output)

        output = Image.fromarray(output)

        output_images.append(output)

    output_images = np.stack(output_images, axis=0).reshape(
        size_y, size_x, 1, p_s_1, p_s_2, channels)
    # print(output_images.shape)

    output_image = unpatchify(output_images, image_size)
    # print(output_image.shape)
    img = Image.fromarray(output_image)
    # width, height = img.size
    # image = image.crop((0, 0, width, height))
    # output.save('final_unet.png')

    labels = ['Farmland',
              'Water', 'Forest', 'Built-Up', 'Meadow']
    colors = ['0, 255, 0', '0, 0, 255',
              '0, 255, 255', '255, 0, 0', '255, 255, 0']
    # print(count_array)
    area = [f'{val * 4.92e-6:,.2f}' for val in count_array[1:]]
    max_pixel = np.sum(count_array[1:]) or 1
    # print(max_pixel)
    count_array = [f'{val / max_pixel * 100:.2f}%' for val in count_array[1:]]
    table = list(zip(labels, list(count_array), area, colors))
    # print(count_array)

    torch.cuda.empty_cache()

    return Image.fromarray(original_image), img, table


if __name__ == '__main__':
    img = Image.open('/home/skeptic/web-lulc/trash/full_image.png')

    pred = predict(np.asarray(img))[1]
    pred.save('/home/skeptic/web-lulc/trash/another_test_5.png')
