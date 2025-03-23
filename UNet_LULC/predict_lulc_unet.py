import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from UNet_LULC.UNet.unet.unet_model import UNet
from utils.utils import make_patches, unpatchify, decode_segmap, LABELS, COLORS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_channels=3, n_classes=6, bilinear=False)

unimatch_path = '/opt/models/exp/unet_0.pth'
checkpoint = torch.load(
    unimatch_path, map_location='cpu', weights_only=False)
new_state_dict = {}
for k, v in checkpoint['model_state_dict'].items():
    new_key = k.replace('module.', '')
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)

model = torch.compile(model, backend="inductor", dynamic=False)
model.to(device)


def predict(image: np.ndarray, patch_size: int = 513) -> tuple[Image.Image, Image.Image, list]:
    image = image[:, :, :3]
    original_image = image
    patch_images, image_size = make_patches(image, patch_size)
    size_y, size_x, _, p_s_1, p_s_2, channels = patch_images.shape
    patch_images = patch_images.reshape(
        size_x * size_y, p_s_1, p_s_2, channels)

    output_images = []
    count_array = np.zeros(6, dtype=np.int_)

    model.eval()

    for image in patch_images:
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

    output_image = unpatchify(output_images, image_size)
    output_image = Image.fromarray(output_image)

    labels = LABELS.get('lulc')
    colors = [str(color) for color in COLORS.get('lulc')[1:]]
    area = [f'{val * 4.92e-6:,.2f}' for val in count_array[1:]]
    max_pixel = np.sum(count_array[1:]) or 1
    count_array = [f'{val / max_pixel * 100:.2f}%' for val in count_array[1:]]
    table = list(zip(labels, list(count_array), area, colors))

    torch.cuda.empty_cache()

    return Image.fromarray(original_image), output_image, table


if __name__ == '__main__':
    image = Image.open(
        '/opt/datasets/unet/BingRGB/test/images/patch_00184.png')
    predict(image)
