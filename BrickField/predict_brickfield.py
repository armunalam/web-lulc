import torch
import torch.nn as nn
import torchvision
from PIL import Image
import torchvision.transforms.functional
import numpy as np
from BrickField.UNet.unet.unet_model import UNet
from utils.utils import make_patches, unpatchify, decode_segmap, LABELS, COLORS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_channels=3, n_classes=2, bilinear=False)

unimatch_path = '/opt/models/exp/brickfield_unet_0.pth'
checkpoint = torch.load(
    unimatch_path, map_location='cpu', weights_only=False)
new_state_dict = {}
for k, v in checkpoint['model_state_dict'].items():
    new_key = k.replace('module.', '')
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)

model = torch.compile(model, backend="inductor", dynamic=False)
model.to(device)


def predict(image: np.ndarray, patch_size: int = 1500) -> tuple[Image.Image, Image.Image, list]:
    image = image[:, :, :3]
    original_image = image
    patch_images, image_size = make_patches(image, patch_size)
    size_y, size_x, _, p_s_1, p_s_2, _ = patch_images.shape
    patch_images = patch_images.reshape(size_x * size_y, p_s_1, p_s_2, 3)

    output_images = []
    count_array = np.zeros(2, dtype=np.int_)

    model.eval()

    for image in patch_images:
        image = torchvision.transforms.functional.to_tensor(
            image).to(device)
        image = torchvision.transforms.functional.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).reshape(1, 3, patch_size, patch_size)
        image = image.to(dtype=torch.float32)

        output = model(image)
        output = output.detach().max(dim=1)[1].cpu().numpy().squeeze(axis=0)

        unique, counter = np.unique(output, return_counts=True)
        count_temp = np.zeros(2, dtype=np.int_)
        count_temp[unique] = counter
        count_array += count_temp

        output = decode_segmap(output, service='brickfield')
        output = Image.fromarray(output)
        output_images.append(output)

    output_images = np.stack(output_images, axis=0).reshape(
        size_y, size_x, 1, p_s_1, p_s_2, 3)

    output_image = unpatchify(output_images, image_size)
    output_image = Image.fromarray(output_image)

    labels = LABELS.get('brickfield')
    colors = [str(color) for color in COLORS.get('brickfield')]
    area = [f'{val * 4.92e-6:,.2f}' for val in count_array]
    max_pixel = np.sum(count_array) or 1
    count_array = [f'{val / max_pixel * 100:.2f}%' for val in count_array]
    table = list(zip(labels, list(count_array), area, colors))

    torch.cuda.empty_cache()

    return Image.fromarray(original_image), output_image, table


if __name__ == '__main__':
    image = Image.open('test_3.png')
    image = np.asarray(image)
    # image = np.transpose(image, (2, 0, 1))

    predict(image)
