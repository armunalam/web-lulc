import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

LABELS = {
    'lulc': ['Farmland', 'Water', 'Forest', 'Built-Up', 'Meadow'],
    'brickfield': ['Non-Brickfield Area', 'Brickfield']
}

COLORS = {
    'lulc': [(0, 0, 0), (0, 255, 0), (0, 0, 255),
             (0, 255, 255), (255, 0, 0), (255, 255, 0)],
    'brickfield': [(0, 0, 0), (255, 0, 0)],
}

n_class = {
    'lulc': 6,
    'brickfield': 2
}


# def _create_patch(args):
#     i, j, patch_size, image_np = args
#     h, w, c = image_np.shape
#     y_start = i * patch_size
#     x_start = j * patch_size

#     patch = image_np[y_start:y_start +
#                      patch_size, x_start:x_start + patch_size]

#     padded_patch = np.zeros((patch_size, patch_size, c), dtype=image_np.dtype)
#     padded_patch[:patch.shape[0], :patch.shape[1], :] = patch

#     return (i, j, padded_patch)


# def make_patches(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
#     print('Making patches...')
#     image_size = image.shape[1], image.shape[0]
#     image_np = image
#     h, w, c = image_np.shape

#     num_patches_vertical = (h + patch_size - 1) // patch_size
#     num_patches_horizontal = (w + patch_size - 1) // patch_size

#     patches = np.zeros(
#         (num_patches_vertical, num_patches_horizontal, 1, patch_size, patch_size, c),
#         dtype=image_np.dtype
#     )

#     args = [
#         (i, j, patch_size, image_np)
#         for i in range(num_patches_vertical)
#         for j in range(num_patches_horizontal)
#     ]

#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#         futures = [executor.submit(_create_patch, arg) for arg in args]
#         for future in as_completed(futures):
#             i, j, patch = future.result()
#             patches[i, j, 0] = patch

#     print('Patches created.')

#     return patches, image_size


# def _place_patch(args):
#     i, j, patch, patch_size = args
#     y_start = i * patch_size
#     x_start = j * patch_size
#     return (y_start, x_start, patch)


# def unpatchify(patches: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
#     original_width, original_height = original_size
#     num_patches_vertical, num_patches_horizontal, _, patch_size, _, c = patches.shape

#     # Full (possibly padded) size
#     full_height = num_patches_vertical * patch_size
#     full_width = num_patches_horizontal * patch_size

#     # Initialize full reconstructed image
#     reconstructed = np.zeros((full_height, full_width, c), dtype=patches.dtype)

#     # Prepare tasks for multiprocessing
#     args = [
#         (i, j, patches[i, j, 0], patch_size)
#         for i in range(num_patches_vertical)
#         for j in range(num_patches_horizontal)
#     ]

#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#         futures = [executor.submit(_place_patch, arg) for arg in args]
#         for future in as_completed(futures):
#             y_start, x_start, patch = future.result()
#             reconstructed[y_start:y_start + patch_size,
#                           x_start:x_start + patch_size] = patch

#     # Crop to original size
#     reconstructed_cropped = reconstructed[:original_height, :original_width]
#     return reconstructed_cropped


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


def decode_segmap(image: np.ndarray, service='lulc') -> np.ndarray:
    nc = n_class.get(service, 'lulc')
    label_colors = np.array(COLORS.get(service, 'lulc'))

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
