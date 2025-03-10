import rasterio
from rasterio.warp import transform
import numpy as np
from PIL import Image


def get_bing_map(min_lat: float, min_lon: float, max_lat: float, max_lon: float, year: str = '2023') -> Image.Image | None:
    # Input .tif file path
    # input_tif = 'bing_rgb/dhaka.tif'
    # input_tif = '/home/skeptic/bd/Bangladesh_BING.tif'
    input_tif_locations = {
        '2023': '/home/skeptic/bd/Bangladesh_BING.tif',
        # '2019': 'bing_rgb/dhaka.tif',
        # '2019': '/opt/models/dhaka_input.tif',
        '2019': '/opt/models/BD_2019_Full.tif',
        # '2019': '/opt/models/Bangladesh_2019_64Districts_Merged.tif',
    }

    input_tif = input_tif_locations.get(year)

    # if year == '2019':
    #     input_tif = 'bing_rgb/dhaka.tif'
    # elif year == '2023':
    #     input_tif = '/home/skeptic/bd/Bangladesh_BING.tif'
    # else:
    #     raise ValueError(
    #         f'Data for the year {year} does not exist in the server.')

    # Open the raster file
    with rasterio.open(input_tif) as src:
        # Print raster metadata for debugging
        print(f'Raster CRS: {src.crs}')
        print(f'Raster bounds: {src.bounds}')

        # If raster is not in lat/lon (EPSG:4326), convert coordinates
        if src.crs.to_string() != 'EPSG:4326':
            print('Transforming coordinates to raster CRS...')
            transformed_x, transformed_y = transform(
                'EPSG:4326', src.crs,
                [min_lon, max_lon], [min_lat, max_lat]
            )
            min_x, max_x = transformed_x
            min_y, max_y = transformed_y
        else:
            min_x, min_y, max_x, max_y = min_lon, min_lat, max_lon, max_lat

        print(
            f'Converted coordinates: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}')

        # Ensure coordinates are inside raster bounds
        if (min_x < src.bounds.left or max_x > src.bounds.right or
                min_y < src.bounds.bottom or max_y > src.bounds.top):
            raise ValueError('Bounding box is outside raster extent.')

        # Convert coordinates to pixel indices
        # Top-left (note: max_y for row)
        min_row, min_col = src.index(min_x, max_y)
        max_row, max_col = src.index(max_x, min_y)  # Bottom-right

        print(
            f'Pixel indices: min_row={min_row}, min_col={min_col}, max_row={max_row}, max_col={max_col}')

        # Ensure valid cropping dimensions
        if min_row >= max_row or min_col >= max_col:
            raise ValueError(
                'Invalid crop dimensions. Check coordinate transformation.')

        # Read cropped raster data
        cropped_data = src.read(
            window=((min_row, max_row), (min_col, max_col)))

        # Normalize for display
        if cropped_data.shape[0] == 1:  # Single-band image
            img_array = cropped_data[0]
        else:  # Multi-band image
            img_array = np.moveaxis(cropped_data, 0, -1)

        img_array = (img_array - img_array.min()) / \
            (img_array.max() - img_array.min()) * 255
        img_array = img_array.astype(np.uint8)

        print(input_tif)

        Image.fromarray(img_array).save('trash/full_image2.png')

        return img_array

        # Convert to PIL image and show
        # img = Image.fromarray(img_array)
        # return img


if __name__ == '__main__':
    # Define bounding box (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat = 90.30638136, 23.78550914
    max_lon, max_lat = 90.53348936, 23.88819931

    img = get_bing_map(min_lat, min_lon, max_lat, max_lon)
    img.save('bing_rgb/test_bing.png')
