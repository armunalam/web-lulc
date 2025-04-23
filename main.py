import math
import requests
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
import base64
from PIL import Image
# from predict_lulc import get_google_map
from bing_map import get_bing_map
# from predict_lulc import predict as predict_lulc
from UNet_LULC.predict_lulc_unet import predict as predict_lulc_unet
from UnimatchV2_LULC.predict_lulc_unimatchv2 import predict as predict_lulc_unimatchv2
from UnimatchV2_LULC.predict_brickfield_unimatchv2 import predict as predict_brickfield_unimatch
from BrickField.predict_brickfield import predict as predict_brickfield
# from contextlib import asynccontextmanager
from typing import AsyncGenerator
# import rasterio

import torch
from torch import nn

from UNet_LULC.UNet.unet.unet_model import UNet as UNet

from time import perf_counter
# from BrickField.UNet.unet.unet_model import UNet as BrickField


# input_tif_locations = None


# def load_raster():

# device = None
# unet = None
# brickfield = None


# async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
#     global device
#     global unet
#     global brickfield

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     unet = UNet(n_channels=3, n_classes=6, bilinear=False)

#     # Wrap the unet with DataParallel to use multiple GPUs
#     if torch.cuda.device_count() > 1:
#         # print(f"Using {torch.cuda.device_count()} GPUs!")
#         unet = nn.DataParallel(unet)

#     unet.to(device)

#     checkpoint = torch.load(
#         'UNet_LULC/checkpoints/best_model.pth', weights_only=False)
#     # Adjust for multi-GPU loading if needed
#     if torch.cuda.device_count() > 1:
#         new_state_dict = {}
#         for key, value in checkpoint["model_state_dict"].items():
#             new_key = "module." + \
#                 key if not key.startswith("module.") else key
#             new_state_dict[new_key] = value
#         unet.load_state_dict(new_state_dict)
#     else:
#         unet.load_state_dict(checkpoint["model_state_dict"])

#     unet.eval()

#     brickfield = BrickField(n_channels=3, n_classes=2, bilinear=False)
#     if torch.cuda.device_count() > 1:
#         # print(f"Using {torch.cuda.device_count()} GPUs!")
#         brickfield = nn.DataParallel(brickfield)
#     brickfield.to(device)

#     checkpoint = torch.load(
#         'BrickField/checkpoints/best_model.pth', weights_only=False)

#     if torch.cuda.device_count() > 1:
#         new_state_dict = {}
#         for key, value in checkpoint["model_state_dict"].items():
#             new_key = "module." + key if not key.startswith("module.") else key
#             new_state_dict[new_key] = value
#         brickfield.load_state_dict(new_state_dict)
#     else:
#         brickfield.load_state_dict(checkpoint["model_state_dict"])

#     brickfield.eval()

#     yield  # The app runs here

# #     # Cleanup: Close dataset and MemoryFile
#     if unet:
#         unet.close()
#     if brickfield:
#         brickfield.close()

# async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
#     # load_raster()
#     """Load the raster dataset into memory once."""
#     # global bing_2023
#     # global bing_2019
#     global input_tif_locations
#     # image_path = "large_image.tif"  # Change this to your actual file path

#     # Open the image once and store it in memory
#     with rasterio.open('/home/skeptic/bd/Bangladesh_BING.tif') as src:
#         memfile = rasterio.MemoryFile()
#         with memfile.open(**src.profile) as dst:
#             dst.write(src.read())  # Copy all bands
#         bing_2023 = memfile.open()  # Reopen in-memory dataset

#     with rasterio.open('/opt/models/dhaka_input.tif') as src:
#         memfile = rasterio.MemoryFile()
#         with memfile.open(**src.profile) as dst:
#             dst.write(src.read())  # Copy all bands
#         bing_2019 = memfile.open()  # Reopen in-memory dataset

#     input_tif_locations = {
#         '2023': bing_2023,
#         # '2019': 'bing_rgb/dhaka.tif',
#         '2019': bing_2019,
#     }

#     yield  # The app runs here

#     # Cleanup: Close dataset and MemoryFile
#     if input_tif_locations:
#         input_tif_locations.close()

app = FastAPI()
# app = FastAPI(lifespan=lifespan)

app.mount('/data', StaticFiles(directory='data'), name='data')
# app.mount('/imagedata', StaticFiles(directory='imagedata'), name='imagedata')
app.mount('/dependencies', StaticFiles(directory='dependencies'),
          name='dependencies')
app.mount('/scripts', StaticFiles(directory='scripts'), name='scripts')
app.mount('/styles', StaticFiles(directory='styles'), name='styles')

template = Jinja2Templates(directory='templates')    #


@app.get('/')
def home(request: Request):
    return template.TemplateResponse('index.html', {'request': request})


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a Base64 string."""
    img_io = io.BytesIO()
    image.save(img_io, format=format)
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode("utf-8")


# @app.on_event("startup")
# def startup_event():

# bing_2023 = None
# bing_2019 = None

def compare_year(request, min_lon, min_lat, max_lon, max_lat, service):
    global image_2023
    global image_2019
    image_2023 = get_bing_map(min_lat, min_lon, max_lat, max_lon, '2023')
    image_2019 = get_bing_map(min_lat, min_lon, max_lat, max_lon, '2019')
    table_2023 = None
    table_2019 = None

    if True or image_2023 is None or image_2019 is None:
        if service == 'LULC (Unimatch V2)':
            input_image_2023, output_image_2023, table_2023 = predict_lulc_unimatchv2(
                image_2023)
            input_image_2019, output_image_2019, table_2019 = predict_lulc_unimatchv2(
                image_2019)
        elif service == 'Brickfield (Unimatch V2)':
            input_image_2023, output_image_2023, table_2023 = predict_brickfield_unimatch(
                image_2023)
            input_image_2019, output_image_2019, table_2019 = predict_brickfield_unimatch(
                image_2019)
        elif service == 'LULC (Unet)':
            input_image_2023, output_image_2023, table_2023 = predict_lulc_unet(
                image_2023)
            input_image_2019, output_image_2019, table_2019 = predict_lulc_unet(
                image_2019)
        elif service == 'Brickfield':
            input_image_2023, output_image_2023, table_2023 = predict_brickfield(
                image_2023)
            input_image_2019, output_image_2019, table_2019 = predict_brickfield(
                image_2019)

        # input_base64_2023 = pil_to_base64(input_image_2023, format='JPEG')
        # input_base64_2019 = pil_to_base64(input_image_2019, format='JPEG')
        # output_base64_2023 = pil_to_base64(output_image_2023)
        # output_base64_2019 = pil_to_base64(output_image_2019)

        return template.TemplateResponse('output_compare.html', {'request': request,
                                                                 'input_image_data': pil_to_base64(input_image_2023, format='JPEG'),
                                                                 'output_image_data': pil_to_base64(output_image_2023),
                                                                 'input_image_data_prev': pil_to_base64(input_image_2019, format='JPEG'),
                                                                 'output_image_data_prev': pil_to_base64(output_image_2019),
                                                                 'table': table_2023,
                                                                 'table_prev': table_2019,
                                                                 'min_lon': min_lon,
                                                                 'min_lat': min_lat,
                                                                 'max_lon': max_lon,
                                                                 'max_lat': max_lat,
                                                                 })

    else:
        print('Error')


@app.post('/')
def submit(request: Request,
           min_lat: float = Form(...),
           min_lon: float = Form(...),
           max_lat: float = Form(...),
           max_lon: float = Form(...),
           service: str = Form(...),
           year: str = Form(...)):

    # print(service)

    min_lon, min_lat = float(min_lon), float(min_lat)
    max_lon, max_lat = float(max_lon), float(max_lat)
    # print(min_lat, min_lon, max_lat, max_lon)
    # min_lon, min_lat = 90.30638136, 23.78550914
    # max_lon, max_lat = 90.53348936, 23.88819931

    # image = get_google_map(min_lat, min_lon, max_lat, max_lon)

    if year == 'compare':
        return compare_year(request, min_lon, min_lat, max_lon, max_lat, service)

    image = get_bing_map(min_lat,
                         min_lon, max_lat, max_lon, year)

    table = None

    if image is not None:
        time_start = perf_counter()
        if service == 'LULC (Unimatch V2)':
            input_image, output_image, table = predict_lulc_unimatchv2(image)
        elif service == 'Brickfield (Unimatch V2)':
            input_image, output_image, table = predict_brickfield_unimatch(
                image)
        elif service == 'LULC (Unet)':
            # input_image, output_image = predict_lulc(image)
            input_image, output_image, table = predict_lulc_unet(image)
        elif service == 'Brickfield':
            input_image, output_image, table = predict_brickfield(image)
        time_stop = perf_counter()
        print(f'Time elapsed during inference: {time_stop - time_start}')

        # buffered = io.BytesIO()
        # input_image.save(buffered, format="JPEG", optimize=True)

        # input_image = Image.open(buffered)

        # map_image = fetch_map_image(
        #     (max_lat, min_lon), (min_lat, max_lon), zoom=14)

        input_base64 = pil_to_base64(input_image, format='JPEG')
        output_base64 = pil_to_base64(output_image)
        # map_base64 = pil_to_base64(map_image)

        # input_image.save('trash/input_image_123.png')
        # output_image.save('imagedata/output_image.png')
    else:
        print('Error')

    # table = {
    #     'C': (1, 2),
    #     'E': (3, 4),
    #     'CAT': (9, 12),
    # }

    return template.TemplateResponse('output.html', {'request': request,
                                                     'input_image_data': input_base64,
                                                     'output_image_data': output_base64,
                                                     #   'map_base64': map_base64,
                                                     'min_lon': min_lon,
                                                     'min_lat': min_lat,
                                                     'max_lon': max_lon,
                                                     'max_lat': max_lat,
                                                     'table': table,
                                                     })
    # {"request": request,  "predict_result": predict_result})

    # @app.post('/')
    # async def home(request: Request, topLeftLattitude: float = Form(...), topLeftLongitude: float = Form()):
    #     print(topLeftLattitude, topLeftLongitude)
    #     data = {
    #         't_left_lang': topLeftLattitude,
    #         't_left_long': topLeftLongitude
    #     }
    #     return template.TemplateResponse('index.html', {'request': request, 'data': data})
