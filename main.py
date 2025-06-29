import math
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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
from utils.utils import n_class as n_class_dict, decode_segmap, LABELS, change_3, generate_stripes, mask_to_rgb_with_stripes

from time import perf_counter
import pandas as pd


app = FastAPI()
# app = FastAPI(lifespan=lifespan)

app.mount('/data', StaticFiles(directory='data'), name='data')
# app.mount('/imagedata', StaticFiles(directory='imagedata'), name='imagedata')
app.mount('/dependencies', StaticFiles(directory='dependencies'),
          name='dependencies')
app.mount('/scripts', StaticFiles(directory='scripts'), name='scripts')
app.mount('/styles', StaticFiles(directory='styles'), name='styles')

template = Jinja2Templates(directory='templates')    #


district_data = pd.read_json('data/adm2.json')  # Replace with your actual file
district_data.set_index(['district'], inplace=True)
upazila_data = pd.read_json('data/adm3.json')  # Replace with your actual file
upazila_data.set_index(['district', 'upazila'], inplace=True)


@app.get('/')
def home(request: Request):
    return template.TemplateResponse('index.html', {'request': request})


def pil_to_base64(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert a PIL Image to a Base64 string."""
    img_io = io.BytesIO()
    image.save(img_io, format=format)
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode("utf-8")


def find_changes(old_image: np.ndarray, new_image: np.ndarray, n_class: int = 6, old_sat_image=None, new_sat_image=None) -> list[tuple[str, bytes]]:
    if n_class == 6:
        class_wise_change = [
            np.zeros(old_image.shape, dtype=np.uint8) for _ in range(n_class - 1)]
        builtup_change = np.zeros(old_image.shape, dtype=np.uint8)
    if n_class == 2:
        brickfield_change = np.zeros(old_image.shape, dtype=np.uint8)
    
    if n_class == 6:
        for i in range(n_class - 1):
            j = i + 1
            # had the class before but not anymore
            had = (old_image == j) & (new_image != j)
            # had the class before and still has it
            same = (old_image == j) & (new_image == j)
            # has the class now but not before
            new = (old_image != j) & (new_image == j)
            
            if n_class == 2:
                j = 4

            # class_wise_change[i][had] = j + n_class
            # class_wise_change[i][same] = j
            # class_wise_change[i][new] = j + n_class * 2
            class_wise_change[i][had] = new_image[had]
            # class_wise_change[i][same] = j + n_class * 2
            class_wise_change[i][same] = 12
            class_wise_change[i][new] = old_image[new] + 6
            
            striped_classes = {6, 7, 8, 9, 10, 11}
            result_classwise_change = mask_to_rgb_with_stripes(class_wise_change[i], change_3, striped_classes)
            class_wise_change[i] = pil_to_base64(result_classwise_change)
            
            # class_wise_change[i] = pil_to_base64(Image.fromarray(decode_segmap(
            #     class_wise_change[i], service='3-change'), mode='RGBA'))
            
    # if n_class == 6:
        # had = (old_image == 4) & (new_image != 4)
        # same = (old_image == 4) & (new_image == 4)
        # new = (old_image != 4) & (new_image == 4)

        # builtup_change[had] = 0
        # builtup_change[same] = 4
        # builtup_change[new] = old_image[new]
        # builtup_change = pil_to_base64(Image.fromarray(decode_segmap(
        #     builtup_change, service='lulc'), mode='RGBA'))
        
    if n_class == 2:
        had = (old_image == 1) & (new_image != 1)
        same = (old_image == 1) & (new_image == 1)
        new = (old_image != 1) & (new_image == 1)
        
        _, _, _, output_image_old = predict_lulc_unimatchv2(old_sat_image, compare=True)
        _, _, _, output_image_new = predict_lulc_unimatchv2(new_sat_image, compare=True)

        brickfield_change[had] = output_image_new[had]
        brickfield_change[same] = 12
        brickfield_change[new] = output_image_old[new] + 6
        striped_classes = {6, 7, 8, 9, 10, 11}
        result_brickfield_change = mask_to_rgb_with_stripes(brickfield_change, change_3, striped_classes)
        brickfield_change = pil_to_base64(result_brickfield_change)
        # brickfield_change = pil_to_base64(Image.fromarray(decode_segmap(
        #     brickfield_change, service='3-change'), mode='RGBA'))
            
    # service_type = {6: 'lulc', 2: 'brickfield'}
    labels = LABELS.get('lulc') if n_class == 6 else [''] * n_class

    
    if n_class == 6:
        class_wise_changes = zip(labels, class_wise_change)
        # return class_wise_changes, builtup_change, 'Built-Up Change'
        return class_wise_changes

    if n_class == 2:
        # return class_wise_changes, brickfield_change, 'Brickfield Change'
        return brickfield_change, 'Brickfield Change'
        
    return class_wise_changes


def compare_year(request, min_lon, min_lat, max_lon, max_lat, service):
    global image_2023
    global image_2019
    image_2023 = Image.fromarray(get_bing_map(
        min_lat, min_lon, max_lat, max_lon, '2023'))
    image_2019 = Image.fromarray(get_bing_map(
        min_lat, min_lon, max_lat, max_lon, '2019'))
    print('Size 2019:', image_2019.size)
    print('Size 2023:', image_2023.size)
    if image_2023.size[1] < image_2019.size[1]:
        image_2023 = image_2023.resize(
            (image_2019.size[0], image_2019.size[1]), Image.LANCZOS)
    elif image_2019.size[1] < image_2023.size[1]:
        image_2019 = image_2019.resize(
            (image_2023.size[0], image_2023.size[1]), Image.LANCZOS)
    print('After resize')
    print('Size 2019:', image_2019.size)
    print('Size 2023:', image_2023.size)
    image_2023 = np.asarray(image_2023)
    image_2019 = np.asarray(image_2019)
    table_2023 = None
    table_2019 = None
    output_image_2023_raw = None
    output_image_2019_raw = None
    class_wise_changes = None
    specific_class_changes = None
    change_title = None
    # builtup_changes = None
    # n_class = 0

    if image_2023.any() or image_2019.any():
        if service == 'LULC (Unimatch V2)':
            input_image_2023, output_image_2023, table_2023, output_image_2023_raw = predict_lulc_unimatchv2(
                image_2023, compare=True)
            input_image_2019, output_image_2019, table_2019, output_image_2019_raw = predict_lulc_unimatchv2(
                image_2019, compare=True)
            # n_class = n_class_dict.get('lulc')
            class_wise_changes = find_changes(output_image_2019_raw,
                                            output_image_2023_raw, 6)
            # class_wise_changes, specific_class_changes, change_title = find_changes(output_image_2019_raw,
            #                                 output_image_2023_raw, 6)
        elif service == 'Brickfield (Unimatch V2)':
            input_image_2023, output_image_2023, table_2023, output_image_2023_raw = predict_brickfield_unimatch(
                image_2023, compare=True)
            input_image_2019, output_image_2019, table_2019, output_image_2019_raw = predict_brickfield_unimatch(
                image_2019, compare=True)
            specific_class_changes, change_title = find_changes(output_image_2019_raw,
                                            output_image_2023_raw, 2, old_sat_image=image_2019, new_sat_image=image_2023)
            # class_wise_changes, specific_class_changes, change_title = find_changes(output_image_2019_raw,
            #                                 output_image_2023_raw, 2, old_sat_image=image_2019)
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
        
        # print(f'Output image 2023 raw shape: {output_image_2023_raw.shape}')
        # print(f'Output image 2019 raw shape: {output_image_2019_raw.shape}')
        # print('Image prediction completed')
        # if output_image_2019_raw.any() and output_image_2023_raw.any():
        #     print('Finding changes')

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
                                                                 'class_wise_changes': class_wise_changes,
                                                                 'specific_class_changes': specific_class_changes,
                                                                 'change_title': change_title,
                                                                 })

    else:
        print('Error')


@app.post('/')
def submit(request: Request,
           service: str = Form(...),
           year: str = Form(...),
           selection: str = Form(...),
           min_lat: str = Form(...),
           min_lon: str = Form(...),
           max_lat: str = Form(...),
           max_lon: str = Form(...),
           district: str = Form(...),
           upazila: str = Form(...)):

    print(selection)

    if selection == 'admin':
        if district and upazila:
            key = (district, upazila)

            if key in upazila_data.index:
                min_lon, min_lat, max_lon, max_lat = tuple(
                    upazila_data.loc[key])
                print(district, upazila, min_lon, min_lat, max_lon, max_lat)
                # return template.TemplateResponse('index.html', {'request': request})
            else:
                print("Upazila not found.")
                # return template.TemplateResponse('index.html', {'request': request})
                return HTMLResponse(content='<h1>Upazila not found.</h1>', status_code=200)
        elif district:
            # return HTMLResponse(content='<h1>Upazila not selected.</h1>', status_code=200)
            key = (district)
            # print(district_data)
            # print(key)

            if key in district_data.index:
                min_lon, min_lat, max_lon, max_lat = tuple(
                    district_data.loc[key])
                print(district, min_lon, min_lat, max_lon, max_lat)
                # return template.TemplateResponse('index.html', {'request': request})
            else:
                print("District not found.")
                # return template.TemplateResponse('index.html', {'request': request})
                return HTMLResponse(content='<h1>District not found.</h1>', status_code=200)
    elif min_lon and min_lat and max_lon and max_lat:
        min_lon, min_lat = float(min_lon), float(min_lat)
        max_lon, max_lat = float(max_lon), float(max_lat)
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
    else:
        print("Invalid coordinates provided.")
        # return template.TemplateResponse('index.html', {'request': request})
        return HTMLResponse(content='<h1>Invalid coordinates provided.</h1>', status_code=200)

    # print(service)

    # print(min_lat, min_lon, max_lat, max_lon)
    # min_lon, min_lat = 90.30638136, 23.78550914
    # max_lon, max_lat = 90.53348936, 23.88819931

    # image = get_google_map(min_lat, min_lon, max_lat, max_lon)

    if year == 'compare':
        return compare_year(request, min_lon, min_lat, max_lon, max_lat, service)

    print('Image fetching started')
    image = get_bing_map(min_lat,
                         min_lon, max_lat, max_lon, year)
    print('Image fetching completed')

    table = None

    if image is not None:
        time_start = perf_counter()
        print('Image prediction started')
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
        print('Image prediction completed')
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
