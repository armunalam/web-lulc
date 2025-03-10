import numpy as np
import ee
import requests
from io import BytesIO
from PIL import Image
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Initialize the Earth Engine API
private_key = "ee-satelliteserviceiub1-b8833d20142e.json"
service_account = "service-account-196@ee-satelliteserviceiub1.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(service_account, private_key)
ee.Initialize(credentials)

# Define the bounding box using latitude and longitude (minLon, minLat, maxLon, maxLat)
min_lon, min_lat = 90.30638136, 23.78550914  # Lower-left corner (longitude, latitude)
max_lon, max_lat = 90.53348936, 23.88819931  # Upper-right corner (longitude, latitude)

roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

# Get Sentinel-2 Image Collection (Least Cloudy Image)
image_collection = ee.ImageCollection("COPERNICUS/S2_SR") \
    .filterBounds(roi) \
    .filterDate('2024-01-01', '2024-02-01') \
    .sort('CLOUD_COVER')

# Get the least cloudy image
image = image_collection.first()

# Select RGB bands (Sentinel-2: B4 = Red, B3 = Green, B2 = Blue)
rgb_bands = ['B4', 'B3', 'B2']
image = image.select(rgb_bands)

# Normalize and scale image to 8-bit (0-255) to avoid white images
def scale_image(image):
    return image.divide(3000).multiply(255).clip(roi).byte()

image = scale_image(image)

# Visualization parameters (needed for PNG)
vis_params = {
    'min': 0,
    'max': 255,  # 8-bit range
    'bands': rgb_bands,
    'region': roi.getInfo(),
    'format': 'PNG',
    'scale': 10  # Max resolution for Sentinel-2
}

# Get High-Resolution Image Download URL
url = image.getDownloadURL(vis_params)
print(f"Download URL: {url}")

# Download and open image
response = requests.get(url)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    img.show()  # Show image
    img.save("gee_high_res_3.png")  # Save as PNG
    print("High-resolution PNG image saved successfully.")
else:
    print("Failed to download image.")