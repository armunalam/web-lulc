import ee
import requests
from io import BytesIO
from PIL import Image

# Initialize the Earth Engine API
private_key = "ee-satelliteserviceiub1-b8833d20142e.json"
service_account = "service-account-196@ee-satelliteserviceiub1.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(service_account, private_key)
ee.Initialize(credentials)

# Define the bounding box using latitude and longitude (minLon, minLat, maxLon, maxLat)
min_lon, min_lat = 90.30638136, 23.78550914  # Lower-left corner (longitude, latitude)
max_lon, max_lat = 90.83348936, 23.89819931  # Upper-right corner (longitude, latitude)
# min_lon, min_lat = 90.30638136, 23.78550914  # Lower-left corner (longitude, latitude)
# max_lon, max_lat = 90.53348936, 23.88819931  # Upper-right corner (longitude, latitude)

roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

# Select Sentinel-2 Image Collection (Surface Reflectance)
# image_collection = ee.ImageCollection("COPERNICUS/S2_SR") \
#     .filterBounds(roi) \
#     .filterDate('2024-01-01', '2024-02-01') \
#     .sort('CLOUD_COVER')

# Get Sentinel-2 Harmonized Image Collection (Least Cloudy Image)
# image_collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
image_collection = ee.ImageCollection("COPERNICUS/S2_SR") \
    .filterBounds(roi) \
    .filterDate('2025-02-01', '2025-02-19') \
    .sort('system:cloud_coverage')

# Get the least cloudy image
image = image_collection.first()

# Select RGB bands (Sentinel-2: B4, B3, B2)
rgb_bands = ['B4', 'B3', 'B2']
image = image.select(rgb_bands)

# Normalize pixel values to 8-bit (0-255 range)
def normalize(image):
    return image.divide(3000).multiply(255).byte()

image = normalize(image)

# Define visualization parameters
vis_params = {
    'min': 0,
    'max': 255,  # Now in 8-bit range
    'bands': rgb_bands
}

# Get high-resolution download URL
url = image.getDownloadURL({
    'scale': 10,  # Sentinel-2 has 10m resolution per pixel
    'region': roi.getInfo(),  # Defines the area
    'format': 'PNG'
})
print(f"Download URL: {url}")

# Download and open image
response = requests.get(url)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    img.show()  # Show image
    img.save("sat_2.png")  # Save to disk
    print("High-resolution image saved successfully.")
else:
    print("Failed to download image.")
