import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

from functools import reduce

from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

csv.field_size_limit(sys.maxsize)

CLASS_BUILDING = '1'
CLASS_MANMADE_STRUCTURE = '2'
CLASS_ROAD = '3'
CLASS_TRACK = '4'
CLASS_TREE = '5'
CLASS_CROP = '6'
CLASS_WATER_MOVING = '7'
CLASS_WATER_STANDING = '8'
CLASS_VEHICLE_LARGE = '9'
CLASS_VEHICLE_SMALL = '10'
POLYGON_CLASSES = [
    CLASS_BUILDING,
    CLASS_BUILDING,
    CLASS_MANMADE_STRUCTURE,
    CLASS_ROAD,
    CLASS_TRACK,
    CLASS_TREE,
    CLASS_CROP,
    CLASS_WATER_MOVING,
    CLASS_WATER_STANDING,
    CLASS_VEHICLE_LARGE,
    CLASS_VEHICLE_SMALL
    ]
POLYGON_CLASS_COLORS = [
    ]
NUMBER_OF_POLYGON_CLASSES = len(POLYGON_CLASSES)

DATA_DIRECTORY = './data/'
DATA_GRIDS_FILE = DATA_DIRECTORY + 'grid_sizes.csv'
DATA_POLYGONS_WKT = DATA_DIRECTORY + 'train_wkt_v4.csv'
DATA_IMAGES_THREEBAND = DATA_DIRECTORY + 'three_band/'
DATA_IMAGE_THREEBAND_FILE = DATA_IMAGES_THREEBAND + '%s.tif'

def get_grid_size_for_image(image_id):
  print('Retrieving grid size for image')
  for i, x, y in csv.reader(open(DATA_GRIDS_FILE)):
    if i != image_id:
      continue
    return (float(x), float(y))

def get_polygons_for_image(image_id):
  print('Loading polygons for image')
  polygons = [[]] * NUMBER_OF_POLYGON_CLASSES
  for i, polygon_type, polygon in csv.reader(open(DATA_POLYGONS_WKT)):
    if i != image_id:
      continue
    polygons[int(polygon_type) - 1] = shapely.wkt.loads(polygon)
  return polygons

def get_image(image_id):
  print('Loading image')
  return tiff.imread(DATA_IMAGE_THREEBAND_FILE % image_id).transpose([1, 2, 0])

def get_scale_factors_for_image(x_max, y_min, width, height):
  print('Calculating polygon scale factors')
  width = width * (width / (width + 1))
  height = height * (height / (height + 1))
  return width / x_max, height / y_min

def scale_polygons(polygons, x_max, y_min, width, height):
  x_scale, y_scale = get_scale_factors_for_image(x_max, y_min, width, height)

  print('Scaling polygons')
  return list(map(lambda polys: \
      list(map(lambda p: \
      shapely.affinity.scale(p, xfact=x_scale, yfact=y_scale, origin=(0, 0, 0)), \
      polys)), \
      polygons))

def get_mask_for_polygons(polygons, image_size):
  mask_image = np.zeros(image_size, np.uint8)

  # if not polygons:
    # return mask_image

  int_coords = lambda x: np.array(x).round().astype(np.int32)
  exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
  interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
  cv2.fillPoly(mask_image, exteriors, 1)
  cv2.fillPoly(mask_image, interiors, 0)

  return mask_image

def scale_percentile(matrix):
  w, h, d = matrix.shape
  matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
  # Get 2nd and 98th percentile
  mins = np.percentile(matrix, 1, axis=0)
  maxs = np.percentile(matrix, 99, axis=0) - mins
  matrix = (matrix - mins[None, :]) / maxs[None, :]
  matrix = np.reshape(matrix, [w, h, d])
  matrix = matrix.clip(0, 1)
  return matrix

def get_mask_image(mask, r, g, b):
  return np.stack([np.multiply(mask, r), np.multiply(mask, g), np.multiply(mask, b)]) \
      .transpose([1, 2, 0])


# def main():
IMAGE_ID = '6120_2_2'
print('Processing image', IMAGE_ID)
x_max, y_min = get_grid_size_for_image(IMAGE_ID)
image = get_image(IMAGE_ID)
image_size = image.shape[:2]
height, width = image_size
polygons = get_polygons_for_image(IMAGE_ID)
polygons_scaled = scale_polygons(polygons, x_max, y_min, width, height)
# polygons = np.array(list(map(lambda polys: np.array(polys), polygons)))
polygon_masks = list(map(lambda p: get_mask_for_polygons(p, image_size), polygons_scaled))
visual_masks = [
    get_mask_image(polygon_masks[int(CLASS_BUILDING) - 1], 178, 178, 178),
    get_mask_image(polygon_masks[int(CLASS_MANMADE_STRUCTURE) - 1], 102, 102, 102),
    get_mask_image(polygon_masks[int(CLASS_ROAD) - 1], 179, 88, 6),
    get_mask_image(polygon_masks[int(CLASS_TRACK) - 1], 223, 194, 125),
    get_mask_image(polygon_masks[int(CLASS_TREE) - 1], 27, 120, 55),
    get_mask_image(polygon_masks[int(CLASS_CROP) - 1], 166, 219, 160),
    get_mask_image(polygon_masks[int(CLASS_WATER_MOVING) - 1], 116, 173, 209),
    get_mask_image(polygon_masks[int(CLASS_WATER_STANDING) - 1], 69, 117, 180),
    get_mask_image(polygon_masks[int(CLASS_VEHICLE_LARGE) - 1], 244, 109, 67),
    get_mask_image(polygon_masks[int(CLASS_VEHICLE_SMALL) - 1], 215, 48, 39)
    ]
mask = reduce(np.add, visual_masks)
plt.imshow(scale_percentile(image))
plt.imshow(mask, alpha=0.6)
plt.show()

# if __name__ == '__main__':
#   main()

