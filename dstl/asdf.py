import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

# from collections import defaultdict
import csv
import sys

# from shapely.geometry import MultiPolygon, Polygon
import numpy as np
import tifffile as tiff

from area_classes import AreaClasses

csv.field_size_limit(sys.maxsize)

DATA_DIRECTORY = './data/'
DATA_GRIDS_FILE = DATA_DIRECTORY + 'grid_sizes.csv'
DATA_IMAGES_THREEBAND = DATA_DIRECTORY + 'three_band/'
DATA_IMAGE_THREEBAND_FILE = DATA_IMAGES_THREEBAND + '%s.tif'

def load_grid_size_for_image(image_id):
  print('Retrieving grid size for image')
  for i, x, y in csv.reader(open(DATA_GRIDS_FILE)):
    if i != image_id:
      continue
    return (float(x), float(y))

def load_image(image_id):
  print('Loading image', image_id)
  return tiff.imread(DATA_IMAGE_THREEBAND_FILE % image_id).transpose([1, 2, 0])

def get_scale_factors_for_image(x_max, y_min, width, height):
  print('Calculating area scale factors')
  width = width * (width / (width + 1))
  height = height * (height / (height + 1))
  return width / x_max, height / y_min

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


# def main():
IMAGE_ID = '6120_2_2'
print('Processing image', IMAGE_ID)

image = load_image(IMAGE_ID)

x_max, y_min = load_grid_size_for_image(IMAGE_ID)
image_size = image.shape[:2]
height, width = image_size
x_scale, y_scale = get_scale_factors_for_image(x_max, y_min, width, height)

area_classes = AreaClasses(IMAGE_ID, image_size, x_scale, y_scale)
area_classes.load()

plt.imshow(scale_percentile(image))
plt.imshow(area_classes.mask, alpha=0.9)
plt.show()

# if __name__ == '__main__':
#   main()

