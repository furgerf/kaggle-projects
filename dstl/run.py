import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

import csv
import sys

import numpy as np
import tifffile as tiff

from utils import Utils
from area_classes import AreaClasses
from area_predictor import AreaPredictor

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

def compare_area_class(area_class, alpha=0.5):
  plt.imshow(area_classes.classes[area_class].mask_image, alpha=alpha)
  plt.imshow(area_classes.classes[area_class].predicted_mask_image, alpha=alpha)
  plt.show()

def evaluate_jaccard(area_class_index):
  area_class = area_classes.classes[area_class_index]
  intersection = area_class.predicted_submission_polygons.intersection(area_class.areas).area
  union = area_class.predicted_submission_polygons.union(area_class.areas).area

  print('Jaccard for area class {}: {:.3g} ({:.3g}/{:.3g})'.format( \
      area_class_index, 0 if union == 0 else intersection / union, intersection, union))

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

# plt.imshow(scale_percentile(image))
# plt.imshow(area_classes.image_mask, alpha=0.5)
# plt.show()

predictor = AreaPredictor(image, area_classes)
predictions = predictor.predict(image)

for area_class, prediction in predictions.items():
  binary_prediction = predictor.prediction_to_binary_prediction(prediction)
  prediction_polygons = predictor.prediction_mask_to_polygons(binary_prediction)
  predictions[area_class] = prediction_polygons
area_classes.add_predictions(predictions)

# plt.figure()
# plt.imshow(scale_percentile(image))
# plt.imshow(area_classes.prediction_image_mask, alpha=0.5)
# plt.show()

# if __name__ == '__main__':
#   main()
