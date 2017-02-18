import csv
import logging
import sys

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np

from area_predictor import AreaPredictor
from image import Image
from utils import Utils

plt.switch_backend('Qt5Agg')

csv.field_size_limit(sys.maxsize)

DATA_DIRECTORY = './data/'
DATA_GRIDS_FILE = DATA_DIRECTORY + 'grid_sizes.csv'

def get_logger():
  logger = logging.getLogger('dstl')
  # TODO: Change field colors
  coloredlogs.install(level='DEBUG')
  handler = logging.StreamHandler()
  # TODO: Add fixed width across module+funcname+lineno
  # log_format = '%(asctime)s %(relativeCreated)d %(module)s.%(funcName)s:%(lineno)d %(levelname)-8s %(message)s'
  log_format = '%(asctime)s %(module)s.%(funcName)s:%(lineno)d %(levelname)-8s %(message)s'
  formatter = coloredlogs.ColoredFormatter(log_format)
  handler.setFormatter(formatter)

  logger.propagate = False
  logger.handlers = []
  logger.addHandler(handler)
  logger.setLevel(logging.DEBUG)

  return logger

def load_grid_sizes():
  log.info('Retrieving grid sizes')

  with open(DATA_GRIDS_FILE) as grid_file:
    reader = csv.reader(grid_file)
    next(reader, None)  # skip header

    result = {}
    for i, x, y in reader:
      # store x_max/y_min for each image
      result[i] = (float(x), float(y))

  return result

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

log = get_logger()
grid_sizes = load_grid_sizes()

IMAGE_ID = '6120_2_2'
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
predictor = AreaPredictor(LABELS)

image = Image(IMAGE_ID, grid_sizes[IMAGE_ID], DATA_DIRECTORY)
image.load_image()
image.load_areas()

# plt.imshow(scale_percentile(image.raw_data))
# plt.imshow(image.area_classes.image_mask, alpha=0.5)
# plt.show()

predictor.train(image)

predictions = predictor.predict(image)
for area_class, prediction in predictions.items():
  binary_prediction = predictor.prediction_to_binary_prediction(prediction)
  prediction_polygons = predictor.prediction_mask_to_polygons(binary_prediction)
  predictions[area_class] = prediction_polygons
image.area_classes.add_predictions(predictions)

plt.figure()
plt.imshow(scale_percentile(image.raw_data))
plt.imshow(image.area_classes.prediction_image_mask, alpha=0.5)
plt.show()

# if __name__ == '__main__':
#   main()

logging.shutdown()
