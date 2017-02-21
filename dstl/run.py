import os
import pickle
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
DATA_THREEBAND = DATA_DIRECTORY + 'three_band/'
DATA_GRIDS_FILE = DATA_DIRECTORY + 'grid_sizes.csv'
DATA_AREAS_WKT = DATA_DIRECTORY + 'train_wkt_v4.csv'

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

def load_area_data():
  with open(DATA_AREAS_WKT) as csv_file:
    data = {}
    reader = csv.reader(csv_file)

    # skip header
    next(reader, None)
    for image_id, area_class, areas in reader:
      if not image_id in data:
        data[image_id] = {}
      data[image_id][area_class] = areas
    return data

def visualize_prediction(predictions, predictor, image):
  # visualize prediction
  for area_class, prediction in predictions.items():
    binary_prediction = predictor.prediction_to_binary_prediction(prediction)
    prediction_polygons = predictor.prediction_mask_to_polygons(binary_prediction)
    predictions[area_class] = prediction_polygons
  image.area_classes.add_predictions(predictions)
  # plot training mask
  plt.imshow(scale_percentile(image.raw_data))
  plt.imshow(image.area_classes.image_mask, alpha=0.5)

  # plot predicted mask
  plt.figure()
  plt.imshow(scale_percentile(image.raw_data))
  plt.imshow(image.area_classes.prediction_image_mask, alpha=0.5)
  plt.show()

def create_and_train_predictor():
  LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
  predictor = AreaPredictor(LABELS)

  for image_id in area_data.keys():
    image = Image(image_id, grid_sizes[image_id], DATA_DIRECTORY)

    image.load_image()
    image.load_areas(area_data[image_id])

    # train and predict same image
    predictor.train(image)

  return predictor

def load_trained_predictor(file_name='predictors.p'):
  with open(file_name, 'rb') as predictors_file:
    predictor_data = pickle.load(predictors_file)
    predictor = AreaPredictor(predictors=predictor_data)
    return predictor

def create_submission(entries, file_name='submission.csv'):
  with open(file_name, 'w') as submission_file:
    writer = csv.writer(submission_file)

    writer.writerow(['ImageId', 'ClassType', 'MultipolygonWKT'])

    log.warning('Creating submission in {} with {} entries...'.format(file_name, len(entries)))
    for entry in entries:
      writer.writerow(entry)
    log.info('... done!')


# def main():

# setup
log = get_logger()

grid_sizes = load_grid_sizes()
area_data = load_area_data()

predictor = load_trained_predictor()

# process all images
entries = []
image_ids = []
training_images_to_predict = []
with open('./correct-order', 'r') as fh:
  reader = csv.reader(fh)
  for row in reader:
    image_ids.append(row[0])

for i, image_id in enumerate(image_ids):
  log.debug('Processing image {}/{} ({})'.format(i + 1, len(image_ids), image_id))

  image = Image(image_id, grid_sizes[image_id], DATA_DIRECTORY)

  image.load_image()
  areas = area_data[image_id] if image_id in area_data else None
  image.load_areas(areas)

  predictions = predictor.predict(image)
  image.area_classes.add_predictions(predictions)
  for area_id, area_class in image.area_classes.classes.items():
    entries.append((image_id, area_id, area_class.predicted_submission_polygons))

create_submission(entries)

"""
# process single image
IMAGE_ID = '6120_2_2'
image = Image(IMAGE_ID, grid_sizes[IMAGE_ID], DATA_DIRECTORY)

image.load_image()
image.load_areas(area_data[IMAGE_ID])

predictions = predictor.predict(image)
image.area_classes.add_predictions(predictions)
create_submission([image])
"""

"""
# train and predict same image
predictor.train(image)
predictions = predictor.predict(image)

# visualize prediction
for area_class, prediction in predictions.items():
  binary_prediction = predictor.prediction_to_binary_prediction(prediction)
  prediction_polygons = predictor.prediction_mask_to_polygons(binary_prediction)
  predictions[area_class] = prediction_polygons
image.area_classes.add_predictions(predictions)

# plot training mask
plt.imshow(scale_percentile(image.raw_data))
plt.imshow(image.area_classes.image_mask, alpha=0.5)

# plot predicted mask
plt.figure()
plt.imshow(scale_percentile(image.raw_data))
plt.imshow(image.area_classes.prediction_image_mask, alpha=0.5)
plt.show()
"""

# if __name__ == '__main__':
#   main()

logging.shutdown()

