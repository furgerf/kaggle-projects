from functools import reduce
import csv

import numpy as np
import shapely.wkt

from area_class import AreaClass

class AreaClasses:
  DATA_AREAS_WKT = './data/train_wkt_v4.csv'

  def __init__(self, image_id, image_size, x_scale, y_scale):
    self._image_id = image_id
    self._image_size = image_size
    self._x_scale = x_scale
    self._y_scale = y_scale

    self.classes  = {
        '1': AreaClass('1', (178, 178, 178)),
        '2': AreaClass('2', (102, 102, 102)),
        '3': AreaClass('3', (179, 88, 6)),
        '4': AreaClass('4', (223, 194, 125)),
        '5': AreaClass('5', (27, 120, 55)),
        '6': AreaClass('6', (166, 219, 160)),
        '7': AreaClass('7', (116, 173, 209)),
        '8': AreaClass('8', (69, 117, 180)),
        '9': AreaClass('9', (244, 109, 67)),
        '10': AreaClass('10', (215, 48, 39))
        }

  def load(self):
    print('Loading areas for image', self._image_id)
    for i, area_class, areas in csv.reader(open(AreaClasses.DATA_AREAS_WKT)):
      if i != self._image_id:
        continue
      self.classes[area_class].set_areas(shapely.wkt.loads(areas), \
          self._image_size, self._x_scale, self._y_scale)
    print('done!')

    self.image_mask = reduce(np.add, list(map(lambda c: c.mask_image, self.classes.values())))

  def add_predictions(self, predictions):
    print('Adding prediction images')
    for key in self.classes.keys():
      self.classes[key].set_predicted_areas(predictions[key], self._image_size)
    self.prediction_image_mask = reduce(np.add, list(map(lambda c: c.predicted_mask_image, self.classes.values())))

