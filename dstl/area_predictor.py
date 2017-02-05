import sys

import numpy as np
import cv2
from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

class AreaPredictor:
  def __init__(self, train_image, train_area_classes):
    # TODO: Pass array of data (images)
    # vector of RGB pixels
    self.image_pixel_data = train_image.reshape(-1, 3).astype(np.float32)
    self.area_class_pixel_data = {}
    self.area_mask_shape = list(train_area_classes.classes.values())[0].area_mask.shape
    print('Training areas')
    for key, value in train_area_classes.classes.items():
      # NOTE: it's possible that an image doesn't have one of the areas, so all
      # values would be zero. in this case we don't train the classifier and will
      # predict all zeros
      pipeline = None

      if value.area_mask.max() > 0:
        print('%s...' % key, end=' ')
        sys.stdout.flush()
        pixel_vector = value.area_mask.reshape(-1)
        pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
        pipeline.fit(self.image_pixel_data, pixel_vector)
      else:
        print('(%s)...' % key, end=' ')
      self.area_class_pixel_data[key] = {
          'pixel_vector': pixel_vector,
          'pipeline': pipeline
          }
    print('done!')

  def predict(self, test_image):
    test_data = test_image.reshape(-1, 3).astype(np.float32)
    results = {}
    for key, value in self.area_class_pixel_data.items():
      print('Predicting class', key)
      if value['pipeline']:
        results[key] = value['pipeline'].predict_proba(test_data)[:, 1]
      else:
        results[key] = np.zeros(test_data.shape[0])
    return results

  def evaluate_prediction(self, prediction, truth):
    return average_precision_score(prediction, truth)

  def prediction_to_binary_prediction(self, prediction, threshold=0.3):
    return (prediction.reshape(self.area_mask_shape) >= threshold)#.astype(np.uint8)

  def prediction_mask_to_polygons(self, mask, epsilon=10., min_area=10.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
      if parent_idx != -1:
        child_contours.add(idx)
        cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
      if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
        assert cnt.shape[1] == 1
        poly = Polygon(
            shell=cnt[:, 0, :],
            holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
              if cv2.contourArea(c) >= min_area])
        all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
      all_polygons = all_polygons.buffer(0)
      # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
      # need to keep it a Multi throughout
      if all_polygons.type == 'Polygon':
        all_polygons = MultiPolygon([all_polygons])
    return all_polygons

