import sys
import logging
from datetime import datetime

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
  def __init__(self, labels=None, predictors=None):
    if labels is not None and predictors is not None:
      raise ValueError('Pass either labels or predictors')

    self.log = logging.getLogger('dstl')

    if labels is not None:
      self.log.info('Creating new predictor for labels {}'.format(','.join(labels)))
      self.predictors = AreaPredictor.get_empty_predictors(labels)
    elif predictors is not None:
      self.log.info('Loading existing predictors with labels {}'.format(','.join(predictors.keys())))
      self.predictors = predictors
    else:
      raise ValueError('Pass either labels or predictors')

  def train(self, image):
    self.log.warning('Training image {}...'.format(image.image_id))
    start_time = datetime.utcnow()
    image_data = image.raw_data.reshape(-1, 3).astype(np.float)
    for area_id, predictor in self.predictors.items():
      area_data = image.area_classes.classes[area_id].area_mask.reshape(-1)
      if area_data.max() > 0:
        print('%s...' % area_id, end=' ')
        sys.stdout.flush()
        # NOTE: Maybe should use partial_fit
        predictor.fit(image_data, area_data)
      else:
        print('(%s)...' % area_id, end=' ')
        # NOTE: Skipping fitting of predictor because the area doesn't exist in the image
        # BUT, this should also be taken into account...
    print()
    self.log.info('... done! ({:.1f}s)'.format((datetime.utcnow() - start_time).total_seconds()))

  @staticmethod
  def get_empty_predictors(keys):
    predictors = {}
    for key in keys:
      predictors[key] = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
    return predictors

  def predict(self, test_image):
    image_shape = test_image.raw_data.shape[:2]
    test_data = test_image.raw_data.reshape(-1, 3).astype(np.float32)
    results = {}
    self.log.warning('Predicting image {}...'.format(test_image.image_id))
    start_time = datetime.utcnow()
    for area_id, predictor in self.predictors.items():
      print('%s...' % area_id, end=' ')
      sys.stdout.flush()
      try:
        results[area_id] = predictor.predict_proba(test_data)[:, 1].reshape(image_shape)
      except AttributeError:
        print('x', end=' ')
        results[area_id] = np.zeros(test_data.shape[0]).reshape(image_shape)
    self.log.info('... done! ({:.1f}s)'.format((datetime.utcnow() - start_time).total_seconds()))
    return results

  def evaluate_prediction(self, prediction, truth):
    return average_precision_score(prediction, truth)

  def prediction_to_binary_prediction(self, prediction, threshold=0.3):
    return prediction >= threshold

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

