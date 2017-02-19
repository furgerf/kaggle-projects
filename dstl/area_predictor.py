import sys
import logging
from datetime import datetime

import numpy as np
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

