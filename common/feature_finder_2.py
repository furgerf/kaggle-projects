import itertools
import re

import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from kaggle import Kaggle

class FeatureFinder():
  def __init__(self, classifiers, train, predictors, metric, min_features=1, max_features=None):
    self.classifiers = classifiers
    self.train = train
    self.predictors = predictors
    self.metric = metric

    self._generate_feature_sets(list(self.train.columns.values), min_features, max_features)


  def _generate_feature_sets(self, all_features, min_features, max_features):
    feature_groups = {}
    regex = re.compile('(\w+)-\d+')
    for feature in all_features:
      match = re.search(regex, feature)
      if match:
        key = match.group(1)
        if key in feature_groups:
          feature_groups[key].append(feature)
        else:
          feature_groups[key] = [feature]
      else:
        feature_groups[feature] = feature

    print('Feature groups: %s' % feature_groups)
    max_features = max_features or len(feature_groups)

    print('Combinations of %d to %d features' % (min_features, max_features))
    self.feature_sets = []
    for list_length in range(min_features, max_features + 1):
      for subset in itertools.combinations(feature_groups.keys(), list_length):
        feature_set = []
        for feature in subset:
          if isinstance(feature_groups[feature], list):
            feature_set.extend(feature_groups[feature])
          else:
            feature_set.append(feature_groups[feature])
        feature_set.sort()
        self.feature_sets.append(feature_set)

    print('%d feature groups (%d features) => %d feature sets' % \
        (len(feature_groups), len(all_features), len(self.feature_sets)))
    print(Kaggle.SEPARATOR)


  def run_predictions(self):
    results = {}
    for i, features in enumerate(self.feature_sets):
      print('Processing feature set %d/%d: %s' % (i+1, len(self.feature_sets), features))

      current_train = self.train[features]

      scores = []
      for classifier in self.classifiers:
        score = cross_val_score(classifier, current_train, self.predictors, cv=10, scoring=self.metric, n_jobs=-1)
        scores.extend(score)

      mean_score = np.mean(scores)
      print('Mean score', mean_score)
      print(Kaggle.SEPARATOR)

      results[mean_score] = (features, np.mean(scores))

    return results


  def evaluate_feature_finder_results(self, results):
    keys = list(results.keys())
    keys.sort()
    for key in keys:
      features, mean_score = results[key]
      # print('Features: %s' % features)
      # print('AVG %s:\t%f' % (self.metric, mean_score))
      print('Avg %s:\t%f:\t%s' % (self.metric, mean_score, features))
    print(Kaggle.SEPARATOR)

