import numpy as np

from kaggle import Kaggle
from cross_validation_result import CrossValidationResult

class FeatureFinder():
  def __init__(self, kaggles):
    self.kaggles = kaggles

    self._generate_feature_sets(self.kaggles[0].ALL_FEATURES)

  def _generate_feature_sets(self, all_features):
    # TODO: Improve...
    feature_sets = []
    feature_sets.append(all_features)
    for i in range(len(all_features)):
      new_feature_set = all_features.copy()
      new_feature_set.remove(all_features[i])
      feature_sets.append(new_feature_set)

      for j in range(len(all_features) - 1):
        very_new_feature_set = new_feature_set.copy()
        very_new_feature_set.remove(new_feature_set[j])
        feature_sets.append(very_new_feature_set)

        for k in range(len(all_features) - 2):
          very_very_new_feature_set = very_new_feature_set.copy()
          very_very_new_feature_set.remove(very_new_feature_set[k])
          feature_sets.append(very_very_new_feature_set)

          for l in range(len(all_features) - 3):
            very_very_very_new_feature_set = very_very_new_feature_set.copy()
            very_very_very_new_feature_set.remove(very_very_new_feature_set[l])
            feature_sets.append(very_very_very_new_feature_set)

    self.feature_sets = np.unique(feature_sets)
    print('Generated %d different feature sets' % len(self.feature_sets))

  def run_predictions(self):
    results = {}
    for i, features in enumerate(self.feature_sets):
      print('Processing feature set %d/%d' % (i+1, len(self.feature_sets)))

      accuracies = []
      f1_scores = []

      for kaggle in self.kaggles:
        train_data, test_data = kaggle.split_data()
        train_predictors = train_data[kaggle.PREDICTOR_COLUMN_NAME]
        train_data = Kaggle.numericalize_data(train_data[features])

        cv_result = kaggle.cross_validate(train_data, train_predictors, silent=True, folds=10)

        accuracies.extend(cv_result.accuracies)
        f1_scores.extend(cv_result.f1_scores)

      mean_accuracy = CrossValidationResult.mean(accuracies)
      mean_f1_score = CrossValidationResult.mean(f1_scores)

      overall_score = CrossValidationResult.mean([mean_accuracy, mean_f1_score])

      results[overall_score] = (features, mean_accuracy, mean_f1_score)

    return results

  @staticmethod
  def evaluate_feature_finder_results(results):
    keys = list(results.keys())
    keys.sort()
    for key in keys:
      features, mean_accuracy, mean_f1_score = results[key]
      print('Features: %s' % features)
      print('AVG:\tACC=%f\tF1=%f' % (mean_accuracy, mean_f1_score))
    print(Kaggle.SEPARATOR)

