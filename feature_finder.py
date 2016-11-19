from kaggle import Kaggle
from cross_validation_result import CrossValidationResult

class FeatureFinder():
  def __init__(self, kaggles):
    self.kaggles = kaggles

    self._generate_feature_sets(self.kaggles[0].ALL_FEATURES)

  def _generate_feature_sets(self, all_features):
    # TODO: Improve...
    self.features_sets = []
    self.features_sets.append(all_features)
    for i in range(len(all_features)):
      new_feature_set = all_features.copy()
      new_feature_set.remove(all_features[i])
      self.features_sets.append(new_feature_set)

      """
      for j in range(len(all_features) - 1):
        very_new_feature_set = new_feature_set.copy()
        very_new_feature_set.remove(new_feature_set[j])
        self.features_sets.append(very_new_feature_set)
      """

    print('Generated %d different feature sets' % len(self.features_sets))

  def run_predictions(self):
    results = {}
    for i, features in enumerate(self.features_sets):
      print('Processing feature set %d/%d' % (i+1, len(self.features_sets)))

      accuracies = []
      f1_scores = []

      for kaggle in self.kaggles:
        train_data, test_data = kaggle.split_data()
        train_predictors = train_data[kaggle.PREDICTOR_COLUMN_NAME]
        train_data = Kaggle.numericalize_data(train_data[features])

        cv_result = kaggle.cross_validate(train_data, train_predictors, silent=True, folds=2)

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

