from titanic_kaggle import TitanicKaggle
from cross_validation_result import CrossValidationResult

class FeatureFinder():
  def __init__(self, titanic, all_features, train, predictor, test, random_seeds=[7, 42, 123, 6340, 43627]):
    self.titanic = titanic
    self.all_features = all_features
    self.train = train
    self.predictor = predictor
    self.test = test
    self.random_seeds = random_seeds

    self._generate_feature_sets()

  def _generate_feature_sets(self):
    # TODO: Improve...
    self.features_sets = []
    self.self.features_sets.append(all_features)
    for i in range(len(self.all_features)):
      new_feature_set = self.all_features.copy()
      new_feature_set.remove(self.all_features[i])
      self.features_sets.append(new_feature_set)

      for j in range(len(self.all_features) - 1):
        very_new_feature_set = new_feature_set.copy()
        very_new_feature_set.remove(new_feature_set[j])
        self.features_sets.append(very_new_feature_set)

    print('Generated %d different feature sets' % len(self.features_sets)))

  def run_predictions(self):
    results = {}
    for i, features in enumerate(self.features_sets):
      print('Processing feature set %d/%d' % (i, len(self.features_sets)))

      current_train = TitanicKaggle.numericalize_data(self.train[features])
      current_test = TitanicKaggle.numericalize_data(self.test[features])

      accuracies = []
      f1_scores = []

      for seed in random_seeds:
        self.titanic.classifier_creator = lambda: RandomForestClassifier(n_estimators=100, random_state=seed)
        cv_result = self.titanic.cross_validate(current_train, self.predictor, silent=True, folds=10)

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
    print(TitanicKaggle.SEPARATOR)

