import numpy as np

class CrossValidationResult():
  """
  Container for the results of a cross validation. Works only for binary classification.

  This could be generalized further if need be.
  """

  SEPARATOR = '-' * 80

  def __init__(self, folds, fold_size, feature_labels):
    """
    Creates a new CrossValidationResult instance.

    Args:
      folds (int): Number of folds to expect.
      fold_size (int): Number of examples in a fold.
      feature_labels (list(str)): The ordered list of feature descriptors. Has no significance for
        the calculations but is useful when describing the feature importance.
    """
    self.folds = folds
    self.fold_size = fold_size
    self.feature_labels = feature_labels

    self.true_positives = []
    self.false_positives = []
    self.false_negatives = []
    self.true_negatives = []
    self.accuracies = []
    self.precisions = []
    self.recalls = []
    self.f1_scores = []
    self.feature_importances = []

  @staticmethod
  def mean(data):
    """
    Calculates the mean value of the provided data.

    Args:
      data (list): Data over which to calculate the mean.

    Returns:
      float: Mean value of the data.
    """
    return sum(data) / len(data)

  def add_fold_predictions(self, predictions, answers, feature_importances):
    """
    Adds the results of a fold prediction to the results.

    Args:
      predictions (list): Predictions that were made in this fold.
      answers (list): Correct answers to the predictions.
      feature_importances (list): Importance of the different features in the model.
    """
    tp = np.sum([(x == y == 1) for x, y in zip(predictions, answers)])
    fp = np.sum([(x == 1 and y == 0) for x, y in zip(predictions, answers)])
    fn = np.sum([(x == 0 and y == 1) for x, y in zip(predictions, answers)])
    tn = np.sum([(x == y == 0) for x, y in zip(predictions, answers)])

    if len(predictions) != self.fold_size or tp + fp + fn + tn != self.fold_size:
      raise ValueError('Unexpected number of prediction results!!')

    accuracy = (tp + tn) / self.fold_size
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * tp / (2 * tp + fp + fn)

    self.true_positives.append(tp/self.fold_size)
    self.false_positives.append(fp/self.fold_size)
    self.false_negatives.append(fn/self.fold_size)
    self.true_negatives.append(tn/self.fold_size)
    self.accuracies.append(accuracy)
    self.precisions.append(precision)
    self.recalls.append(recall)
    self.f1_scores.append(f1_score)
    self.feature_importances.append(feature_importances)

  def print_results(self):
    """
    Prints statistics of the cross validation results.
    """
    print('Cross-validation results') # TODO: Add stdev
    print(CrossValidationResult.SEPARATOR)
    print('TP:\tmin=%f\tmean=%f\tmax=%f' % (min(self.true_positives),
      sum(self.true_positives)/len(self.true_positives), max(self.true_positives)))
    print('FP:\tmin=%f\tmean=%f\tmax=%f' % (min(self.false_positives),
      sum(self.false_positives)/len(self.false_positives), max(self.false_positives)))
    print('FN:\tmin=%f\tmean=%f\tmax=%f' % (min(self.false_negatives),
      sum(self.false_negatives)/len(self.false_negatives), max(self.false_negatives)))
    print('TN:\tmin=%f\tmean=%f\tmax=%f' % (min(self.true_negatives),
      sum(self.true_negatives)/len(self.true_negatives), max(self.true_negatives)))

    print('ACC:\tmin=%f\tmean=%f\tmax=%f' % (min(self.accuracies),
      sum(self.accuracies)/len(self.accuracies), max(self.accuracies)))
    print('PREC:\tmin=%f\tmean=%f\tmax=%f' % (min(self.precisions),
      sum(self.precisions)/len(self.precisions), max(self.precisions)))
    print('REC:\tmin=%f\tmean=%f\tmax=%f' % (min(self.recalls),
      sum(self.recalls)/len(self.recalls), max(self.recalls)))
    print('F1:\tmin=%f\tmean=%f\tmax=%f' % (min(self.f1_scores),
      sum(self.f1_scores)/len(self.f1_scores), max(self.f1_scores)))
    print(CrossValidationResult.SEPARATOR)

    mean_importance_ranking = []
    max_attribute_name_length = max(map(lambda s: len(s), self.feature_labels))
    for i in range(len(self.feature_importances[0])):
      imp = []
      for j in self.feature_importances:
        imp.append(j[i])
      mean = sum(imp)/len(imp)
      # print('IMP(%d):\tmin=%f\tmean=%f\tmax=%f' % (i, min(imp), mean, max(imp)))
      mean_importance_ranking.append((mean, i))
    mean_importance_ranking.sort()
    print('Mean importance ranking: \n%s' % '\n'.join(list(map(lambda x: '%s: %f' % (self.feature_labels[x[1]].ljust(max_attribute_name_length + 1), x[0]), mean_importance_ranking))))
    print(CrossValidationResult.SEPARATOR)

