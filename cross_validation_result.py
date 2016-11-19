import numpy as np

from kaggle import Kaggle

class CrossValidationResult():
  def __init__(self, folds, fold_size, feature_labels):
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

  def add_fold_predictions(self, predictions, answers, feature_importances):
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
    print(Kaggle.SEPARATOR)
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
    print(Kaggle.SEPARATOR)

    mean_importance_ranking = []
    for i in range(len(self.feature_importances[0])):
      imp = []
      for j in self.feature_importances:
        imp.append(j[i])
      mean = sum(imp)/len(imp)
      print('IMP(%d):\tmin=%f\tmean=%f\tmax=%f' % (i, min(imp), mean, max(imp)))
      mean_importance_ranking.append((mean, i))
    mean_importance_ranking.sort()
    print('Mean importance ranking: \n%s' % '\n'.join(list(map(lambda x: '%s: %f' % (self.feature_labels[x[1]].ljust(16), x[0]), mean_importance_ranking))))
    print(Kaggle.SEPARATOR)

