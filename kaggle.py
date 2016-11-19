import csv as csv
import numpy as np
import pandas as pd

class Kaggle():
  SEPARATOR = '-' * 80

  def __init__(self, train_file, test_file, prediction_file, classifier_creator):
    self.train_file = train_file
    self.test_file = test_file
    self.prediction_file = prediction_file
    self.classifier_creator = classifier_creator

    print('')
    print('')
    print('')
    print(Kaggle.SEPARATOR)
    print('Created new Kaggle instance')
    print(Kaggle.SEPARATOR)

  def _load_data(self):
    train_df = pd.read_csv(self.train_file, header=0)
    test_df = pd.read_csv(self.test_file, header=0)
    print('Loaded training and test data')
    return train_df, test_df

  @staticmethod
  def _merge_data(data):
    train_df, test_df = data
    train_df['Train'] = True
    test_df['Train'] = False
    return pd.concat([train_df, test_df])

  def split_data(self):
    train = self.df.loc[(self.df.Train == True)].drop('Train', axis=1)
    test = self.df.loc[(self.df.Train == False)].drop('Train', axis=1)
    return train, test


  def _engineer_features(self):
    raise NotImplementedError('Feature engineering must be implemented in the derived class')

  def remove_columns(self, columns):
    """
    Removes the specified additional columns from `self.df`.
    """
    print('Dropping columns %s' % columns)
    self.df = self.df.drop(columns, axis=1)


  def initialize(self):
    """
    Initializes the Kaggle by loading the data from the training and test CSV files and storing them
    in a combined DataFrame. Then, the feature engineering steps are carried out, which depend on
    the Kaggle implementation.
    """
    self.df = Kaggle._merge_data(self._load_data())
    self._engineer_features()
    print('Initialized Kaggle instance')
    print(Kaggle.SEPARATOR)

  @staticmethod
  def integerize_data(data):
    for header in data:
      type = data[header].dtype
      if type != 'int64' and type != 'float64':
        print('Integerizing row %s (%s)' % (header, type))
        raise NotImplementedError()
    return data


  def _predict(self, train, predictor, test):
    fitted_classifier = self.classifier_creator().fit(train, predictor)
    return fitted_classifier.predict(test).astype(int), fitted_classifier

  def predict_test_data(self, train, predictor, test, ids, header):
    predictions, fitted_classifier = self._predict(train, predictor, test)

    with open(self.prediction_file, 'wt') as predictions_file:
      open_file_object = csv.writer(predictions_file)
      open_file_object.writerow(header)
      open_file_object.writerows(zip(ids, predictions))
      predictions_file.close()

    print('Predicted test data and written results to %s' % self.prediction_file)

  def cross_validate(self, data, predictor, folds=10):
    fold_size = int(len(data)/folds)
    features = data.columns

    tps = []
    fps = []
    fns = []
    tns = []
    accs = []
    precs = []
    recs = []
    f1s = []
    importances = []

    print('Starting %d-fold cross-validation with fold size %d based on features:\n%s' % (folds, fold_size, ', '.join(features)))

    print('Fold', end=' ', flush=True)
    for i in range(folds):
      print(str(i), end='... ', flush=True)

      # prepare training set
      train = pd.concat([data[:i * fold_size:], data[(i+1) * fold_size:]])
      labels = pd.concat([predictor[:i * fold_size:], predictor[(i+1) * fold_size:]])

      # prepare test set
      test = data[i*fold_size:(i+1)*fold_size]
      answers = predictor[i*fold_size:(i+1)*fold_size]

      predictions, forest = self._predict(train.values, labels.values, test.values)
      tp = np.sum([(x == y == 1) for x, y in zip(predictions, answers)])
      fp = np.sum([(x == 1 and y == 0) for x, y in zip(predictions, answers)])
      fn = np.sum([(x == 0 and y == 1) for x, y in zip(predictions, answers)])
      tn = np.sum([(x == y == 0) for x, y in zip(predictions, answers)])
      if len(predictions) != fold_size or tp + fp + fn + tn != fold_size:
        raise Error('Unexpected number of prediction results!!')

      # print('    TP:  %f,\tFP:   %f,\tFN:  %f,\tTN: %f' % (tp/fold_size, fp/fold_size, fn/fold_size, tn/fold_size))
      acc = (tp + tn) / fold_size
      prec = tp / (tp + fp)
      rec = tp / (tp + fn)
      f1 = 2 * tp / (2 * tp + fp + fn)
      # print('    acc: %f,\tprec: %f,\trec: %f,\tf1: %f' % (acc, prec, rec, f1))

      tps.append(tp/fold_size)
      fps.append(fp/fold_size)
      fns.append(fn/fold_size)
      tns.append(tn/fold_size)
      accs.append(acc)
      precs.append(prec)
      recs.append(rec)
      f1s.append(f1)
      importances.append(forest.feature_importances_)

    print('\n... finished running cros-validation!')
    print(Kaggle.SEPARATOR)

    return (tps, fps, fns, tns, accs, precs, recs, f1s, importances, features)

  @staticmethod
  def evaluate_cross_validation_results(results):
    tps, fps, fns, tns, accs, precs, recs, f1s, importances, features = results
    print('Cross-validation results') # TODO: Add stdev
    print(Kaggle.SEPARATOR)
    print('TP:\tmin=%f\tmean=%f\tmax=%f' % (min(tps), sum(tps)/len(tps), max(tps)))
    print('FP:\tmin=%f\tmean=%f\tmax=%f' % (min(fps), sum(fps)/len(fps), max(fps)))
    print('FN:\tmin=%f\tmean=%f\tmax=%f' % (min(fns), sum(fns)/len(fns), max(fns)))
    print('TN:\tmin=%f\tmean=%f\tmax=%f' % (min(tns), sum(tns)/len(tns), max(tns)))
    print('ACC:\tmin=%f\tmean=%f\tmax=%f' % (min(accs), sum(accs)/len(accs), max(accs)))
    print('PREC:\tmin=%f\tmean=%f\tmax=%f' % (min(precs), sum(precs)/len(precs), max(precs)))
    print('REC:\tmin=%f\tmean=%f\tmax=%f' % (min(recs), sum(recs)/len(recs), max(recs)))
    print('F1:\tmin=%f\tmean=%f\tmax=%f' % (min(f1s), sum(f1s)/len(f1s), max(f1s)))
    print(Kaggle.SEPARATOR)
    mean_importance_ranking = []
    for i in range(len(importances[0])):
      imp = []
      for j in importances:
        imp.append(j[i])
      mean = sum(imp)/len(imp)
      print('IMP(%d):\tmin=%f\tmean=%f\tmax=%f' % (i, min(imp), mean, max(imp)))
      mean_importance_ranking.append((mean, i))
    mean_importance_ranking.sort()
    print('Mean importance ranking: \n%s' % '\n'.join(list(map(lambda x: '%s: %f' % (features[x[1]].ljust(16), x[0]), mean_importance_ranking))))
    print(Kaggle.SEPARATOR)

  def print_sample(self):
    print('Sample rows (head/tail):')
    print(Kaggle.SEPARATOR)
    print(pd.concat([self.df.head(3), self.df.tail(3)]))
    print(Kaggle.SEPARATOR)

