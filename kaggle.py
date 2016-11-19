import csv as csv
import numpy as np
import pandas as pd

from cross_validation_result import CrossValidationResult

class Kaggle():
  """
  Base class of all Kaggle problems. TODO: Figure out how to describe properties in comments.
  """

  SEPARATOR = '-' * 80

  # NOTE: Derived classes must assign:
  # PREDICTOR_COLUMN_NAME
  # ID_COLUMN_NAME
  # ALL_FEATURES

  def __init__(self, train_file, test_file, prediction_file, classifier_creator):
    """
    Creates a new `Kaggle` instance.

    Args:
      train_file (str): Path to the file containing the training data.
      test_file (str):  Path to the file containing the test data.
      prediction_file (str): Path to the file where the predictions should be written to.
      classifier_creator (func): Function that creates a new instance of the desired classifier.
    """
    self.train_file = train_file
    self.test_file = test_file
    self.prediction_file = prediction_file
    self.classifier_creator = classifier_creator

  def _load_data(self):
    """
    Loads the data from the instance's `train` and `test` files.

    Returns:
      tuple(DataFrame): Training and test data.
    """
    train_df = pd.read_csv(self.train_file, header=0)
    test_df = pd.read_csv(self.test_file, header=0)
    print('Loaded training and test data')
    return train_df, test_df

  @staticmethod
  def _merge_data(data):
    """
    Merges the supplied training- and test-data into a single DataFrame with an additional column
    "Train". TODO: Figure out how to specify the expanded tuple parameters directly.

    Args:
      data (tuple(DataFrame)): A tuple containing the training and test DataFrames.

    Returns:
      DataFrame: The combined training- and test-data.
    """
    train_df, test_df = data
    train_df['Train'] = True
    test_df['Train'] = False
    return pd.concat([train_df, test_df])

  def split_data(self):
    """
    Splits the instance's DataFrame into training and test DataFrames according to the "Train"
    column.

    Returns:
      tuple(DataFrame): A tuple containing the training and test DataFrames.
    """
    train = self.df.loc[(self.df.Train == True)].drop('Train', axis=1)
    test = self.df.loc[(self.df.Train == False)].drop('Train', axis=1)
    return train, test


  def analyze_data(self):
    """
    Derived classes must implement this method where the the instance's DataFrame is analyzed and
    statistics are printed.
    """
    raise NotImplementedError('Feature engineering must be implemented in the derived class')

  def _engineer_features(self):
    """
    Derived classes must implement this method where the additional features are engineered on the
    instance's DataFrame.
    """
    raise NotImplementedError('Feature engineering must be implemented in the derived class')

  def remove_columns(self, columns):
    """
    Removes the specified columns from the instance's DataFrame.

    Args:
      list(str): List of column indices that should be removed.

    Returns:
      None: Only the instance data is modified.
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
  def numericalize_data(data):
    """
    Converts non-numeric columns to numeric columns by assigning each label an integer.

    Args:
      data (DataFrame): Data that should only contain numeric columns.

    Returns:
      DataFrame: Numericalized data.
    """
    for header in data:
      type = data[header].dtype
      if type != 'int64' and type != 'float64':
        print('Integerizing row %s (%s)' % (header, type))
        raise NotImplementedError()
    return data


  def _predict(self, train, predictor, test):
    """
    Runs a prediction.

    Args:
      train (DataFrame): Features of the trainig set.
      predictor (Series): Labels of the training set.
      test (DataFrame): Features of the test set.

    Returns:
      tuple(DataFrame, Classifier): The predicted labels and the fitted classifier.
    """
    fitted_classifier = self.classifier_creator().fit(train, predictor)
    return fitted_classifier.predict(test).astype(int), fitted_classifier

  def predict_test_data(self, train, predictor, test, ids, header):
    """
    Runs a prediction on the supplied test data and stores the predicted labels together with the
    ids as CSV in the instance's `prediction_file`.

    Args:
      train (DataFrame): Features of the trainig set.
      predictor (DataFrame): Labels of the training set.
      test (DataFrame): Features of the test set.
      ids (Series): Ids of the test examples, to be mapped with the predictions.
      header (str): Name of the two columns, for the header row of the CSV file.
    """
    predictions, fitted_classifier = self._predict(train, predictor, test)
    with open(self.prediction_file, 'wt') as file:
      open_file_object = csv.writer(file)
      open_file_object.writerow(header)
      open_file_object.writerows(zip(ids, predictions))
      file.close()

    print('Predicted test data and written results to %s' % self.prediction_file)
    print(Kaggle.SEPARATOR)

  def cross_validate(self, data, predictor, silent=False, folds=10):
    """
    Runs a cross-validation of the training data.

    Args:
      data (DataFrame): Features of the training set.
      predictor (Series): Matching labels of the training set.
      folds (int): Number of folds to use. Defaults to 10.
      silent (boolean): Suppresses all output if set to True. Defaults to False.

    Returns:
      CrossValidationResult: Results of the the cross validation.
    """
    fold_size = int(len(data)/folds)
    features = data.columns
    result = CrossValidationResult(folds, fold_size, features)

    if not silent:
      print('Starting %d-fold cross-validation with fold size %d based on features:\n%s' % (folds, fold_size, ', '.join(features)))

    if not silent:
      print('Running fold', end=' ', flush=True)
    for i in range(folds):
      if not silent:
        print(str(i), end='... ', flush=True)

      # prepare training set
      train = pd.concat([data[:i * fold_size:], data[(i+1) * fold_size:]])
      labels = pd.concat([predictor[:i * fold_size:], predictor[(i+1) * fold_size:]])

      # prepare test set
      test = data[i*fold_size:(i+1)*fold_size]
      answers = predictor[i*fold_size:(i+1)*fold_size]

      predictions, classifier = self._predict(train.values, labels.values, test.values)
      result.add_fold_predictions(predictions, answers, classifier.feature_importances_)

    if not silent:
      print('\n... finished running cross-validation!')
      print(Kaggle.SEPARATOR)

    return result

  def print_sample_data(self):
    """
    Prints a few sample rows from the head and tail of the current instance DataFrame.
    """
    print('Sample rows (head/tail):')
    print(Kaggle.SEPARATOR)
    print(pd.concat([self.df.head(3), self.df.tail(3)]))
    print(Kaggle.SEPARATOR)

