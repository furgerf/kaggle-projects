import csv as csv
import numpy as np
import pandas as pd
from sklearn import preprocessing

from cross_validation_result import CrossValidationResult

class Kaggle():
  """
  Base class of all Kaggle problems. TODO: Figure out how to describe properties in comments.
  """

  SEPARATOR = '-' * 80

  _TRAIN_COLUMN_NAME = 'Train'

  # NOTE: Derived classes must assign:
  PREDICTOR_COLUMN_NAME = None
  ID_COLUMN_NAME = None
  ALL_FEATURES = None

  def __init__(self, train_file, test_file, prediction_file, classifier_creator, silent=False):
    """
    Creates a new `Kaggle` instance.

    Args:
      train_file (str): Path to the file containing the training data.
      test_file (str):  Path to the file containing the test data.
      prediction_file (str): Path to the file where the predictions should be written to.
      classifier_creator (func): Function that creates a new instance of the desired classifier.
      silent (boolean): Suppresses all output if set to True. Defaults to False.
    """
    self.train_file = train_file
    self.test_file = test_file
    self.prediction_file = prediction_file
    self.classifier_creator = classifier_creator
    self.silent = silent


  def _load_data(self):
    """
    Loads the data from the instance's `train` and `test` files.

    Returns:
      tuple(DataFrame): Training and test data.
    """
    train_df = pd.read_csv(self.train_file, header=0)
    test_df = pd.read_csv(self.test_file, header=0)
    if not self.silent:
      print('Loaded training and test data')
    return train_df, test_df


  @staticmethod
  def _merge_data(data):
    """
    Merges the supplied training- and test-data into a single DataFrame with an additional column
    "Train".

    Args:
      data (tuple(DataFrame)): A tuple containing the training and test DataFrames.

    Returns:
      DataFrame: The combined training- and test-data.
    """
    train_df, test_df = data
    train_df[Kaggle._TRAIN_COLUMN_NAME] = True
    test_df[Kaggle._TRAIN_COLUMN_NAME] = False
    return pd.concat([train_df, test_df])


  @staticmethod
  def split_data(data):
    """
    Splits the provided data into training and test DataFrames according to the _TRAIN_COLUMN_NAME
    column and removes said column.

    Args:
      data (DataFrame): The data that should be split

    Returns:
      tuple(DataFrame): A tuple containing the training and test DataFrames.
    """
    train = data.loc[(data[Kaggle._TRAIN_COLUMN_NAME] == True)].drop(Kaggle._TRAIN_COLUMN_NAME, axis=1)
    test = data.loc[(data[Kaggle._TRAIN_COLUMN_NAME] == False)].drop(Kaggle._TRAIN_COLUMN_NAME, axis=1)
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


  '''
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
  '''


  def initialize(self):
    """
    Initializes the Kaggle by loading the data from the training and test CSV files and storing them
    in a combined DataFrame. Then, the feature engineering steps are carried out, which depend on
    the Kaggle implementation.
    """
    self.df = Kaggle._merge_data(self._load_data())
    self._engineer_features()
    if not self.silent:
      print('Initialized Kaggle instance')
      print(Kaggle.SEPARATOR)


  @staticmethod
  def _numericalize_data(data, silent=False):
    """
    Converts non-numeric columns to numeric columns by assigning each label an integer.

    Args:
      data (DataFrame): Data that should only contain numeric columns.
      silent (boolean): Suppresses all output if set to True. Defaults to False.

    Returns:
      tuple(DataFrame, list): Numericalized data; list of tuples with removed columns and added columns.
    """
    column_modifications = []
    for header in data:
      if header == Kaggle._TRAIN_COLUMN_NAME:
        if not silent:
          print('Skipping numericalization of column `%s`' % header)
        continue

      type = data[header].dtype
      if type != 'int64' and type != 'float64':
        if not silent:
          print('Numericalizing row %s (%s)' % (header, type))

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(data[header])
        if not silent:
          print('Classes: %s' % label_encoder.classes_)
        encoded_labels = label_encoder.transform(data[header]).reshape(-1, 1)

        one_hot_encoder = preprocessing.OneHotEncoder()
        one_hot_encoder.fit(encoded_labels)
        one_hot_labels = one_hot_encoder.transform(encoded_labels).toarray()
        data = data.drop(header, axis=1)
        new_columns = []
        for value in range(one_hot_encoder.n_values_[0]):
          new_header = '%s-%d' % (header, value)
          data[new_header] = one_hot_labels[:, value]
          new_columns.append(new_header)
        column_modifications.append((header, new_columns))

    return data, column_modifications


  def get_prepared_data(self, features):
    """
    Prepares the instance's data for prediction with the given features.

    Args:
      features (list(str)): List of the features that should be used for the model.

    Returns:
      tuple: training data, training data labels, test data, test data ids.
    """
    # don't modify input list
    features = features[:]
    if not self.silent:
      print('Preparing data for prediction with features: %s' % features)
    # initially, we also need the predictor, id, and train columns
    additional_features = [self.PREDICTOR_COLUMN_NAME, self.ID_COLUMN_NAME, Kaggle._TRAIN_COLUMN_NAME]
    all_relevant_features = features + additional_features

    # select all relevant rows
    data = self.df[all_relevant_features]

    # TODO check for null entries
    """
    if data.isnull():
      raise ValueError('Data must not contain `null` entries!')
    """

    # numericalize data
    data, column_modifications = Kaggle._numericalize_data(data, silent=self.silent)

    # update feature list
    for column in column_modifications:
      removed_column, added_columns = column
      if removed_column in features:
        if not self.silent:
          print('Replacing feature "%s" with features %s' % (removed_column, added_columns))

        features.remove(removed_column)
        features.extend(added_columns)

    # split into training and test data
    train_data, test_data = Kaggle.split_data(data)

    # prepare results - select relevant rows
    train = train_data[features]
    predictor = train_data[self.PREDICTOR_COLUMN_NAME]
    test = test_data[features]
    ids = test_data[self.ID_COLUMN_NAME]

    if not self.silent:
      print(Kaggle.SEPARATOR)

    return train, predictor, test, ids


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


  def predict_test_data(self, train, predictor, test, ids):
    """
    Runs a prediction on the supplied test data and stores the predicted labels together with the
    ids as CSV in the instance's `prediction_file`.

    Args:
      train (DataFrame): Features of the trainig set.
      predictor (DataFrame): Labels of the training set.
      test (DataFrame): Features of the test set.
      ids (Series): Ids of the test examples, to be mapped with the predictions.
    """
    header = [self.ID_COLUMN_NAME, self.PREDICTOR_COLUMN_NAME]
    predictions, classifier = self._predict(train, predictor, test)
    with open(self.prediction_file, 'wt') as file:
      open_file_object = csv.writer(file)
      open_file_object.writerow(header)
      open_file_object.writerows(zip(ids, predictions))
      file.close()

    if not self.silent:
      print('Predicted test data and written results to %s' % self.prediction_file)
      print(Kaggle.SEPARATOR)


  def cross_validate(self, data, predictor, folds=10):
    """
    Runs a cross-validation of the training data.

    Args:
      data (DataFrame): Features of the training set.
      predictor (Series): Matching labels of the training set.
      folds (int): Number of folds to use. Defaults to 10.

    Returns:
      CrossValidationResult: Results of the the cross validation.
    """
    fold_size = int(len(data)/folds)
    features = data.columns
    result = CrossValidationResult(folds, fold_size, features)

    if not self.silent:
      print('Starting %d-fold cross-validation with fold size %d based on features:\n%s' % (folds, fold_size, ', '.join(features)))

    if not self.silent:
      print('Running fold', end=' ', flush=True)
    for i in range(folds):
      if not self.silent:
        print(str(i), end='... ', flush=True)

      # prepare training set
      train = pd.concat([data[:i * fold_size:], data[(i+1) * fold_size:]])
      labels = pd.concat([predictor[:i * fold_size:], predictor[(i+1) * fold_size:]])

      # prepare test set
      test = data[i*fold_size:(i+1)*fold_size]
      answers = predictor[i*fold_size:(i+1)*fold_size]

      predictions, classifier = self._predict(train.values, labels.values, test.values)
      result.add_fold_predictions(predictions, answers, classifier.feature_importances_)

    if not self.silent:
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

  def get_deviation_per_feature(self):
    """
    Calculates the deviation of the output class of descrete features. It doesn't actually detect
    which features are discrete. Instead, the number of different values a feature can assume is
    compared to a maximum value.
    """
    MAX_CLASSES_PER_FEATURE = 25
    data, test = self.split_data(self.df)
    output_classes = data[self.PREDICTOR_COLUMN_NAME].unique()
    print('Analyzing prediction deviation per feature')

    for column in data:
      if column == self.PREDICTOR_COLUMN_NAME:
        continue
      classes = data[column].unique()
      class_count = len(classes)
      print(Kaggle.SEPARATOR)
      if class_count > MAX_CLASSES_PER_FEATURE:
        print('Ignoring feature %s because it has too many classes (%d > %d)' % (column, class_count, MAX_CLASSES_PER_FEATURE))
        continue

      print('Calculating prediction deviation of feature %s with %d classes' % (column, class_count))
      column_distribution = []
      for c in classes:
        output_distribution = []
        for o in output_classes:
          count = len(data.loc[(data[column] == c) & (data[self.PREDICTOR_COLUMN_NAME] == o)])
          output_distribution.append(count)

        total_count = sum(output_distribution)
        output_distribution = [od / total_count for od in output_distribution]
        # print('%s = %s divides like that: %s' % (column, c, output_distribution))
        column_distribution.append(output_distribution)

      for output in range(0, len(output_classes)):
        class_probabilities = [c[output] for c in column_distribution]
        print('* %s = %s by class with average\t%f\t(stdev %f)' % (column.ljust(15), output_classes[output], np.average(class_probabilities), np.std(class_probabilities)))

    print(Kaggle.SEPARATOR)

