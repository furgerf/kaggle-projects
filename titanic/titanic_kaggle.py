import os
import numpy as np
import matplotlib.pyplot as plot
import re

from kaggle import Kaggle

class TitanicKaggle(Kaggle):
  """
  Solver for the Titanic kaggle problem.
  """

  PREDICTOR_COLUMN_NAME = 'Survived'
  ID_COLUMN_NAME = 'PassengerId'
  ALL_FEATURES = ['Embarked', 'Parch', 'SibSp', 'Pclass', 'Sex', 'AgeGroup', 'FamilySize', 'FamilyGroup', 'Title']

  def __init__(self, classifier_creator, silent=False):
    """
    Creates a new `TitanicKaggle` instance.

    Args:
      classifier_creator (func): Function that creates a new instance of the desired classifier.
      silent (boolean): Suppresses all output if set to True. Defaults to False.
    """
    data_location = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    super(TitanicKaggle, self).__init__(os.path.join(data_location, 'train.csv'), os.path.join(data_location, 'test.csv'), 'prediction.csv', classifier_creator, silent=silent)

    if not self.silent:
      print(Kaggle.SEPARATOR)
      print('Created new TitanicKaggle instance')
      print(Kaggle.SEPARATOR)


  def _engineer_features(self):
    if not self.silent:
      print('Preparing data...')

    if not self.silent:
      print('- Filling missing ports...')
    most_common_port = self.df.Embarked.dropna().mode().values[0]
    if not self.silent:
      print('  using most common port "%s" for null-entries' % most_common_port)
    self.df.loc[self.df.Embarked.isnull(), 'Embarked'] = most_common_port
    # TODO: use common port based on some criteria

    if not self.silent:
      print('- Filling missing fares...')
    mean_fares = np.zeros((3))
    classes = np.unique(self.df.Pclass)
    if not self.silent:
      print('  calculating class-based mean fares')
    for index, pclass in enumerate(classes):
      mean_fares[index] = self.df[self.df.Pclass == pclass].Fare.mean()
    if not self.silent:
      print('  filling fares with means')
    for index, pclass in enumerate(classes):
      self.df.loc[(self.df.Fare.isnull()) & (self.df.Pclass == pclass), 'Fare'] = mean_fares[index]
    # TODO: explore other criteria for missing fares

    if not self.silent:
      print('- Preparing ages...')
    median_ages = np.zeros((2,3))
    sexes = np.unique(self.df.Sex)
    if not self.silent:
      print('  calculating sex- and class-based median ages')
    for sex_index, sex in enumerate(sexes):
      for class_index, pclass in enumerate(classes):
        median_ages[sex_index,class_index] = self.df[(self.df.Sex == sex) & (self.df.Pclass == pclass)].Age.dropna().median()
    if not self.silent:
      print('  filling missing ages with medians')
    for sex_index, sex in enumerate(sexes):
      for class_index, pclass in enumerate(classes):
        self.df.loc[(self.df.Age.isnull()) & (self.df.Sex == sex) & (self.df.Pclass == pclass), 'Age'] = median_ages[sex_index, class_index]
    age_group_count = 20
    if not self.silent:
      print('  distributing ages across %d groups' % age_group_count)
    age_bins = np.linspace(self.df.Age.min(),self.df.Age.max(), age_group_count)
    self.df['AgeGroup'] = np.digitize(self.df.Age, age_bins)

    if not self.silent:
      print('- Preparing families...')
    self.df['FamilySize'] = self.df.SibSp + self.df.Parch

    if not self.silent:
      print('- Preparing family groups...')
    self.df['FamilyGroup'] = self.df.FamilySize.map(lambda s: 'singleton' if s == 0 else 'large family' if s > 3 else 'small family')

    if not self.silent:
      print('- Preparing titles...')
    self.df['Title'] = self.df.Name.map(lambda n: re.sub('(.*, )|(\\..*)', '', n))
    miss_titles = ['Mlle', 'Ms', 'Mme']
    rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
    self.df.loc[(self.df.Title.isin(rare_titles)), 'Title'] = 'Rare title'
    self.df.loc[(self.df.Title.isin(miss_titles)), 'Title'] = 'Miss'
    if not self.silent:
      print('... done preparing data!')
      print(self.SEPARATOR)


  def analyze_data(self):
    if not self.silent:
      print('Calculating some statistics...')
    total_passengers = len(self.df.index)
    survivor_count = len (self.df[self.df.Survived == 1])
    if not self.silent:
      print('- Survivors/passengers (ratio): %d/%d (%f)' % (survivor_count, total_passengers, survivor_count/total_passengers))

    women_on_board =  len(self.df[self.df.Sex == 'female'])
    female_survivors = len(self.df[(self.df.Survived == 1) & (self.df.Sex == 'female')])
    men_on_board =  len(self.df[self.df.Sex == 'male'])
    male_survivors = len(self.df[(self.df.Survived == 1) & (self.df.Sex == 'male')])
    if not self.silent:
      print('- Women/men on board (survival rate): %d/%d (%f/%f)' % (women_on_board, men_on_board, female_survivors/women_on_board, male_survivors/men_on_board))

    class_counts = np.zeros(3)
    for i in range(0, 3):
      class_counts[i] = len(self.df[self.df.Pclass == i+1])
    class_survival = np.zeros((2,3))
    for i in range(0, 2):
      for j in range(0, 3):
        class_survival[i,j] = len(self.df[(self.df.Survived == i) & (self.df.Pclass == j+1)]) / class_counts[j]
    if not self.silent:
      print('- Chance of survival by class:')
      print(class_survival)

    embark_counts = np.zeros(3)
    for i in range(0, 3):
      embark_counts[i] = len(self.df[self.df.Embarked == i])
    # if not self.silent:
    #   print('- Number of embarkers per port: %s' % embark_counts)

    embark_survival = np.zeros((2,3))
    for i in range(0, 2):
      for j in range(0, 3):
        embark_survival[i,j] = len(self.df[(self.df.Survived == i) & (self.df.Embarked == j)]) / embark_counts[j]
    if not self.silent:
      print('- Chance of survival by port:')
      print(embark_survival)

    if not self.silent:
      print('- Unique siblings: %s' % self.df.SibSp.unique())
    for sib in self.df.SibSp.unique():
      counts = len(self.df[self.df.SibSp == sib])
      if not self.silent:
        print('%d siblings: %d times, rate: %f' % (sib, counts, len(self.df[(self.df.SibSp == sib) & (self.df.Survived == 1)]) / counts))

    if not self.silent:
      print('- Unique parch: %s' % self.df.Parch.unique())
    for parch in self.df.Parch.unique():
      counts = len(self.df[self.df.Parch == parch])
      if not self.silent:
        print('%d parchlings: %d times, rate: %f' % (parch, counts, len(self.df[(self.df.Parch == parch) & (self.df.Survived == 1)]) / counts))

    if not self.silent:
      print('- Unique family size: %s' % self.df.FamilySize.unique())
    for family in self.df.FamilySize.unique():
      counts = len(self.df[self.df.FamilySize == family])
      if not self.silent:
        print('%d familylings: %d times, rate: %f' % (family, counts, len(self.df[(self.df.FamilySize == family) & (self.df.Survived == 1)]) / counts))

    if not self.silent:
      print('- Unique groups: %s' % self.df.FamilyGroup.unique())
    for family in self.df.FamilyGroup.unique():
      counts = len(self.df[self.df.FamilyGroup == family])
      if not self.silent:
        print('family group %s: %d times, rate: %f' % (family, counts, len(self.df[(self.df.FamilyGroup == family) & (self.df.Survived == 1)]) / counts))

    if not self.silent:
      print('- Unique titles: %s' % self.df.Title.unique())
    for title in self.df.Title.unique():
      counts = len(self.df[self.df.Title == title])
      if not self.silent:
        print('title %s: %d times, rate: %f' % (title, counts, len(self.df[(self.df.Title == title) & (self.df.Survived == 1)]) / counts))

    if not self.silent:
      print('... done printing statistics!')
      print(self.SEPARATOR)

  def _plot_features(self, feature_1, feature_2, feature_1_map = None, feature_2_map = None):
    not_survived = self.df.loc[(self.df.Survived == 0)]
    survived = self.df.loc[(self.df.Survived == 1)]

    if feature_1_map is not None:
      not_survived[feature_1] = not_survived[feature_1].map(feature_1_map)
      survived[feature_1] = survived[feature_1].map(feature_1_map)
      plot.xticks(list(feature_1_map.values()), feature_1_map.keys())
    else:
      plot.xticks(not_survived[feature_1].unique())

    if feature_2_map is not None:
      not_survived[feature_2] = not_survived[feature_2].map(feature_2_map)
      survived[feature_2] = survived[feature_2].map(feature_2_map)
      plot.yticks(list(feature_2_map.values()), feature_2_map.keys())
    else:
      plot.yticks(not_survived[feature_2].unique())

    not_survived_counted = not_survived.groupby([feature_1, feature_2]).size().reset_index()
    not_survived_counted['Ratio'] = not_survived_counted[0] / not_survived_counted[0].sum()
    survived_counted = survived.groupby([feature_1, feature_2]).size().reset_index()
    survived_counted['Ratio'] = survived_counted[0] / survived_counted[0].sum()

    plot.xlabel(feature_1)
    plot.ylabel(feature_2)
    plot.scatter(not_survived_counted[feature_1], not_survived_counted[feature_2], not_survived_counted.Ratio * 10000, c='r', alpha=0.5)
    plot.scatter(survived_counted[feature_1], survived_counted[feature_2], survived_counted.Ratio * 10000, c='b', alpha=0.5)

  def create_some_plots(self):
    plot.subplot(5, 3, 1)
    self._plot_features('FamilySize', 'Pclass')
    plot.subplot(5, 3, 4)
    self._plot_features('FamilySize', 'Sex', None, {'male': 0, 'female': 1})
    plot.subplot(5, 3, 7)
    self._plot_features('FamilySize', 'Embarked', None, {'S': 0, 'C': 1, 'Q': 2})
    plot.subplot(5, 3, 10)
    self._plot_features('FamilySize', 'FamilyGroup', None, {'singleton': 0, 'small family': 1, 'large family': 2})
    plot.subplot(5, 3, 13)
    self._plot_features('FamilySize', 'Title', None, {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare title': 4})

    plot.subplot(5, 3, 2)
    self._plot_features('Title', 'Pclass', {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare title': 4})
    plot.subplot(5, 3, 5)
    self._plot_features('Title', 'Sex', {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare title': 4}, {'male': 0, 'female': 1})
    plot.subplot(5, 3, 8)
    self._plot_features('Title', 'Embarked', {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare title': 4}, {'S': 0, 'C': 1, 'Q': 2})
    plot.subplot(5, 3, 11)
    self._plot_features('Title', 'FamilyGroup', {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare title': 4}, {'singleton': 0, 'small family': 1, 'large family': 2})
    plot.subplot(5, 3, 14)
    self._plot_features('Title', 'FamilyGroup', {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare title': 4}, {'singleton': 0, 'small family': 1, 'large family': 2})

    plot.subplot(5, 3, 3)
    self._plot_features('AgeGroup', 'Pclass')
    plot.subplot(5, 3, 6)
    self._plot_features('AgeGroup', 'Sex', None, {'male': 0, 'female': 1})
    plot.subplot(5, 3, 9)
    self._plot_features('AgeGroup', 'Embarked', None, {'S': 0, 'C': 1, 'Q': 2})
    plot.subplot(5, 3, 12)
    self._plot_features('AgeGroup', 'FamilyGroup', None, {'singleton': 0, 'small family': 1, 'large family': 2})
    plot.subplot(5, 3, 15)
    self._plot_features('AgeGroup', 'Title', None, {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare title': 4})

    # plot.tight_layout()
    plot.show()

