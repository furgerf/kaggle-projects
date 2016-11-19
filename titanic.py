#!/usr/bin/python

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import re

from kaggle import Kaggle

class TitanicKaggle(Kaggle):
  def __init__(self, classifier_creator):
    # TODO: determine base class dynamically
    Kaggle.__init__(self, 'data/train.csv', 'data/test.csv', 'prediction.csv', classifier_creator)

  def _engineer_features(self):
    print('Preparing data...')

    print('- Converting gender...')
    genders = {'female': 0, 'male': 1}
    self.df['Gender'] = self.df['Sex'].map(genders)

    print('- Converting ports...')
    most_common_port = self.df.Embarked.dropna().mode().values[0]
    print('  using most common port "%s" for null-entries' % most_common_port)
    self.df.loc[self.df.Embarked.isnull(), 'Embarked'] = most_common_port
    ports = list(enumerate(np.unique(self.df['Embarked'])))
    ports_dict = {name: i for i, name in ports}
    self.df.Embarked = self.df.Embarked.map(ports_dict)

    print('- Preparing fares...')
    mean_fares = np.zeros((3))
    print('  calculating class-based mean fares')
    for i in range(0, 3):
      mean_fares[i] = self.df[self.df.Pclass == i+1].Fare.mean()
    print('  filling fares with means')
    for i in range(0, 3):
      self.df.loc[(self.df.Fare.isnull()) & (self.df.Pclass == i+1), 'Fare'] = mean_fares[i]

    print('- Preparing ages...')
    median_ages = np.zeros((2,3))
    print('  calculating gender- and class-based median ages')
    for i in range(0, 2):
      for j in range(0, 3):
        median_ages[i,j] = self.df[(self.df.Gender == i) & (self.df.Pclass == j+1)].Age.dropna().median()
    print('  filling missing ages with medians')
    for i in range(0, 2):
      for j in range(0, 3):
        self.df.loc[(self.df.Age.isnull()) & (self.df.Gender == i) & (self.df.Pclass == j+1), 'Age'] = median_ages[i,j]
    age_bins = np.linspace(self.df.Age.min(),self.df.Age.max(), 20)
    # print('  grouping ages with bins: %s' % age_bins)
    self.df['AgeGroup'] = np.digitize(self.df.Age, age_bins)

    print('- Preparing families...')
    self.df['FamilySize'] = self.df.SibSp + self.df.Parch

    print('- Preparing family groups...')
    self.df['FamilyGroup'] = self.df.FamilySize.map(lambda s: 0 if s == 0 else 2 if s > 3 else 1)

    print('- Preparing titles...')
    self.df['Title'] = self.df.Name.map(lambda n: re.sub('(.*, )|(\\..*)', '', n))
    self.df.loc[(self.df.Title == 'Mlle'), 'Title'] = 'Miss'
    self.df.loc[(self.df.Title == 'Ms'), 'Title'] = 'Miss'
    self.df.loc[(self.df.Title == 'Mme'), 'Title'] = 'Miss'
    rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
    self.df.loc[(self.df.Title.isin(rare_titles)), 'Title'] = 'Rare title'
    titles = {'Rare title': 0, 'Master': 1, 'Mr': 2, 'Miss': 3, 'Mrs': 4}
    self.df.Title = self.df.Title.map(titles).astype(int)
    # group = self.df.Title.groupby(self.df.Title).agg([np.count_nonzero])
    # print(group)
    print('... done preparing data!')
    print(self.SEPARATOR)

  def print_stats(self):
    print('Calculating some statistics...')
    total_passengers = len(self.df.index)
    survivor_count = len (self.df[self.df.Survived == 1])
    print('- Survivors/passengers (ratio): %d/%d (%f)' % (survivor_count, total_passengers, survivor_count/total_passengers))

    women_on_board =  len(self.df[self.df.Gender == 0])
    female_survivors = len(self.df[(self.df.Survived == 1) & (self.df.Gender == 0)])
    men_on_board =  len(self.df[self.df.Gender == 1])
    male_survivors = len(self.df[(self.df.Survived == 1) & (self.df.Gender == 1)])
    print('- Women/men on board (survival rate): %d/%d (%f/%f)' % (women_on_board, men_on_board, female_survivors/women_on_board, male_survivors/men_on_board))

    class_counts = np.zeros(3)
    for i in range(0, 3):
      class_counts[i] = len(self.df[self.df.Pclass == i+1])
    class_survival = np.zeros((2,3))
    for i in range(0, 2):
      for j in range(0, 3):
        class_survival[i,j] = len(self.df[(self.df.Survived == i) & (self.df.Pclass == j+1)]) / class_counts[j]
    print('- Chance of survival by class:')
    print(class_survival)

    embark_counts = np.zeros(3)
    for i in range(0, 3):
      embark_counts[i] = len(self.df[self.df.Embarked == i])
    #print('- Number of embarkers per port: %s' % embark_counts)

    embark_survival = np.zeros((2,3))
    for i in range(0, 2):
      for j in range(0, 3):
        embark_survival[i,j] = len(self.df[(self.df.Survived == i) & (self.df.Embarked == j)]) / embark_counts[j]
    print('- Chance of survival by port:')
    print(embark_survival)

    print('- Unique siblings: %s' % self.df.SibSp.unique())
    for sib in self.df.SibSp.unique():
      counts = len(self.df[self.df.SibSp == sib])
      print('%d siblings: %d times, rate: %f' % (sib, counts, len(self.df[(self.df.SibSp == sib) & (self.df.Survived == 1)]) / counts))

    print('- Unique parch: %s' % self.df.Parch.unique())
    for parch in self.df.Parch.unique():
      counts = len(self.df[self.df.Parch == parch])
      print('%d parchlings: %d times, rate: %f' % (parch, counts, len(self.df[(self.df.Parch == parch) & (self.df.Survived == 1)]) / counts))

    print('- Unique family size: %s' % self.df.FamilySize.unique())
    for family in self.df.FamilySize.unique():
      counts = len(self.df[self.df.FamilySize == family])
      print('%d familylings: %d times, rate: %f' % (family, counts, len(self.df[(self.df.FamilySize == family) & (self.df.Survived == 1)]) / counts))

    print('- Unique groups: %s' % self.df.FamilyGroup.unique())
    for family in self.df.FamilyGroup.unique():
      counts = len(self.df[self.df.FamilyGroup == family])
      print('family group %d: %d times, rate: %f' % (family, counts, len(self.df[(self.df.FamilyGroup == family) & (self.df.Survived == 1)]) / counts))

    print('- Unique titles: %s' % self.df.Title.unique())
    for title in self.df.Title.unique():
      counts = len(self.df[self.df.Title == title])
      print('title %d: %d times, rate: %f' % (title, counts, len(self.df[(self.df.Title == title) & (self.df.Survived == 1)]) / counts))

    print('... done printing statistics!')
    print(self.SEPARATOR)


# set up
titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=100))
titanic.initialize()

# analyze
titanic.print_head()
titanic.print_stats()

# prepare for prediction
train, test = titanic.split_data()

passenger_id = 'PassengerId'
survived = 'Survived'
header = [passenger_id, survived]
predictor = train[survived]
ids = test[passenger_id]

features = ['Embarked', 'Parch', 'SibSp', 'Pclass', 'Gender', 'AgeGroup', 'FamilySize', 'FamilyGroup', 'Title']
train = titanic.integerize_data(train[features])
test = titanic.integerize_data(test[features])

# predict
results = titanic.cross_validate(train, predictor)
titanic.evaluate_cross_validation_results(results)
titanic.predict_test_data(train, predictor, test, ids, header)

