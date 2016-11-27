#!/usr/bin/python

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from cross_validation_result import CrossValidationResult
from titanic_kaggle import TitanicKaggle
from feature_finder_2 import FeatureFinder

def experiment():
  print(TitanicKaggle.SEPARATOR)
  print('EXPERIMENTING')
  print(TitanicKaggle.SEPARATOR)

  # set up experiment titanic instance
  titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=50, random_state=123))
  titanic.initialize()

  # analyze
  titanic.print_sample_data()
  titanic.analyze_data()

  # prepare data for prediction
  train, predictors, test, ids = titanic.get_prepared_data(TitanicKaggle.ALL_FEATURES)

  titanic.cross_validate(train, predictors).print_results()


def predict_test_data():
  print(TitanicKaggle.SEPARATOR)
  print('PREDICTING TEST DATA')
  print(TitanicKaggle.SEPARATOR)

  # set up experiment titanic instance
  titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=50, random_state=123))
  titanic.initialize()

  # prepare data for prediction
  train, predictors, test, ids = titanic.get_prepared_data(TitanicKaggle.ALL_FEATURES)

  all_feature_sets = [
    ['AgeGroup', 'FamilyGroup-0', 'FamilyGroup-1', 'FamilyGroup-2', 'Pclass', 'Sex-0', 'Sex-1'],
    ]

  # features = ['Parch', 'Pclass', 'Sex', 'AgeGroup', 'FamilyGroup', 'Title']
  for feature_set in all_feature_sets:
    current_train = train[feature_set]
    current_test = test[feature_set]
    titanic.cross_validate(current_train, predictors).print_results()
    titanic.predict_test_data(current_train, predictors, current_test, ids)


def find_features():
  print(TitanicKaggle.SEPARATOR)
  print('FINDING FEATURES')
  print(TitanicKaggle.SEPARATOR)

  random_seeds = [7, 42, 123, 6340, 43627]
  classifiers = list(map(lambda seed: RandomForestClassifier(n_estimators=50, random_state=seed), random_seeds))
  dummy_titanic = TitanicKaggle(None, None)
  dummy_titanic.initialize()
  train, predictors, test, ids = dummy_titanic.get_prepared_data(TitanicKaggle.ALL_FEATURES)

  finder = FeatureFinder(classifiers, train, predictors, 'f1', 3, 9)
  finder_results = finder.run_predictions()
  finder.evaluate_feature_finder_results(finder_results)


def main():
  # set up
  start_time = datetime.utcnow()
  print('')
  print('')
  print('')
  print(TitanicKaggle.SEPARATOR)
  print('Evaluation start time: %s' % start_time)
  print(TitanicKaggle.SEPARATOR)


  # experiment()
  predict_test_data()
  # find_features()

  end_time = datetime.utcnow()
  duration = end_time - start_time
  print('Evaluation end time: %s' % end_time)
  print('Total evaluation duration: %s' % duration)
  print(TitanicKaggle.SEPARATOR)


if __name__ == "__main__":
  main()

