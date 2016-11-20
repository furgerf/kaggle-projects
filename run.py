#!/usr/bin/python

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from cross_validation_result import CrossValidationResult
from titanic_kaggle import TitanicKaggle
from feature_finder import FeatureFinder

def experiment():
  # set up experiment titanic instance
  titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=50, random_state=123))
  titanic.initialize()

  # analyze
  titanic.print_sample_data()
  titanic.analyze_data()

  # prepare data for prediction
  train, test = titanic.split_data()
  training_predictors = train[titanic.PREDICTOR_COLUMN_NAME]

  prepared_train_data = TitanicKaggle.numericalize_data(train[TitanicKaggle.ALL_FEATURES])
  prepared_test_data = TitanicKaggle.numericalize_data(test[TitanicKaggle.ALL_FEATURES])

  titanic.cross_validate(prepared_train_data, training_predictors).print_results()

def predict_test_data():
  # set up experiment titanic instance
  titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=50, random_state=123))
  titanic.initialize()

  # prepare data for prediction
  train, test = titanic.split_data()
  training_predictors = train[titanic.PREDICTOR_COLUMN_NAME]
  ids = test[titanic.ID_COLUMN_NAME]

  header = [titanic.ID_COLUMN_NAME, titanic.PREDICTOR_COLUMN_NAME]

  features = ['Parch', 'Pclass', 'Gender', 'AgeGroup', 'FamilyGroup', 'Title']
  prepared_train_data = TitanicKaggle.numericalize_data(train[features])
  prepared_test_data = TitanicKaggle.numericalize_data(test[features])

  titanic.predict_test_data(prepared_train_data, training_predictors, prepared_test_data, ids, header)

def find_features():
  random_seeds = [7, 42, 123, 6340, 43627]
  titanics = list(map(lambda seed: TitanicKaggle(lambda: RandomForestClassifier(n_estimators=50, random_state=seed)), random_seeds))
  for titanic in titanics:
    titanic.initialize()
  finder = FeatureFinder(titanics)
  finder_results = finder.run_predictions()
  FeatureFinder.evaluate_feature_finder_results(finder_results)


# set up
start_time = datetime.utcnow()
print('')
print('')
print('')
print(TitanicKaggle.SEPARATOR)
print('Evaluation start time: %s' % start_time)
print(TitanicKaggle.SEPARATOR)


experiment()
# predict_test_data()
# find_features()

end_time = datetime.utcnow()
duration = end_time - start_time
print('Evaluation end time: %s' % end_time)
print('Total evaluation duration: %s' % duration)
print(TitanicKaggle.SEPARATOR)

