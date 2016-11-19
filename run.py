#!/usr/bin/python

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from cross_validation_result import CrossValidationResult
from titanic_kaggle import TitanicKaggle

def experiment():
  # set up experiment titanic instance
  titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=100, random_state=123))
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
  titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=100, random_state=123))
  titanic.initialize()

  # prepare data for prediction
  train, test = titanic.split_data()
  training_predictors = train[titanic.PREDICTOR_COLUMN_NAME]
  ids = test[titanic.ID_COLUMN_NAME]

  header = [titanic.ID_COLUMN_NAME, titanic.PREDICTOR_COLUMN_NAME]

  TitanicKaggle.ALL_FEATURES = ['Embarked', 'Parch', 'SibSp', 'Pclass', 'Gender', 'AgeGroup', 'FamilySize', 'FamilyGroup', 'Title']
  prepared_train_data = TitanicKaggle.numericalize_data(train[TitanicKaggle.ALL_FEATURES])
  prepared_test_data = TitanicKaggle.numericalize_data(test[TitanicKaggle.ALL_FEATURES])

  titanic.predict_test_data(prepared_train_data, training_predictors, prepared_test_data, ids, header)


# set up
start_time = datetime.utcnow()
print('')
print('')
print('')
print(TitanicKaggle.SEPARATOR)
print('Evaluation start time: %s' % start_time)
print(TitanicKaggle.SEPARATOR)


experiment()
predict_test_data()


end_time = datetime.utcnow()
duration = end_time - start_time
print('Evaluation end time: %s' % end_time)
print('Total evaluation duration: %s' % duration)
print(TitanicKaggle.SEPARATOR)

