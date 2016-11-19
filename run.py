#!/usr/bin/python

from sklearn.ensemble import RandomForestClassifier

from kaggle import Kaggle
from titanic import TitanicKaggle

# set up
print('')
print('')
print('')
titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=100, random_state=123))
titanic.initialize()

# analyze
titanic.print_sample_data()
titanic.analyze_data()

# prepare for prediction
train, test = titanic.split_data()

passenger_id = 'PassengerId'
survived = 'Survived'
header = [passenger_id, survived]
predictor = train[survived]
ids = test[passenger_id]

features = ['Embarked', 'Parch', 'SibSp', 'Pclass', 'Gender', 'AgeGroup', 'FamilySize', 'FamilyGroup', 'Title']
train = Kaggle.numericalize_data(train[features])
test = Kaggle.numericalize_data(test[features])

# predict
results = titanic.cross_validate(train, predictor)
Kaggle.evaluate_cross_validation_results(results)

titanic.predict_test_data(train, predictor, test, ids, header)

