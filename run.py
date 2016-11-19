#!/usr/bin/python

from sklearn.ensemble import RandomForestClassifier

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
train = TitanicKaggle.numericalize_data(train[features])
test = TitanicKaggle.numericalize_data(test[features])

# predict
results = titanic.cross_validate(train, predictor, folds=5)
results.print_results()

titanic.predict_test_data(train, predictor, test, ids, header)

