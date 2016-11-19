#!/usr/bin/python

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from cross_validation_result import CrossValidationResult
from titanic import TitanicKaggle

# set up
start_time = datetime.utcnow()
print('')
print('')
print('')
print('Evaluation start time: %s' % start_time)
titanic = TitanicKaggle(lambda: RandomForestClassifier(n_estimators=100, random_state=123))
titanic.initialize()

# analyze
# titanic.print_sample_data()
# titanic.analyze_data()

# prepare for prediction
train, test = titanic.split_data()

passenger_id = 'PassengerId'
survived = 'Survived'
header = [passenger_id, survived]
predictor = train[survived]
ids = test[passenger_id]

all_features = ['Embarked', 'Parch', 'SibSp', 'Pclass', 'Gender', 'AgeGroup', 'FamilySize', 'FamilyGroup', 'Title']
features_sets = []

features_sets.append(all_features)
for i in range(len(all_features)):
  new_feature_set = all_features.copy()
  new_feature_set.remove(all_features[i])
  features_sets.append(new_feature_set)

  for j in range(len(all_features) - 1):
    very_new_feature_set = new_feature_set.copy()
    very_new_feature_set.remove(new_feature_set[j])
    features_sets.append(very_new_feature_set)

five_fold_results = []
ten_fold_results = []

for features in features_sets:
  current_train = TitanicKaggle.numericalize_data(train[features])
  current_test = TitanicKaggle.numericalize_data(test[features])

  # predict
  five_fold_results.append(titanic.cross_validate(current_train, predictor, folds=5))
  ten_fold_results.append(titanic.cross_validate(current_train, predictor, folds=10))

print(TitanicKaggle.SEPARATOR)
for i, features in enumerate(features_sets):
  print('Features: %s' % features)
  # five_fold_results[i].print_short_results()
  # ten_fold_results[i].print_short_results()

  accuracies = [CrossValidationResult.mean(five_fold_results[i].accuracies), CrossValidationResult.mean(ten_fold_results[i].accuracies)]
  f1_scores = [CrossValidationResult.mean(five_fold_results[i].f1_scores), CrossValidationResult.mean(ten_fold_results[i].f1_scores)]

  print('AVG:\t\t\tACC=%f\tF1=%f' % (CrossValidationResult.mean(accuracies), CrossValidationResult.mean(f1_scores)))
  print(TitanicKaggle.SEPARATOR)
print(TitanicKaggle.SEPARATOR)

# titanic.predict_test_data(train, predictor, test, ids, header)

end_time = datetime.utcnow()
duration = end_time - start_time
print('Evaluation end time: %s' % end_time)
print('Total evaluation duration: %s' % duration)

