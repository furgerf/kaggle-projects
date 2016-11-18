import csv as csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SEPARATOR = '-' * 80
print('')
print('')
print('')
print('')
print(SEPARATOR)
print('STARTING...')
print(SEPARATOR)

def load_data():
  train_df = pd.read_csv('data/train.csv', header=0)
  test_df = pd.read_csv('data/test.csv', header=0)
  return train_df, test_df

def generate_age_bins(train, test):
  return np.linspace(min(train.Age.min(), test.Age.min()), max(train.Age.max(), test.Age.max()), 20)

def prepare_data(df, age_bins):
  print('Preparing data...')

  print('- Converting gender...')
  genders = {'female': 0, 'male': 1}
  df['Gender'] = df['Sex'].map(genders)

  print('- Converting ports...')
  most_common_port = df.Embarked.dropna().mode().values[0]
  print('  using most common port "%s" for null-entries' % most_common_port)
  df.loc[df.Embarked.isnull(), 'Embarked'] = most_common_port
  ports = list(enumerate(np.unique(df['Embarked'])))
  ports_dict = {name: i for i, name in ports}
  df.Embarked = df.Embarked.map(ports_dict)

  print('- Preparing fares...')
  mean_fares = np.zeros((3))
  print('  calculating class-based mean fares')
  for i in range(0, 3):
    mean_fares[i] = df[df.Pclass == i+1].Fare.mean()
  print('  filling fares with means')
  for i in range(0, 3):
    df.loc[(df.Fare.isnull()) & (df.Pclass == i+1), 'Fare'] = mean_fares[i]

  print('- Preparing ages...')
  median_ages = np.zeros((2,3))
  print('  calculating gender- and class-based median ages')
  for i in range(0, 2):
    for j in range(0, 3):
      median_ages[i,j] = df[(df.Gender == i) & (df.Pclass == j+1)].Age.dropna().median()
  print('  filling missing ages with medians')
  for i in range(0, 2):
    for j in range(0, 3):
      df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'Age'] = median_ages[i,j]
  print('  grouping ages with bins: %s' % age_bins)
  df['AgeGroup'] = np.digitize(df.Age, age_bins)

  print('- Preparing families...')
  df['FamilySize'] = df.SibSp + df.Parch

  print('- Dropping unused columns...')
  print('  dropping passenger id, name, ticket, cabin, sex...')
  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex'], axis=1)

  print('  also dropping fare, age...')
  df = df.drop(['Fare', 'Age', 'SibSp', 'Parch', 'Embarked'], axis=1)

  print('... done preparing data!')

  print(SEPARATOR)
  print('Sample rows:')
  print(SEPARATOR)
  print(df.head(5))
  print(SEPARATOR)

  return df

def print_stats(df):
  print('Calculating some statistics...')
  total_passengers = len(df.index)
  survivor_count = len (df[df.Survived == 1])
  print('- Survivors/passengers (ratio): %d/%d (%f)' % (survivor_count, total_passengers, survivor_count/total_passengers))

  women_on_board =  len(df[df.Gender == 0])
  female_survivors = len(df[(df.Survived == 1) & (df.Gender == 0)])
  men_on_board =  len(df[df.Gender == 1])
  male_survivors = len(df[(df.Survived == 1) & (df.Gender == 1)])
  print('- Women/men on board (survival rate): %d/%d (%f/%f)' % (women_on_board, men_on_board, female_survivors/women_on_board, male_survivors/men_on_board))

  class_counts = np.zeros(3)
  for i in range(0, 3):
    class_counts[i] = len(df[df.Pclass == i+1])
  class_survival = np.zeros((2,3))
  for i in range(0, 2):
    for j in range(0, 3):
      class_survival[i,j] = len(df[(df.Survived == i) & (df.Pclass == j+1)]) / class_counts[j]
  print('- Chance of survival by class:')
  print(class_survival)

  """
  embark_counts = np.zeros(3)
  for i in range(0, 3):
    embark_counts[i] = len(df[df.Embarked == i])
  #print('- Number of embarkers per port: %s' % embark_counts)

  embark_survival = np.zeros((2,3))
  for i in range(0, 2):
    for j in range(0, 3):
      embark_survival[i,j] = len(df[(df.Survived == i) & (df.Embarked == j)]) / embark_counts[j]
  print('- Chance of survival by port:')
  print(embark_survival)

  print('- Unique siblings: %s' % df.SibSp.unique())
  for sib in df.SibSp.unique():
    counts = len(df[df.SibSp == sib])
    print('%d siblings: %d times, rate: %f' % (sib, counts, len(df[(df.SibSp == sib) & (df.Survived == 1)]) / counts))

  print('- Unique parch: %s' % df.Parch.unique())
  for parch in df.Parch.unique():
    counts = len(df[df.Parch == parch])
    print('%d parchlings: %d times, rate: %f' % (parch, counts, len(df[(df.Parch == parch) & (df.Survived == 1)]) / counts))
    """

  print('- Unique family size: %s' % df.FamilySize.unique())
  for family in df.FamilySize.unique():
    counts = len(df[df.FamilySize == family])
    print('%d familylings: %d times, rate: %f' % (family, counts, len(df[(df.FamilySize == family) & (df.Survived == 1)]) / counts))

  print('... done printing statistics!')
  print(SEPARATOR)

def run_prediction(train, test):
  forest = RandomForestClassifier(n_estimators=100)
  forest = forest.fit(train[0::,1::], train[0::,0] )

  return forest.predict(test).astype(int)

def write_predictions(ids, predictions):
  with open('prediction.csv', 'wt') as predictions_file:
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['PassengerId','Survived'])
    open_file_object.writerows(zip(ids, predictions))
    predictions_file.close()

def predict(train, test):
  predictions = run_prediction(train.values, test.values)
  write_predictions(passenger_ids, predictions)

def cross_validate(data):
  folds = 10
  fold_size = int(len(data)/folds)

  tps = []
  fps = []
  fns = []
  tns = []
  accs = []
  precs = []
  recs = []
  f1s = []

  print('Starting %d-fold cross-validation with fold size %d...' % (folds, fold_size))

  for i in range(folds):
    test = data[i*fold_size:(i+1)*fold_size]
    answers = np.array(test.Survived)
    test = test.drop('Survived', axis=1)

    train = pd.concat([data[:i * fold_size:], data[(i+1) * fold_size:]])

    print('- running prediction of fold %d...' % i)
    predictions = run_prediction(train.values, test.values)
    tp = np.sum([(x == y == 1) for x, y in zip(predictions, answers)])
    fp = np.sum([(x == 1 and y == 0) for x, y in zip(predictions, answers)])
    fn = np.sum([(x == 0 and y == 1) for x, y in zip(predictions, answers)])
    tn = np.sum([(x == y == 0) for x, y in zip(predictions, answers)])
    if len(predictions) != fold_size or tp + fp + fn + tn != fold_size:
      raise Error('Unexpected number of prediction results!!')

    print('    TP:  %f,\tFP:   %f,\tFN:  %f,\tTN: %f' % (tp/fold_size, fp/fold_size, fn/fold_size, tn/fold_size))
    acc = (tp + tn) / fold_size
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    print('    acc: %f,\tprec: %f,\trec: %f,\tf1: %f' % (acc, prec, rec, f1))

    tps.append(tp/fold_size)
    fps.append(fp/fold_size)
    fns.append(fn/fold_size)
    tns.append(tn/fold_size)
    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)

  print('... finished running cros-validation!')
  print(SEPARATOR)

  return (tps, fps, fns, tns, accs, precs, recs, f1s)

def evaluate_cross_validation_results(results):
  tps, fps, fns, tns, accs, precs, recs, f1s = results
  print('Cross-validation results')
  print(SEPARATOR)
  print('TP:\tmin=%f\tmean=%f\tmax=%f' % (min(tps), sum(tps)/len(tps), max(tps)))
  print('FP:\tmin=%f\tmean=%f\tmax=%f' % (min(fps), sum(fps)/len(fps), max(fps)))
  print('FN:\tmin=%f\tmean=%f\tmax=%f' % (min(fns), sum(fns)/len(fns), max(fns)))
  print('TN:\tmin=%f\tmean=%f\tmax=%f' % (min(tns), sum(tns)/len(tns), max(tns)))
  print('ACC:\tmin=%f\tmean=%f\tmax=%f' % (min(accs), sum(accs)/len(accs), max(accs)))
  print('PREC:\tmin=%f\tmean=%f\tmax=%f' % (min(precs), sum(precs)/len(precs), max(precs)))
  print('REC:\tmin=%f\tmean=%f\tmax=%f' % (min(recs), sum(recs)/len(recs), max(recs)))
  print('F1:\tmin=%f\tmean=%f\tmax=%f' % (min(f1s), sum(f1s)/len(f1s), max(f1s)))
  print(SEPARATOR)

train_df, test_df = load_data()

# pre-prepare data
age_bins = generate_age_bins(train_df, test_df)
passenger_ids = test_df.PassengerId.values

# prepare data
train = prepare_data(train_df, age_bins)
test = prepare_data(test_df, age_bins)

print_stats(train)
evaluate_cross_validation_results(cross_validate(train))
predict(train, test)

