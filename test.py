import csv as csv
import numpy as np

def print_survival_stats(data, index):
  num_passengers = np.size(data[index,1].astype(np.float))
  survivors = np.sum(data[index,1].astype(np.float))
  survival_rate = survivors / num_passengers
  print('passengers: %d, survivors: %d, rate: %f' % (num_passengers, survivors, survival_rate))

def load_data():
  with open('data/train.csv', 'rt') as train:
    train_csv = csv.reader(train)

    header = next(train_csv)
    print(header)

    data = []

    for row in train_csv:
      data.append(row)

    data = np.array(data)
    print('sample row: %s' % data[0])
    print('-------')

    return data

def print_some_stats():
  pclass = data[0::,2].astype(np.float)

  index_females = data[0::,4] == "female"
  index_males = data[0::,4] != "female"

  print('all:')
  print_survival_stats(data, np.repeat(True, np.size(data[0::,1])))
  print('-------')
  print('females:')
  print_survival_stats(data, index_females)
  print('males:')
  print_survival_stats(data, index_males)
  print('-------')

  index_age_set = data[0::,5] != ''
  age_data = data[index_age_set]

  bins = 5
  ages = age_data[0::,5].astype(np.float)
  age_histogram = np.histogram(ages, bins=bins)
  age_counts = age_histogram[0]
  age_bins = age_histogram[1]

  age_bins[bins - 1] = age_bins[bins - 1] + 1

  print('people with age:')
  print_survival_stats(age_data, np.repeat(True, np.size(age_data[0::,1])))
  print('-------')

  for i in range(0, bins):
    print('ages from %d to %d:' % (age_bins[i], age_bins[i+1]))
    index_age_range = np.array([a >= age_bins[i] and a < age_bins[i+1] for a in ages])
    print_survival_stats(age_data, index_age_range)

def print_some_other_stats():
  embarks = data[0::,11]
  unique_embarks = np.unique(embarks)
  print(unique_embarks)
  #embark_to_index = lambda e: np.where(unique_embarks == e)[0][0]

  #indexer = np.vectorize(embark_to_index)
  #indexed_embarks = indexer(embarks)

  index_embark_s = data[0::, 11] == 'S'
  index_embark_c = data[0::, 11] == 'C'
  index_embark_q = data[0::, 11] == 'Q'
  index_embark_empty = data[0::, 11] == ''

  print_survival_stats(data, index_embark_s)
  print_survival_stats(data, index_embark_c)
  print_survival_stats(data, index_embark_q)
  print_survival_stats(data, index_embark_empty)

  #print(np.bincount(data[0::,11]))

data = load_data()
print_some_other_stats()

