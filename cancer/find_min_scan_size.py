import pickle
import os

min_size = [123456789, 123456789, 123456789]
min_length = 987654321
with open('scan-sizes', 'w') as fh:
  for i, file_name in enumerate(os.listdir('./preprocessed')):
    print(i)
    with open('./preprocessed/' + file_name, "rb") as f:
      data = pickle.load(f)[0]
      # min_size[0] = min(min_size[0], data.shape[0])
      # min_size[1] = min(min_size[1], data.shape[1])
      # min_size[2] = min(min_size[2], data.shape[2])
      # min_length = min(min_length, data.shape[0] * data.shape[1] * data.shape[2])
      fh.write(','.join(map(str, data.shape)) + '\n')

# print(min_length)

