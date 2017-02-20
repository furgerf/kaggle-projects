import pickle

class Scan(object):

  def __init__(self, scan_id, label=None, data_index=0):
    self.scan_id = scan_id
    self.data_index = data_index
    self.label = label
    self._data = None

  @property
  def data(self):
    if self._data is None:
      with open('./preprocessed/{}.pickle'.format(self.scan_id), "rb") as f:
        self._data = pickle.load(f)[self.data_index]

    return self._data

