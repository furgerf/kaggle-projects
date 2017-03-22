import pickle

class Scan(object):

  def __init__(self, scan_id, label=None, data_index=None):
    self.scan_id = scan_id
    self.data_index = data_index
    self.label = label
    self.predicted_label = None
    self._data = None

  @property
  def data(self):
    if self._data is None:
      with open('./preprocessed/{}.pickle'.format(self.scan_id), "rb") as f:
        tmp = pickle.load(f)
        self._data = tmp[self.data_index] if self.data_index is not None else tmp

    return self._data

