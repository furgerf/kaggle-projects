import pickle
import numpy as np

class Scan(object):

  def __init__(self, scan_id, label=None, data_index=None):
    """
    scan_id:    (string) ID of the scan, this is expected to be the file name
    label:      (any)    Label of the scan, defaults to None (label unknown)
    data_index: (int)    Index to use when retrieving data from pickle (for tuple/array),
                         defaults to None (apply no index)
    """
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
        self._data = np.array(self._data) / 1000

    return self._data

