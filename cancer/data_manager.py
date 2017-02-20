import os

class DataManager(object):
  def __init__(self, scan_directory):
    self.scans = os.listdir(scan_directory)

