import logging

import tifffile as tiff

from area_classes import AreaClasses

class Image:
  def __init__(self, image_id, grid, data_directory):
    self.image_id = image_id
    self.grid = grid
    self.data_file = '{}three_band/{}.tif'.format(data_directory, image_id)
    self.log = logging.getLogger('dstl')

  def load_image(self):
    self.log.info('Loading image {}'.format(self.image_id))
    self.raw_data = tiff.imread(self.data_file).transpose([1, 2, 0])

    self.image_size = self.raw_data.shape[:2]
    self.height, self.width = self.image_size
    self.x_scale, self.y_scale = Image.get_scale_factors_for_image(self.grid, self.width, self.height)

  @staticmethod
  def get_scale_factors_for_image(grid, width, height):
    width = width * (width / (width + 1))
    height = height * (height / (height + 1))
    return width / grid[0], height / grid[1]

  def load_areas(self, areas):
    self.log.warning('Loading areas for image {}'.format(self.image_id))
    self.area_classes = AreaClasses(self.image_id, self.image_size, self.x_scale, self.y_scale)
    self.area_classes.load(areas)

