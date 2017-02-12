import sys

from utils import Utils

class AreaClass:
  areas = None
  scaled_areas = None
  area_mask = None

  def __init__(self, id, color):
    self.id = id
    self.color = color

  def set_areas(self, areas, image_size, x_scale, y_scale):
    print('%s...' % self.id, end=' ')
    sys.stdout.flush()
    self.areas = areas

    self.x_scale = x_scale
    self.y_scale = y_scale

    self.scaled_areas = Utils.scale_multi_polygon(self.areas, x_scale, y_scale)

    self.area_mask = Utils.multi_polygon_to_pixel_mask(self.scaled_areas, image_size)

    self.mask_image = Utils.pixel_mask_to_image(self.area_mask, \
        self.color[0], self.color[1], self.color[2])

  def set_predicted_areas(self, predicted_areas, image_size):
    self.predicted_areas = predicted_areas

    self.predicted_area_mask = Utils.multi_polygon_to_pixel_mask(self.predicted_areas, image_size)
    self.predicted_mask_image = Utils.pixel_mask_to_image(self.predicted_area_mask, \
        self.color[0], self.color[1], self.color[2])

    self.predicted_submission_polygons = Utils.scale_multi_polygon(self.predicted_areas, 1/self.x_scale, 1/self.y_scale)

