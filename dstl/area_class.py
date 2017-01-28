
import numpy as np
import cv2
import shapely.affinity

class AreaClass:
  areas = None
  scaled_areas = None
  area_mask = None

  def __init__(self, id, color):
    self.id = id
    self.color = color

  def set_areas(self, areas, image_size, x_scale, y_scale):
    self.areas = areas

    self.scaled_areas = shapely.affinity.scale(self.areas, \
        xfact=x_scale, yfact=y_scale, origin=(0, 0, 0))

    self.area_mask = self._get_mask_for_areas(image_size)

    self.mask_image = np.stack([
      np.multiply(self.area_mask, self.color[0]),
      np.multiply(self.area_mask, self.color[1]),
      np.multiply(self.area_mask, self.color[2])
      ]).transpose([1, 2, 0])

  def _get_mask_for_areas(self, image_size):
    mask = np.zeros(image_size, np.uint8)

    # if not areas:
      # return mask_image

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in self.scaled_areas]
    interiors = [int_coords(pi.coords) for poly in self.scaled_areas for pi in poly.interiors]
    cv2.fillPoly(mask, exteriors, 1)
    cv2.fillPoly(mask, interiors, 0)

    return mask

