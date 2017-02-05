import numpy as np
import shapely.affinity
import cv2

class Utils:
  @staticmethod
  def multi_polygon_to_pixel_mask(polygon, image_size):
    mask = np.zeros(image_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygon]
    interiors = [int_coords(pi.coords) for poly in polygon for pi in poly.interiors]
    cv2.fillPoly(mask, exteriors, 1)
    cv2.fillPoly(mask, interiors, 0)

    return mask

  @staticmethod
  def pixel_mask_to_image(mask, r, g, b):
    return np.stack([
      np.multiply(mask, r),
      np.multiply(mask, g),
      np.multiply(mask, b)
      ]).transpose([1, 2, 0])

  @staticmethod
  def scale_multi_polygon(polygon, x_scale, y_scale):
    return shapely.affinity.scale(polygon, xfact=x_scale, yfact=y_scale, origin=(0, 0, 0))

