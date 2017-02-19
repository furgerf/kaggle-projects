import numpy as np
import shapely.affinity
import cv2
from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon

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

  @staticmethod
  def evaluate_prediction(prediction, truth):
    return average_precision_score(prediction, truth)

  @staticmethod
  def prediction_to_binary_prediction(prediction, threshold=0.3):
    return prediction >= threshold

  @staticmethod
  def prediction_mask_to_polygons(mask, epsilon=10., min_area=10.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
      if parent_idx != -1:
        child_contours.add(idx)
        cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
      if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
        assert cnt.shape[1] == 1
        poly = Polygon(
            shell=cnt[:, 0, :],
            holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
              if cv2.contourArea(c) >= min_area])
        all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
      all_polygons = all_polygons.buffer(0)
      # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
      # need to keep it a Multi throughout
      if all_polygons.type == 'Polygon':
        all_polygons = MultiPolygon([all_polygons])
    return all_polygons

