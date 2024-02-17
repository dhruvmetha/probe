import numpy as np
from shapely.geometry import Polygon
from math import cos, sin
from functools import partial

def create_polygon(vertices):
    return Polygon([tuple(coord) for coord in rectangle_vertices(vertices)])

def rectangle_vertices(raw_vals):
    cx,cy,angle,w,h = raw_vals
    # angle in radians
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return np.array([
        [cx - dxcos + dysin, cy - dxsin - dycos],
        [cx + dxcos + dysin, cy + dxsin - dycos],
        [cx + dxcos - dysin, cy + dxsin + dycos],
        [cx - dxcos - dysin, cy - dxsin + dycos],
    ])

def rect_coords_all(bboxes_row):
    return np.array([bboxes_row[:5]])

def get_bbox_coords(bboxes):
    return np.apply_along_axis(rect_coords_all, axis=2, arr=bboxes)

def get_bbox_intersections(gt_bboxes, pred_bboxes):
    gt_rects = get_bbox_coords(gt_bboxes)
    pred_rects = get_bbox_coords(pred_bboxes)
    B = gt_rects.shape[0]
    N = gt_rects.shape[1]
    R = gt_rects.shape[2]
    S = gt_rects.shape[3]
    
    gt_polygons = list(np.apply_along_axis(create_polygon,1,gt_rects.reshape((B*N*R,S))))
    pred_polygons = list(np.apply_along_axis(create_polygon,1,pred_rects.reshape((B*N*R,S))))
    intersection_results = list(map(
        partial(Polygon.intersection),
        gt_polygons, pred_polygons
    ))

    intersection_areas = np.array(list(map(lambda polygon_int : polygon_int.area, intersection_results)))
    gt_areas = np.array(list(map(lambda polygon_int : polygon_int.area, gt_polygons)))
    pred_areas = np.array(list(map(lambda polygon_int : polygon_int.area, pred_polygons)))
    union_areas = gt_areas + pred_areas - intersection_areas
    intersection_areas = intersection_areas.reshape((B,N,R))
    union_areas = union_areas.reshape((B,N,R))
    return intersection_areas, union_areas