from tqdm import tqdm
from shapely.geometry import Polygon
from math import cos, sin
from functools import partial
import numpy as np
import os
import pickle
import multiprocessing as mp
from matplotlib import pyplot as plt

iou_input_dir_path = "/common/users/dm1487/legged_manipulation_data_store/evaluation_data2/"
iou_output_dir_path = "/common/users/dm1487/legged_manipulation_data_store/evaluation_data_iou2"
# iou_input_dir_path = "/common/home/dm1487/Downloads/sep14"
# iou_output_dir_paths = "/common/home/dm1487/Downloads/real_iou "

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
    return np.vstack([bboxes_row[:5],bboxes_row[5:10]])

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

def get_bbox_intersections_by_files(curr_box_files,dummy_arg=None):
    for curr_box_file in tqdm(curr_box_files):
        curr_boxes = None
        with open(os.path.join(iou_input_dir_path,curr_box_file), 'rb') as f:
            curr_boxes = pickle.load(f)
        # curr_boxes = np.load(os.path.join(iou_input_dir_path,curr_box_file))
        curr_boxes_gt = curr_boxes["gt"]
        curr_boxes_pred = curr_boxes["pred"]
        curr_boxes_mask = curr_boxes["mask"]
        
        intersection_areas_full, union_areas_full = get_bbox_intersections(curr_boxes_gt, curr_boxes_pred)
        ious_full = intersection_areas_full/union_areas_full

        inter_mov = intersection_areas_full[curr_boxes_gt[:,:,:1].nonzero()[:2]][:, 0]
        union_mov = union_areas_full[curr_boxes_gt[:,:,:1].nonzero()[:2]][:, 0]

        inter_fixed = intersection_areas_full[curr_boxes_gt[:,:,5:6].nonzero()[:2]][:, 1]
        union_fixed = union_areas_full[curr_boxes_gt[:,:,5:6].nonzero()[:2]][:, 1]
        # # print('fixed', np.mean(a/b))

        position_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,:2] - curr_boxes_pred[:,:,:2], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        abs_ang_diff = np.abs(curr_boxes_gt[:,:,2] - curr_boxes_pred[:,:,2])
        abs_ang_diff = np.where(abs_ang_diff > np.pi, 2*np.pi - abs_ang_diff, abs_ang_diff)
        angular_error = np.sum(abs_ang_diff * curr_boxes_mask[:, :, 0], axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        shape_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,3:5] - curr_boxes_pred[:,:,3:5], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        result = {}
        result['movable'] = {
                'intersection': inter_mov,
                'union': union_mov,
                'iou': inter_mov/union_mov,
                'position error': position_error,
                'angular error': angular_error,
                'shape error': shape_error
        }

        position_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,5:7] - curr_boxes_pred[:,:,5:7], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        abs_ang_diff = np.abs(curr_boxes_gt[:,:,7] - curr_boxes_pred[:,:,7])
        abs_ang_diff = np.where(abs_ang_diff > np.pi, 2*np.pi - abs_ang_diff, abs_ang_diff)
        angular_error = np.sum(abs_ang_diff * curr_boxes_mask[:, :, 0], axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        shape_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,8:10] - curr_boxes_pred[:,:,8:10], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        
        result['fixed'] = {
                'intersection': inter_fixed,
                'union': union_fixed,
                'iou': inter_fixed/union_fixed,
                'position error': position_error,
                'angular error': angular_error,
                'shape error': shape_error
        }

        
        with open(os.path.join(iou_output_dir_path,curr_box_file.split(".")[0] + ".pkl"), 'wb') as f:
            pickle.dump(result, f)


def get_bbox_intersections_func(curr_box):
    for curr_boxes in tqdm(curr_box):
        curr_boxes_gt = curr_boxes["gt"]
        curr_boxes_pred = curr_boxes["pred"]
        curr_boxes_mask = curr_boxes["mask"]
        
        intersection_areas_full, union_areas_full = get_bbox_intersections(curr_boxes_gt, curr_boxes_pred)
        ious_full = intersection_areas_full/union_areas_full

        inter_mov = intersection_areas_full[curr_boxes_gt[:,:,:1].nonzero()[:2]][:, 0]
        union_mov = union_areas_full[curr_boxes_gt[:,:,:1].nonzero()[:2]][:, 0]

        inter_fixed = intersection_areas_full[curr_boxes_gt[:,:,5:6].nonzero()[:2]][:, 1]
        union_fixed = union_areas_full[curr_boxes_gt[:,:,5:6].nonzero()[:2]][:, 1]
        # print('fixed', curr_boxes_gt[:,:,5:6].nonzero()[:2], curr_boxes_gt[:,:,5:6].nonzero()[:2])

        position_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,:2] - curr_boxes_pred[:,:,:2], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        abs_ang_diff = np.abs(curr_boxes_gt[:,:,2] - curr_boxes_pred[:,:,2])
        abs_ang_diff = np.where(abs_ang_diff > np.pi, 2*np.pi - abs_ang_diff, abs_ang_diff)
        angular_error = np.sum(abs_ang_diff * curr_boxes_mask[:, :, 0], axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        shape_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,3:5] - curr_boxes_pred[:,:,3:5], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        result = {}
        result['movable'] = {
                'intersection': inter_mov,
                'union': union_mov,
                'iou': inter_mov/union_mov,
                'position error': position_error,
                'angular error': angular_error,
                'shape error': shape_error
        }

        position_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,5:7] - curr_boxes_pred[:,:,5:7], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        abs_ang_diff = np.abs(curr_boxes_gt[:,:,7] - curr_boxes_pred[:,:,7])
        abs_ang_diff = np.where(abs_ang_diff > np.pi, 2*np.pi - abs_ang_diff, abs_ang_diff)
        angular_error = np.sum(abs_ang_diff * curr_boxes_mask[:, :, 0], axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        shape_error = np.sum((np.linalg.norm(curr_boxes_gt[:,:,8:10] - curr_boxes_pred[:,:,8:10], axis=-1) * curr_boxes_mask[:, :, 0]), axis=-1)/curr_boxes_mask[:, :, 0].sum(axis=-1)

        
        result['fixed'] = {
                'intersection': inter_fixed,
                'union': union_fixed,
                'iou': inter_fixed/union_fixed,
                'position error': position_error,
                'angular error': angular_error,
                'shape error': shape_error
        }

        return result

if __name__ == '__main__':

    num_workers= 30
    box_files = os.listdir(iou_input_dir_path)
    files_per_worker = max(1, int(len(box_files)/num_workers))
    if not os.path.exists(iou_output_dir_path): 
        os.makedirs(iou_output_dir_path)

    workers = []
    for i in range(num_workers):
        workers.append(mp.Process(target=get_bbox_intersections_by_files,args=(box_files[i*files_per_worker:(i+1)*files_per_worker],"dummy")))
    
    for worker in workers:
        worker.daemon = True
        worker.start()
    
    for worker in workers:
        worker.join()

    comb_files = os.listdir(iou_output_dir_path)


    final_result = {
        'movable': {
            'intersection': [],
            'union': [],
            'ious': [],
            'position error': [],
            'angular error': [],
            'shape error': []
        },
        'fixed': {
            'intersection': [],
            'union': [],
            'ious': [],
            'position error': [],
            'angular error': [],
            'shape error': []
        }
    }

    for i in comb_files:
        with open(os.path.join(iou_output_dir_path,i), 'rb') as f:
            data = pickle.load(f)

        final_result['movable']['intersection'].extend(data['movable']['intersection'])
        final_result['movable']['union'].extend(data['movable']['union'])
        final_result['movable']['ious'].extend(data['movable']['iou'])
        final_result['movable']['position error'].extend(data['movable']['position error'])
        final_result['movable']['angular error'].extend(data['movable']['angular error'])
        final_result['movable']['shape error'].extend(data['movable']['shape error'])

        final_result['fixed']['intersection'].extend(data['fixed']['intersection'])
        final_result['fixed']['union'].extend(data['fixed']['union'])
        final_result['fixed']['ious'].extend(data['fixed']['iou'])
        final_result['fixed']['position error'].extend(data['fixed']['position error'])
        final_result['fixed']['angular error'].extend(data['fixed']['angular error'])
        final_result['fixed']['shape error'].extend(data['fixed']['shape error'])


    print('movable', np.mean(final_result['movable']['ious']))
    print('fixed', np.mean(final_result['fixed']['ious']))

    print('movable', np.mean(final_result['movable']['position error']))
    print('fixed', np.mean(final_result['fixed']['position error']))

    print('movable', np.mean(final_result['movable']['angular error']))
    print('fixed', np.mean(final_result['fixed']['angular error']))

    print('movable', np.mean(final_result['movable']['shape error']))
    print('fixed', np.mean(final_result['fixed']['shape error']))


    # movable 0.39725017159816456
    # fixed 0.3141681600026176
    # movable 0.2097117057075257
    # fixed 0.26885102443053266
    # movable 0.08794615413629678
    # fixed 0.01599065901335186
    # movable 0.17128330461285043
    # fixed 0.1143723142882693
  

