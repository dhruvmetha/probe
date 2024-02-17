import numpy as np
import cv2
import os
from glob import glob

video_id = 15
sim_video = cv2.VideoCapture(f"/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-22_00-22-04/final_animations_126/{video_id}.mp4")

sim_frames_dir = f"/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/sim_frames/{video_id}"

result = cv2.VideoWriter(os.path.join('/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/demo_vids_sim',f'demo_{video_id}.avi'),cv2.VideoWriter_fourcc(*'XVID'),15.0,(2560,720))
# frame_num = 68

fps_equalizer_flag = False
frame_prev = None
frame = None

iteration = 0
sim_frame_path = sorted(glob(sim_frames_dir + '/*.png'))
while True:
    if iteration == len(sim_frame_path):
        break
    ret, frame = sim_video.read()
    if not ret:
        break
    sim_frame = cv2.imread(sim_frame_path[iteration])
    merged_frame = cv2.hconcat([sim_frame, frame])
    result.write(merged_frame)
    iteration += 1

    # print(iteration)
    # if not ret:
    #     break


    # if sim_frame is None:
    #     continue
    
    # frame_new = np.ones((720,1280,3),dtype=np.uint8)*255
    # frame_new[160:560,40:1240,:] = frame
    # sim_frame[:10,:,:] = 255
    # sim_frame[700:,:,:] = 255

    
    # frame_num += 1

sim_video.release()
result.release()