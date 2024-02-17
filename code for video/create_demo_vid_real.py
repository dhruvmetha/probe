import numpy as np
import cv2
import os

video_id = 6
real_video = cv2.VideoCapture(f"/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/real data videos/00{video_id}/traj.avi")
sim_video = cv2.VideoCapture(f"/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-22_00-22-04/real_robot_videos/sep15/126/video_{video_id}.mp4")

result = cv2.VideoWriter(os.path.join('/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/demo_vids',f'demo_{video_id}.avi'),cv2.VideoWriter_fourcc(*'XVID'), 30.0,(3840,1080))
frame_num = 723
sim_wait = 50
# 9: 200, 3: 72, 8: 300. 4: 723. 5: 650
iteration = 0
ret_sim = None
while True:
    if sim_wait > iteration:
        ret_real, real_frame = real_video.read()
        iteration += 1
        continue   
    
    # if iteration % 3 == 0:
    if iteration % 3 == 0:
        ret_real, real_frame = real_video.read()
        ret_real, real_frame = real_video.read()
    
    if iteration % 3 == 0 or iteration % 5 == 0 or iteration % 6 == 0 or  iteration % 7 == 0:
        ret_sim, sim_frame = sim_video.read()
    if ret_sim is None:
        iteration += 1
        continue

    
    if not ret_real or not ret_sim:
        break


    # print(real_frame.shape, sim_frame.shape)
    # sim_frame_path = os.path.join(sim_frames_dir,f'{str(frame_num).zfill(5)}.jpg')

    # sim_frame = cv2.imread(sim_frame_path)

    if sim_frame is None:
        continue

    # frame_new = np.ones((1080,1920,3),dtype=np.uint8)*255
    # frame_new[340:740,360:1560,:] = frame
    merged_frame = cv2.hconcat([real_frame, sim_frame])
    result.write(merged_frame)
    iteration += 1
    
    # if ((iteration+1) % 4) == 0:
    #     frame_num += 2
    # else:
    #     frame_num += 1
    
    iteration += 1

sim_video.release()
real_video.release()
result.release()