import os
# from mask_for_keypoint import keypoint
import cv2
from class500.color3 import keypoint

all_one_video_file_path=[]
all_one_json_file_path=[]
video_name=[]

global filepath

for i in range(0, 10, 1):
    videopath = "D:\\DEVISIGN_D\\DEVISION\\DEVISIGN_D\\000{}".format(i)
    jsonpath= "D:\\DEVISIGN_D\\DEVISION\\DEVISION_D_json\\000{}".format(i)
    filepath='D:\\DEVISIGN_D\\DEVISION\\color3onimage\\000{}'.format(i)
    os.mkdir(filepath)
    all_one_video_file_path.clear()
    all_one_json_file_path.clear()
    video_name.clear()
    for video_file_name in os.listdir(videopath):# video file name example P01_01_00_0_color.avi
        one_video_file_path=os.path.join(videopath,video_file_name)#generate path
        video_name.append(video_file_name)#get video name
        all_one_video_file_path.append(one_video_file_path)#save all video path in one list
    for json_file_name in os.listdir(jsonpath):
        one_json_file_path=os.path.join(jsonpath,json_file_name)
        all_one_json_file_path.append(one_json_file_path)


    for index in range(len(all_one_video_file_path)):

        keypoints = keypoint(all_one_json_file_path[index], all_one_video_file_path[index])
        video_frame=keypoints.video_frame()#get video frame from keypoint class
        out = cv2.VideoWriter(os.path.join(filepath,video_name[index]),
                          cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))

        for frame in range(len(video_frame)):
            out.write(video_frame[frame])
        out.release()
        print('finish!!')



