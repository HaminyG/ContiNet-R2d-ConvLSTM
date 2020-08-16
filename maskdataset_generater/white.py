import json
import cv2
import numpy as np

video_path = 'C:\\Users\\han006\\Desktop\\sample\\video\\000000\\P01_01_00_0_color.avi'
json_path = 'C:\\Users\\han006\\Desktop\\sample\\json\\000000\\P01_01_00_0_color.json'


class keypoint:
    def __init__(self, json_path, video_path):
        cap = cv2.VideoCapture(video_path)
        # self.hand_POSE_PAIRS = [(0, 1), (0, 5)]
        # self.body_POSE_PAIRS = [(0, 1), (1, 2)]
        self.hand_POSE_PAIRS = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (7, 8),#19,20
                                (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
        self.body_POSE_PAIRS = [(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]
        self.bodykeypoint_color=[(0,0,255),(32,0,255),(64,0,255),(96,0,255),(128,0,255),( 160,0,255),( 192,0,255),( 224,0,255),( 255,0,255),( 0,32,255),
                                 ( 0,64,255),( 0,96,255),( 0,128,255),( 0,160,255),( 0,192,255),( 0,224,255),( 0,255,255)]
        self.leftkeypint_color=[(12,255,0), (24,255,0), (36,255,0), (48,255,0), (60,255,0), (72,255,0), (84,255,0), (96,255,0), (108,255,0), (120,255,0), (132,255,0),
                                (144,255,0), (156,255,0), (168,255,0), (180,255,0), (196,255,0), (208,255,0), (220,255,0), (232,255,0), (248,255,0), (0,255,0)]

        self.rightkeypoint_color=[(255,0,12), (255,0,24), (255,0,36), (255,0,48), (255,0,60), (255,0,72), (255,0,84), (255,0,96), (255,0,108), (255,0,120), (255,0,132),
                                  (255,0,144), (255,0,156), (255,0,168), (255,0,180), (255,0,196), (255,0,208), (255,0,220), (255,0,232), (255,0,248), (255,0,0)]
        self.path = json_path
        self.video_path = video_path
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.mask = np.zeros((self.framecount, self.height, self.width, 3), dtype='uint8')
        cap = cv2.VideoCapture(self.video_path)
        signvideoframe = []
        while cap.isOpened():
            ret, videoframe = cap.read()
            if ret == True:
                signvideoframe.append(videoframe)
            else:
                break
        self.signvideoframe = np.array(signvideoframe)

    def get_x_y(self, keypoint):
        video_frame_coordinate = []
        with open(self.path, 'r') as f:
            data = json.load(f)  # all frames data in one video
            for num_frame in range(1, len(data), 1):
                # frame number in one video
                frame_coordinate = []
                frame_data = data[num_frame]  # keypoint for each frame
                if keypoint == 'Face keypoint':
                    data_body = frame_data[
                        keypoint]  # which kind of keypoint, in this case the keypoint is Face keypoint
                    data_body_list = data_body[0]  # keypoint in one list. [[[],[]]], after is {[],[]}.
                    for data_x_y_num in range(48, 68, 1):  # one frame x and y
                        data_keypoint = data_body_list[data_x_y_num]
                        data_x_y = tuple((data_keypoint[0], data_keypoint[1], data_keypoint[2]))
                        frame_coordinate.append(data_x_y)
                    video_frame_coordinate.append(frame_coordinate)

                else:
                    data_body = frame_data[keypoint]
                    data_body_list = data_body[0]
                    for data_x_y_num in range(len(data_body_list)):  # one frame x and y
                        data_keypoint = data_body_list[data_x_y_num]
                        data_x_y = tuple((data_keypoint[0], data_keypoint[1], data_keypoint[2]))
                        frame_coordinate.append(data_x_y)
                    video_frame_coordinate.append(frame_coordinate)

        return video_frame_coordinate

    def left_hand_keypoint(self):
        return self.get_x_y('Left hand keypoint')

    def right_hand_keypoint(self):
        return self.get_x_y('right hand keypoint')

    def body_keypoint(self):
        return self.get_x_y('body keypoint')

    def Face_keypoint(self):
        return self.get_x_y('Face keypoint')

    def video_frame(self):
        # self.mask=cv2.merge((self.mask,self.mask,self.mask))
        left=self.left_hand_keypoint()
        right=self.right_hand_keypoint()
        body=self.body_keypoint()
        for frame in range(1, self.framecount, 1):
            for left_hand_keypoint in (
            left[frame - 1]):  # frame-1 means the first frame of input video is black, so we have to delete that frame
                # left_hand_keypoint in this case, it is single point coordinate[x,y]
                cv2.circle(self.signvideoframe[frame, :, :, :],
                           (round(left_hand_keypoint[0]), round(left_hand_keypoint[1])),
                           3, (255, 255, 255), -1)
                # cv2.circle(self.mask[frame-1,:,:],(int(left_hand_keypoint[0]),int(left_hand_keypoint[1])),2,(255,255,255),-1)
            for right_hand_keypoint in right[frame - 1]:
                cv2.circle(self.signvideoframe[frame, :, :, :],
                           (round(right_hand_keypoint[0]), round(right_hand_keypoint[1])), 3, (255, 255, 255), -1)
            for body_keypoint in body[frame - 1]:
                cv2.circle(self.signvideoframe[frame, :, :, :], (round(body_keypoint[0]), round(body_keypoint[1])), 3,
                           (255, 255, 255), -1)
            # for Face_keypoint in self.Face_keypoint()[frame - 1]:
            #     if Face_keypoint[2] > 0.6:
            #         cv2.circle(self.mask[frame - 1, :, :, :], (round(Face_keypoint[0]), round(Face_keypoint[1])), 2,
            #                    (255, 255, 255), -1)

            for index, pair in enumerate(self.hand_POSE_PAIRS):

                partA = pair[0]
                partB = pair[1]

                if int(100*left[frame - 1][partA][2]) and int(100*left[frame - 1][partB][2])>10:
                    cv2.line(self.signvideoframe[frame, :, :,:], (int(left[frame - 1][partA][0]),
                                                          int(left[frame - 1][partA][1])),
                             (int(left[frame - 1][partB][0]),
                              int(left[frame - 1][partB][1])), (255,255,255), 2,
                             lineType=cv2.LINE_AA)
                else:
                    continue

            for index, pair in enumerate(self.hand_POSE_PAIRS):

                partA = pair[0]
                partB = pair[1]

                if int(100*right[frame - 1][partA][2]) and int(100*right[frame - 1][partB][2])>10:
                    cv2.line(self.signvideoframe[frame, :, :,:], (int(right[frame - 1][partA][0]),
                                                          int(right[frame - 1][partA][1])),
                             (int(right[frame - 1][partB][0]),
                              int(right[frame - 1][partB][1])), (255,255,255), 2,
                             lineType=cv2.LINE_AA)
                else:
                    continue

            for index, pair in enumerate(self.body_POSE_PAIRS):

                partA = pair[0]
                partB = pair[1]

                if int(100*body[frame - 1][partA][2]) and int(100*body[frame - 1][partB][2])>0:
                    cv2.line(self.signvideoframe[frame, :, :,:], (int(body[frame - 1][partA][0]),
                                                          int(body[frame - 1][partA][1])),
                             (int(body[frame - 1][partB][0]),
                              int(body[frame - 1][partB][1])), (255,255,255), 2,
                             lineType=cv2.LINE_AA)
                else:
                    continue
        return self.signvideoframe