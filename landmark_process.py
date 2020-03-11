from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import numpy as np
import math
import cv2
class landmark_process:
    face=0
    face_rect=0

    def set_face_landmarks(self,landmark):
        #얼굴 좌표 입력 함수
        self.face=landmark

    def set_face_rect(self,rect):
        #얼굴 주변 사각형 좌표 입력 함수
        self.face_rect=rect

    def get_eye_landmarks(self):
        #눈 좌표 추출 함수
        (rstart,rend)=face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        (lstart,lend)=face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        reye=self.face[rstart:rend]
        leye=self.face[lstart:lend]

        return reye,leye

    def get_mouth_landmarks(self):
        #입 좌표 추출 함수
        (start,end)=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        mouth=self.face[start:end]

        return mouth

    def get_eye_aspect_ratio(self):
        #양 눈의 종횡비율을 반환하는 함수
        reye,leye=self.get_eye_landmarks(self)

        rear=self.eye_aspect_ratio(reye)
        lear=self.eye_aspect_ratio(leye)

        return rear,lear

    def get_mouth_aspect_ratio(self):
        #입의 종횡비를 반환하는 함수
        mouth=self.get_mouth_landmarks(self)

        ear=self.mouth_aspect_ratio(mouth)

        return ear

    def eye_aspect_ratio(eye):
        # 눈의 종방향 좌표의 차를 계산
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # 눈의 횡방향 좌표의 차를 계산
        C = dist.euclidean(eye[0], eye[3])

        # ear 공식을 이용하여 계산
        ear = (A + B) / (2.0 * C)

        return ear

    def mouth_aspect_ratio(mouth):
        #입의 종횡비를 계산
        A = dist.euclidean(mouth[14], mouth[18])
        B = dist.euclidean(mouth[12], mouth[16])

        ear = A / B

        return ear

    def get_pose_angle_aspect(self):
        landmarks=self.face
        rect=self.face_rect
        # 2D points
        image_points = np.array(
            [
                (landmarks[30][0], landmarks[30][1]),  # nose tip
                (landmarks[8][0], landmarks[8][1]),  # chin
                (landmarks[36][0], landmarks[36][1]),  # left eye left corner
                (landmarks[45][0], landmarks[45][1]),  # right eye right corner
                (landmarks[48][0], landmarks[48][1]),  # left mouth corner
                (landmarks[54][0], landmarks[54][1])  # right mouth corner
            ],
            dtype="double",
        )

        # 3D model points
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # nose tip
                (0.0, -330.0, -65.0),  # chin
                (-165.0, 170.0, -135.0),  # left eye left corner
                (165.0, 170.0, -135.0),  # right eye right corner
                (-150.0, -150.0, -125.0),  # left mouth corner
                (150.0, -150.0, -125.0)  # right mouth corner
            ]
        )

        (x, y, w, h) = rect

        # print("in func :: x:%f,y:%f",x + w / 2, y + h / 2)
        center = (x + w / 2, y + h / 2)
        focal_length = center[0] / np.tan(60 / (2 * np.pi / 180))
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ],
            dtype="double",
        )

        dist_coeffs = np.zeros((4, 1))
        _, r_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)

        r_vector_matrix = cv2.Rodrigues(r_vec)[0]

        project_matrix = np.hstack((r_vector_matrix, trans_vec))
        euler_angles = cv2.decomposeProjectionMatrix(project_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return int(pitch), int(roll), int(yaw)
