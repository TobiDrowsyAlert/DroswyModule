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
        #landmark=list(map(int,landmark))

        self.face=np.array(landmark).reshape(68,2)

    def set_face_rect(self,rect):
        #얼굴 주변 사각형 좌표 입력 함수
        #rect=list(map(int,rect))
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

    def return_sleep_data(self):
        data=dict()
        data["r_ear"],data["l_ear"]=self.get_eye_aspect_ratio(self)
        data["m_ear"]=self.get_mouth_aspect_ratio(self)
        data["pitch"],data["roll"],data["yaw"]=self.get_pose_angle_aspect(self)

        return data


class sleep_data_calc:
    #status code 정의
    C_BLINK=100
    C_BLIND=101
    C_YAWN=200
    C_DRIVER_AWAY=300
    C_DIRVER_AWARE_FAIL=301
    C_NOMAL=400

    #일반 변수
    __sleep_step=0
    __sleep_weight=0
    __last_yawn=0
    __last_sleep_weight=0
    __last_sleep_step=0
    __sleep_service_flag=True

    #졸음 징후 변수
    l_ear=0
    r_ear=0
    m_ear=0
    yaw=0
    pitch=0
    roll=0
    blink=0
    yawn=0

    #졸음 판단 데이터
    sleep_code=0
    ear_THRESH=0.21
    m_ear_THRESH=0.4
    blink_THRESHOLD=21
    blind_FRAME=180
    blink_FRAME=3
    yawn_FRAME=60
    E_counter=0
    M_counter=0
    driver_counter=0

    def set_data(self,data):
        self.l_ear=data["l_ear"]
        self.r_dar=data["r_ear"]
        self.m_ear=data["m_ear"]
        self.pitch=data["pitch"]
        self.roll=data["roll"]
        self.yaw=data["yaw"]

    def set_persnal_data(self,data):
        self.ear_TRESH=data["e_ear"]
        self.m_ear_TRESH=data["m_ear"]
        self.blink_TRESHOLD=data["blink"]

    def event_sleep_step(self):
        if self.__sleep_weight==1:
            if not self.__sleep_service_flag:
                # 졸음 1단계 서비스
                self.__sleep_service_flag=True
                return self.get_sleep_data(self)
        elif self.__sleep_weight==2:
            if not self.__sleep_service_flag:
                # 졸음 2단계 서비스
                self.__sleep_service_flag = True
                return self.get_sleep_data(self)
        elif self.__sleep_weight==3:
            if not self.__sleep_service_flag:
                # 졸음 3단계 서비스
                self.__sleep_service_flag = True
                return self.get_sleep_data(self)
        #else:
            #예외 처리

    def raise_sleep_weight(self):
        #졸음 가중치 상승
        self.__last_sleep_weight=self.__sleep_weight
        self.__sleep_weight=self.__sleep_weight+1
        self.raise_sleep_step_by_weight(self)

    def raise_sleep_step(self):
        #졸음 단계 상승
        if self.__sleep_weight<5:
            __sleep_service_flag = False
            self.__last_sleep_weight = self.__sleep_weight
            self.__sleep_weight=5
            self.__last_sleep_step=self.__sleep_step
            self.__sleep_step+1
        elif self.__sleep_weight>7 and self.__sleep_weight<5:
            __sleep_service_flag = False
            self.__last_sleep_weight = self.__sleep_weight
            self.__sleep_weight=7
            self.__last_sleep_step = self.__sleep_step
            self.__sleep_step+1
        elif self.__sleep_weight>7 and self.__sleep_weight<7:
            __sleep_service_flag = False
            self.__last_sleep_weight = self.__sleep_weight
            self.__sleep_weight=10
            self.__last_sleep_step = self.__sleep_step
            self.__sleep_step+1

    def raise_sleep_step_by_weight(self):
        if self.__sleep_weight==5 and self.__sleep_step<1:
            self.raise_sleep_step(self)

        elif self.__sleep_weight==7 and self.__sleep_step==1:
            self.raise_sleep_step(self)

        elif self.__sleep_weight==10 and self.__sleep_step==2:
            self.raise_sleep_step(self)

    def cancle_sleep_weight(self):
        #졸음 가중치 상승 취소
        self.__sleep_weight=self.__last_sleep_weight

    def calc_sleep_data(self):
        print(self.E_counter)
        print(self.M_counter)
        #프레임별 졸음 징후 계산
        # 양 눈이 임계치 보다 작은 동안의 프레임 수를 측정
        if self.l_ear < self.ear_THRESH and self.r_ear < self.ear_THRESH:
            self.E_counter += 1
            #눈 감음
            if self.E_counter>=180:
                return self.blind_detection(self)

        # 양 눈이 임계치보다 큰 조건에 수행
        else:
            # 눈이 감겨있던 동안의 프레임을 검사하여 눈 깜빡임 계산
            if self.E_counter>= self.blink_FRAME:
                self.blink += 1
            # 프레임 수 초기화
            self.E_counter = 0

        # 입이 임계치보다 큰 동안 프레임 수 측정
        if self.m_ear > self.m_ear_THRESH:
            self.M_counter += 1

        # 입이 임계치보다 작은 조건 하에 수행
        else:
            # 입이 열려있던 동안의 프레임을 측정하여 하품 수 계산
            if self.M_counter >= self.yawn_FRAME:
                self.yawn += 1

            # 프레임 수 초기화
            self.M_counter = 0

        if self.blink_detection(self):
            if not self.__sleep_service_flag:
                self.sleep_code=self.C_BLINK

        if self.yawn_detection(self):
            if not self.__sleep_service_flag:
                self.sleep_code=self.C_YAWN

        if not self.__sleep_service_flag:
            return self.event_sleep_step(self)
        else:
            self.sleep_code=self.C_NOMAL
            return self.get_sleep_data(self)

    def blink_detection(self):
        if self.blink > self.blink_THRESHOLD:
            self.raise_sleep_weight(self)
            return True
        else:
            return False

    def yawn_detection(self):
        if self.yawn > 1:
            if self.__last_yawn!=self.yawn:
                self.raise_sleep_weight(self)
                self.__last_yawn=self.yawn
                return True
            else:
                return False
        else:
            return False

    def blind_detection(self):
        self.sleep_code=self.C_BLIND
        self.raise_sleep_step(self)

        return self.event_sleep_step(self)

    def angle_detection(self):
        if abs(self.pitch)>25 or abs(self.roll)>30:
            self.sleep_code=self.C_DIRVER_AWARE_FAIL
            self.raise_sleep_step(self)
            return self.event_sleep_step(self)

    def no_driver(self):
        self.driver_counter+1
        self.angle_detection(self)

        if self.driver_counter>=300:
            self.sleep_code=self.C_DRIVER_AWAY
            self.raise_sleep_weight(self)
            return self.event_sleep_step(self)
        else:
            self.sleep_code=self.C_NOMAL
            return self.get_sleep_data(self)

    def get_sleep_data(self):
        sleep_data=dict()
        sleep_data["sleep_step"]=self.__sleep_step
        sleep_data["status_code"]=self.sleep_code
        sleep_data["blink"]=self.blink
        sleep_data["yawn"]=self.yawn
        sleep_data["pitch"]=self.pitch
        sleep_data["yaw"]=self.yaw
        sleep_data["roll"]=self.roll

        return sleep_data




