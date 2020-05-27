from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import math
import cv2
import queue

class landmark_process:
    def __init__(self):
        __face = 0
        __face_rect = 0
        print("landmark process 객체 생성")

    def set_face_landmarks(self,landmark):
        #얼굴 좌표 입력 함수
        #landmark=list(map(int,landmark))

        self.__face=np.array(landmark).reshape(68,2)

    def set_face_rect(self,rect):
        #얼굴 주변 사각형 좌표 입력 함수
        #rect=list(map(int,rect))
        self.__face_rect=rect

    def get_eye_landmarks(self):
        #눈 좌표 추출 함수
        (rstart,rend)=face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        (lstart,lend)=face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        reye=self.__face[rstart:rend]
        leye=self.__face[lstart:lend]

        return reye,leye

    def get_mouth_landmarks(self):
        #입 좌표 추출 함수
        (start,end)=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        mouth=self.__face[start:end]

        return mouth

    def get_eye_aspect_ratio(self):
        #양 눈의 종횡비율을 반환하는 함수
        reye,leye=self.get_eye_landmarks()

        rear=self.eye_aspect_ratio(reye)
        lear=self.eye_aspect_ratio(leye)

        return rear,lear

    def get_mouth_aspect_ratio(self):
        #입의 종횡비를 반환하는 함수
        mouth=self.get_mouth_landmarks()

        ear=self.mouth_aspect_ratio(mouth)

        return ear

    def eye_aspect_ratio(self,eye):
        # 눈의 종방향 좌표의 차를 계산
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # 눈의 횡방향 좌표의 차를 계산
        C = dist.euclidean(eye[0], eye[3])

        # ear 공식을 이용하여 계산
        ear = (A + B) / (2.0 * C)

        return ear

    def mouth_aspect_ratio(self,mouth):
        #입의 종횡비를 계산
        A = dist.euclidean(mouth[14], mouth[18])
        B = dist.euclidean(mouth[12], mouth[16])

        ear = A / B

        return ear

    def get_pose_angle_aspect(self):
        landmarks=self.__face
        rect=self.__face_rect
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
        data["r_ear"],data["l_ear"]=self.get_eye_aspect_ratio()
        data["m_ear"]=self.get_mouth_aspect_ratio()
        data["pitch"],data["roll"],data["yaw"]=self.get_pose_angle_aspect()

        return data


class sleep_data_calc:

    #클래스 변수 선언
    #status code 정의
    __C_BLINK=100
    __C_BLIND=101
    __C_YAWN=200
    __C_DRIVER_AWAY=300
    __C_DIRVER_AWARE_FAIL=301
    __C_NOMAL=400

    def __init__(self,data=None):

        #인스턴스 변수 선언
        # 일반 변수
        self.__sleep_step = 0
        self.__sleep_weight = 0
        self.__last_blink = 0
        self.__last_yawn = 0
        self.__last_sleep_weight_queue = queue.LifoQueue()
        self.__last_sleep_step_queue = queue.LifoQueue()
        self.__sleep_service_flag = True
        """
        sleep_service_flag는 초기 상태는 ture
        졸음 단계가 상승하여 경고 서비스 수행 전이면 false, 경고 서비스 수행 이후에는 true
        졸음 단계가 상승한 후 계속 경고를 보내지 않게 하기 위한 flag
        """
        self.__raise_sleep_step_flag = False
        """
        raise_sleep_step_flag는 초기 상태는 false
        졸음 단계 상승 했으면 true, 상승하지 않았으면 false
        운전자 인식 안됨, 눈 오래 감음과 같이 단발성이 아닌 유지되는 징후인 경우 단계를 한 번만 상승하기 위한 flag
        """
        # 졸음 징후 변수
        self.__l_ear = 0
        self.__r_ear = 0
        self.__m_ear = 0
        self.__yaw = 0
        self.__pitch = 0
        self.__roll = 0
        self.__blink = 0
        self.__yawn = 0

        # 졸음 판단 데이터
        self.__status_code = 0
        self.__ear_THRESH = 0.21
        self.__m_ear_THRESH = 0.4
        self.__blink_THRESHOLD = 21
        self.__blind_FRAME = 75
        self.__blink_FRAME = 4
        self.__yawn_FRAME = 50
        self.__driver_away_FRAME = 75

        # 프레임 카운터
        self.__E_counter = 0
        self.__M_counter = 0
        self.__driver_counter = 0

        #개인화 데이터 삽입할 때,
        if data is not None:
            self.__ear_THRESH = data["e_ear"]
            self.__m_ear_THRESH = data["m_ear"]
            self.__blink_THRESHOLD = data["blink"]

            print("sleep step calc 객체 생성")


        else: #개인화 데이터 삽입이 없을 때
            print("sleep step calc 객체 생성")

    def set_data(self,data):
        self.__l_ear=data["l_ear"]
        self.__r_ear=data["r_ear"]
        self.__m_ear=data["m_ear"]
        self.__pitch=data["pitch"]
        self.__roll=data["roll"]
        self.__yaw=data["yaw"]

    def reset_data(self):
        #데이터 리셋 메소드

        #졸음 단계 및 가중치 초기화
        self.__sleep_step=0
        self.__last_sleep_step_queue.queue.clear()
        self.__sleep_weight=0
        self.__last_sleep_weight_queue.queue.clear()

        #졸음 징후 변수들 초기화
        self.__l_ear=0
        self.__r_ear=0
        self.__m_ear=0

        self.__yawn=0
        self.__blink=0

        self.__last_blink=0
        self.__last_yawn=0

        self.__yaw=0
        self.__pitch=0
        self.__roll=0

        #플래그 값 초기화
        self.__sleep_service_flag=True
        self.__raise_sleep_step_flag=False

        #프레임 카운터 초기화
        self.__E_counter=0
        self.__M_counter=0
        self.__driver_counter=0

    def set_persnal_data(self,data):
        self.__ear_TRESH=data["e_ear"]
        self.__m_ear_TRESH=data["m_ear"]
        self.__blink_TRESHOLD=data["blink"]

    def raise_sleep_weight(self):
        #졸음 가중치 상승
        self.__last_sleep_weight_queue.put(self.__sleep_weight)
        self.__sleep_weight=self.__sleep_weight+1
        self.raise_sleep_step_by_weight()

    def raise_sleep_step(self):
        if self.__sleep_step<3:
            self.__sleep_service_flag=False
            self.__last_sleep_step_queue.put(self.__sleep_step)
            self.__sleep_step+=1
            if self.__sleep_step == 1:
                self.__last_sleep_weight_queue.put(self.__sleep_weight)
                self.__sleep_weight = 4

            elif self.__sleep_step == 2:
                self.__last_sleep_weight_queue.put(self.__sleep_weight)
                self.__sleep_weight = 7

            elif self.__sleep_step == 3:
                self.__last_sleep_weight_queue.put(self.__sleep_weight)
                self.__sleep_weight = 10
        else:
            self.__sleep_service_flag=False

    def raise_sleep_step_by_weight(self):
        if self.__sleep_weight==4 and self.__sleep_step==0:
            self.raise_sleep_step()

        elif self.__sleep_weight==7 and self.__sleep_step==1:
            self.raise_sleep_step()

        elif self.__sleep_weight==10 and self.__sleep_step==2:
            self.raise_sleep_step()

        elif self.__sleep_service_flag>=10 and self.__sleep_step==3:
            if self.__sleep_service_flag:
                self.raise_sleep_step()

    def cancle_sleep_step(self):
        #졸음 가중치 상승 취소
        #이전 졸음 단계 기록이 없는 경우
        if self.__last_sleep_step_queue.empty():
            self.__sleep_step=0
            #졸음 단계 0단계

            #이전 졸음 가중치 기록이 없는 경우
            if self.__last_sleep_weight_queue.empty():
                self.__sleep_weight=0
                #졸음 가중치 0
            #이전 졸음 가중치 기록이 있는 경우
            else:
                self.__sleep_weight=self.__last_sleep_weight_queue.get()
                #졸음 가중치 롤백
        #이전 졸음 단계 기록이 있는 경우
        else:
            self.__sleep_step=self.__last_sleep_step_queue.get()
            #졸음 단계 롤백
            if self.__last_sleep_weight_queue.empty():
                self.__sleep_weight=0
            else:
                self.__sleep_weight=self.__last_sleep_weight_queue.get()

    def drop_sleep_step(self):
        #졸음 단계를 낮추는 메소드
        if self.__sleep_step>0:
            if self.__sleep_step==1:
                self.__sleep_step=0
                self.__sleep_weight=0

                self.__last_sleep_step_queue.queue.clear()
                self.__last_sleep_weight_queue.queue.clear()

            elif self.__sleep_step==2:
                self.__sleep_step=1
                self.__sleep_weight=4

                self.__last_sleep_step_queue.get()
                self.__last_sleep_weight_queue.get()

            elif self.__sleep_step==3:
                self.__sleep_step=2
                self.__sleep_weight=7

                self.__last_sleep_step_queue.get()
                self.__last_sleep_weight_queue.get()

        else:
            self.__sleep_step = 0
            self.__sleep_weight = 0

    def calc_sleep_data(self):
        self.driver_counter=0
        print("sleep_step:{}".format(self.__sleep_step))
        print("sleep_weight:{}".format(self.__sleep_weight))
        print("E_counter:{}".format(self.__E_counter))
        print("M_counter:{}".format(self.__M_counter))
        #프레임별 졸음 징후 계산
        # 양 눈이 임계치 보다 작은 동안의 프레임 수를 측정
        if self.__l_ear < self.__ear_THRESH and self.__r_ear < self.__ear_THRESH:
            self.__E_counter += 1
            #눈 감음
            if self.__E_counter>=self.__blind_FRAME:
                return self.__blind_detection()
            else:
                self.__raise_sleep_step_flag=False

        # 양 눈이 임계치보다 큰 조건에 수행
        else:
            # 눈이 감겨있던 동안의 프레임을 검사하여 눈 깜빡임 계산
            if self.__E_counter>= self.__blink_FRAME:
                self.__blink += 1
            # 프레임 수 초기화
            self.__E_counter = 0

        # 입이 임계치보다 큰 동안 프레임 수 측정
        if self.__m_ear > self.__m_ear_THRESH:
            self.__M_counter += 1

        # 입이 임계치보다 작은 조건 하에 수행
        else:
            # 입이 열려있던 동안의 프레임을 측정하여 하품 수 계산
            if self.__M_counter >= self.__yawn_FRAME:
                self.__yawn += 1

            # 프레임 수 초기화
            self.__M_counter = 0

        if self.blink_detection():
            print("service_flag:{}".format(self.__sleep_service_flag))
            if self.__sleep_service_flag==False:
                return self.get_sleep_data(self.__C_BLINK)
            else:
                return self.get_sleep_data(self.__C_NOMAL)

        elif self.yawn_detection():
            if self.__sleep_service_flag==False:
                return self.get_sleep_data(self.__C_YAWN)
            else:
                return self.get_sleep_data(self.__C_NOMAL)
        else:
            return self.get_sleep_data(self.__C_NOMAL)

    def blink_detection(self):
        if self.__blink>1 and (self.__blink % self.__blink_THRESHOLD)==0:
            if self.__last_blink!=self.__blink:
                self.__last_blink=self.__blink
                #self.raise_sleep_weight(self)
                self.raise_sleep_step()
                return True
            else:
                return False
        else:
            return False

    def reset_blink(self):
        self.__blink=0
        self.__last_blink=0

    def yawn_detection(self):
        if self.__yawn > 1:
            if self.__last_yawn!=self.__yawn:
                self.__last_yawn=self.__yawn
                #self.raise_sleep_weight()
                self.raise_sleep_step()
                return True
            else:
                return False
        else:
            return False

    def reset_yawn(self):
        self.__yawn=0
        self.__last_yawn=0

    def blind_detection(self):
        if self.__raise_sleep_step_flag==False:
            self.raise_sleep_step()
            self.__raise_sleep_step_flag=True
            return self.get_sleep_data(self.__C_BLIND)
        else:
            return self.get_sleep_data(self.__C_NOMAL)


    def no_driver(self):
        self.__driver_counter+=1
        print("sleep_step:{}".format(self.__sleep_step))
        print("sleep_weight:{}".format(self.__sleep_weight))
        print("driver_counter:{}".format(self.__driver_counter))

        if self.driver_counter>=self.__driver_away_FRAME:
            print("service_flag:{}".format(self.__sleep_service_flag))
            if self.__raise_sleep_step_flag==False:
                # 운전자가 감지되었다가 정면을 제대로 주시하지 않는 경우
                if abs(self.__pitch) > 25 or abs(self.__roll) > 30:
                    self.raise_sleep_step()
                    self.__raise_sleep_step_flag=True
                    return self.get_sleep_data(self.__C_DIRVER_AWARE_FAIL)
                # 운전자가 아예 운전석에서 감지되지 않는 경우
                else:
                    self.raise_sleep_step()
                    self.__raise_sleep_step_flag=True
                    return self.get_sleep_data(self.__C_DRIVER_AWAY)
            else:
                return self.get_sleep_data(self.__C_NOMAL)

        else:
            self.__raise_sleep_step_flag=False
            return self.get_sleep_data(self.__C_NOMAL)

    def get_sleep_data(self,code=None):
        if code is None:
            sleep_data = dict()
            sleep_data["sleep_step"] = self.__sleep_step
            sleep_data["status_code"] = self.__status_code
            sleep_data["blink"] = self.__blink
            sleep_data["yawn"] = self.__yawn
            sleep_data["pitch"] = self.__pitch
            sleep_data["yaw"] = self.__yaw
            sleep_data["roll"] = self.__roll
            sleep_data["left_ear"] = self.__l_ear
            sleep_data["right_ear"] = self.__r_ear
            sleep_data["m_ear"] = self.__m_ear

            return sleep_data

        else:
            self.__status_code = code
            sleep_data = dict()
            sleep_data["sleep_step"] = self.__sleep_step
            sleep_data["status_code"] = code
            sleep_data["blink"] = self.__blink
            sleep_data["yawn"] = self.__yawn
            sleep_data["pitch"] = self.__pitch
            sleep_data["yaw"] = self.__yaw
            sleep_data["roll"] = self.__roll
            sleep_data["left_ear"] = self.__l_ear
            sleep_data["right_ear"] = self.__r_ear
            sleep_data["m_ear"] = self.__m_ear

            if code != self.__C_NOMAL:
                self.__sleep_service_flag = True

            return sleep_data

    def print_sleep_data(self):
        sleep_data=self.get_sleep_data(self.__status_code)
        print(sleep_data)

