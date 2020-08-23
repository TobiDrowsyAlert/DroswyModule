from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import math
import cv2
import queue
import random

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

    def __init__(self):
        #인스턴스 변수 선언
        # 일반 변수
        self.__sleep_step = 0
        self.__sleep_weight = 0
        self.__last_blink = 0
        self.__last_yawn = 0
        self.__last_r_ear=0
        self.__last_l_ear=0
        self.__last_m_ear=0
        self.__sum_r_ear=0
        self.__sum_l_ear=0
        self.__sum_m_ear=0
        self.__frequency_symptom=0
        self.__last_sleep_weight_queue = queue.LifoQueue()
        self.__last_sleep_step_queue = queue.LifoQueue()
        self.__sensitive_yawn = False
        self.__sensitive_blink=False
        self.__nod_delay_flag=False
        self.__blink_delay_flag=False
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
        self.__avg_ear=0
        self.__m_ear = 0
        self.__yaw = 0
        self.__pitch = 0
        self.__roll = 0
        self.__blink = 0
        self.__yawn = 0
        self.__nod=0

        # 졸음 판단 데이터
        self.__status_code = 0
        self.__ear_THRESHOLD = 0.2
        self.__m_ear_THRESHOLD = 0.4
        self.__blink_THRESHOLD = 21
        self.__nod_angle_THRESHOLD=-15
        self.__nod_THRESHOLD=3
        self.__no_driver_exception_THRESHOLD=30
        self.__blind_FRAME = 50
        self.__blink_FRAME = 1
        self.__yawn_FRAME = 30
        self.__driver_away_FRAME = 50

        # 프레임 카운터
        self.__blink_counter = 0
        self.__blind_counter=0
        self.__M_counter = 0
        self.__driver_counter = 0
        self.__nod_delay_counter=0
        self.__nod_delay=10
        self.__blink_delay_counter=0
        self.__blink_delay=20

        #누적 ear 큐
        self.__avg_ear_queue=queue.Queue(maxsize=5)

        #스트레칭 기능 변수
        #self.__stretch_order=["pitch","roll","yaw"]
        #self.__stretch_order_length=len(self.__stretch_order)
        #다양한 각도에 대해 스트레칭을 구현했으나 실제 사용은 "roll" 각도에 대해서만 사용하기로 결정

        #스트레칭 타입
        self.__stretch_type=["angle","mouth"]
        self.__selected_stretch_type=""
        self.__stretch_order_number=0

        self.__stretch_start_flag=False
        self.__stretch_end_flag=False
        self.__positive_angle_flag=False
        self.__negative_angle_flag=False

        self.__stretch_progress_flag=False

        self.__stretch_pitch_threshold=20
        self.__stretch_roll_threshold=20
        self.__stretch_yaw_threshold=15

        self.__stretch_positive_frame=0
        self.__stretch_negative_frame=0
        self.__stretch_angle_threshold=10

        self.__stretch_mouth_threshold=0.4
        self.__stretch_mouth_frame_threshold=30
        self.__stretch_mouth_frame=0

        self.__stretch_delay_flag=False
        self.__stretch_delay_frame=0
        self.__stretch_delay_threshold=50

        #스트레칭이 종료되지 않고 오래 진행되면 강제 종료를 제어하는 변수
        self.__stretch_shut_down_frame_threshold=1000
        self.__stretch_shut_down_frame=0

        print("sleep step calc 객체 생성")

    def set_data(self,data):
        self.__l_ear=data["l_ear"]
        self.__r_ear=data["r_ear"]
        self.__m_ear=data["m_ear"]
        self.__pitch=data["pitch"]
        self.__roll=data["roll"]
        self.__yaw=data["yaw"]
        self.__avg_ear=(self.__l_ear+self.__r_ear)/2

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

    def reset_stretch_flag(self,finish=None):
        self.__stretch_end_flag = False
        self.__positive_angle_flag = False
        self.__negative_angle_flag = False

        if finish==True:
            self.__stretch_start_flag=False
            self.__stretch_order_number=0
            self.__stretch_delay_flag=False
            self.__stretch_progress_flag=False


    def sensitive_yawn(self):
        self.__sensitive_yawn=True

    def sensitive_blink(self):
        self.__sensitive_blink=True

    def sensitive_blind(self):
        self.__blind_FRAME=30

    def sensitive_no_driver(self):
        self.__driver_away_FRAME=40
        self.__nod_THRESHOLD=2

    def sensitive_sleep_symptom(self):
        self.sensitive_yawn()
        self.sensitive_no_driver()
        self.sensitive_blind()
        self.sensitive_blink()


    def set_persnal_data(self,data):
        self.__ear_THRESHOLD=data["ear"]
        self.__m_ear_THRESHOLD=data["mar"]

        if data["isWeakTime"]==True:
            self.sensitive_sleep_symptom()

        self.__frequency_symptom=data["frequencyReason"]

        if self.__frequency_symptom== "100":
            self.sensitive_blink()
        elif self.__frequency_symptom=="101":
            self.sensitive_blind()
        elif self.__frequency_symptom=="200":
            self.sensitive_yawn()
        elif self.__frequency_symptom=="300":
            self.sensitive_no_driver()


    def set_default_data(self):
        self.__nod_THRESHOLD = 3
        self.__blind_FRAME = 50
        self.__yawn_FRAME = 30
        self.__driver_away_FRAME = 50

    def non_weak_time(self):
        self.__nod_THRESHOLD = 3
        self.__blind_FRAME = 50
        self.__yawn_FRAME = 30
        self.__driver_away_FRAME = 50

        if self.__frequency_symptom == "100":
            self.sensitive_blink()
        elif self.__frequency_symptom == "101":
            self.sensitive_blind()
        elif self.__frequency_symptom == "200":
            self.sensitive_yawn()
        elif self.__frequency_symptom == "300":
            self.sensitive_no_driver()


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

        elif self.__sleep_weight>=10 and self.__sleep_step==3:
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
        self.__driver_counter=0

        #스트레칭 수행 중이었으면 강제 종료
        if self.__stretch_progress_flag==True:
            self.stretch_shutdown()

        #꾸벅거림 딜레이
        if self.__nod_delay_flag:
            self.__nod_delay_counter+=1
            if self.__nod_delay_counter==self.__nod_delay:
                self.__nod_delay_flag=False
                self.__nod_delay_counter=0

        print("sleep_step:{}".format(self.__sleep_step))
        print("sleep_weight:{}".format(self.__sleep_weight))

        #프레임별 졸음 징후 계산

        if self.__avg_ear_queue.full():
            self.__avg_ear_queue.get()
            self.__avg_ear_queue.put(self.__avg_ear)

            ear_list=list(self.__avg_ear_queue.queue)
            ear_list.sort()
            midian_ear=ear_list[2]
            print("midian_ear: {}".format(midian_ear))
            if midian_ear>self.__ear_THRESHOLD:
                self.__blind_counter=0
                if self.blink_detection():
                    return self.get_sleep_data(self.__C_BLINK)
            else:
                if self.blind_detection():
                    return self.get_sleep_data(self.__C_BLIND)
        else:
            self.__avg_ear_queue.put(self.__avg_ear)

        if self.yawn_detection():
            return self.get_sleep_data(self.__C_YAWN)

        return self.get_sleep_data(self.__C_NOMAL)

    def blink_detection(self):
        print("blink_counter:{}".format(self.__blink_counter))
        print("blink_delay:{}".format(self.__blink_delay_counter))
        #두 눈 ear 값의 평균이 임계치보다 작으면
        if self.__avg_ear<self.__ear_THRESHOLD:
            if not self.__blink_delay_flag:
                self.__blink_counter+=1

                #눈 감았던 동안의 ear 값의 합 저장
                self.__sum_l_ear+=self.__l_ear
                self.__sum_r_ear+=self.__r_ear
        else:
            if self.__blink_counter>=self.__blink_FRAME:
                self.__blink+=1

                #눈 감았던 동안 최근 ear 값 저장
                self.__last_l_ear=self.__sum_l_ear/self.__blink_counter
                self.__last_r_ear=self.__sum_r_ear/self.__blink_counter

                #ear 값의 합 초기화
                self.__sum_r_ear=0
                self.__sum_l_ear=0

                self.__blink_delay_counter = 0
                self.__blink_delay_flag = True

            self.__blink_counter = 0

            if self.__blink_delay_flag:
                # 눈 깜빡임 인식에 지연 부여
                self.__blink_delay_counter += 1
                if self.__blink_delay_counter == self.__blink_delay:
                    self.__blink_delay_flag = False
                    self.__blink_delay_counter = 0


        if self.__blink>0 and (self.__blink % self.__blink_THRESHOLD)==0:
            if self.__last_blink!=self.__blink:
                self.__last_blink=self.__blink
                self.raise_sleep_weight()
                #취약시간이면 가중치 한 번 더 상승
                if self.__sensitive_blink:
                    self.raise_sleep_weight()
                #self.raise_sleep_step()

                if self.__sleep_service_flag == False:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def reset_blink(self):
        self.__blink=0
        self.__last_blink=0

    def yawn_detection(self):
        self.__last_m_ear=self.__m_ear
        print("M_counter:{}".format(self.__M_counter))
        # 입이 임계치보다 큰 동안 프레임 수 측정
        if self.__m_ear > self.__m_ear_THRESHOLD:
            self.__M_counter += 1

            #입이 열렸던 동안 ear 값의 합 저장
            self.__sum_m_ear+=self.__m_ear

        # 입이 임계치보다 작은 조건 하에 수행
        else:
            # 입이 열려있던 동안의 프레임을 측정하여 하품 수 계산
            if self.__M_counter >= self.__yawn_FRAME:
                self.__yawn += 1

                #최근 입이 열려있던 동안의 평균 ear 값 저장
                self.__last_m_ear=self.__sum_m_ear/self.__M_counter

                #ear 값 초기화
                self.__sum_m_ear=0
            # 프레임 수 초기화
            self.__M_counter = 0

        if self.__yawn > 1:
            if self.__last_yawn!=self.__yawn:
                self.__last_yawn=self.__yawn
                self.raise_sleep_weight()
                #취약시간이면 가중치 한번 더 상승
                if self.__sensitive_yawn:
                    self.raise_sleep_weight()
                #self.raise_sleep_step()

                if self.__sleep_service_flag == False:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def reset_yawn(self):
        self.__yawn=0
        self.__last_yawn=0

    def blind_detection(self):
        print("blind_counter:{}".format(self.__blind_counter))
        self.__blind_counter+=1

        # 눈 감았던 동안의 ear 값의 합 저장
        self.__sum_l_ear += self.__l_ear
        self.__sum_r_ear += self.__r_ear

        if self.__blind_counter>=self.__blind_FRAME:
            if self.__raise_sleep_step_flag == False:

                # 눈 감았던 동안 최근 ear 값 저장
                self.__last_l_ear = self.__sum_l_ear / self.__blind_counter
                self.__last_r_ear = self.__sum_r_ear / self.__blind_counter

                # ear 값의 합 초기화
                self.__sum_r_ear = 0
                self.__sum_l_ear = 0

                self.raise_sleep_step()
                self.__raise_sleep_step_flag = True
                return True
            else:
                return False
        else:
            self.__raise_sleep_step_flag=False
            return False

    def no_driver(self):
        self.__nod_delay_counter=0
        #운전자 이탈 중 예외 : 우회전, 좌회전 등으로 옆 상황을 보는 경우
        if (abs(self.__pitch) <= self.__no_driver_exception_THRESHOLD and abs(self.__roll)) and abs(self.__yaw) >= self.__no_driver_exception_THRESHOLD:
            print("Exception of No driver")

            if self.__stretch_progress_flag==True:
                return self.get_stretch_data()

            return self.get_sleep_data(self.__C_NOMAL)

        #운전자가 운전석에서 인식되지 않는 경우
        else:
            # 운전자가 꾸벅거리는 경우
            if self.__pitch < self.__nod_angle_THRESHOLD:
                print("nod : {}".format(self.__nod))
                if not self.__nod_delay_flag:
                    self.__nod += 1
                    self.__nod_delay_flag = True
                    if self.__nod > 0 and (self.__nod % self.__nod_THRESHOLD) == 0:
                        if self.__raise_sleep_step_flag == False:
                            self.raise_sleep_step()
                            self.__raise_sleep_step_flag = True
                            return self.get_sleep_data(self.__C_DRIVER_AWAY)

            self.__driver_counter += 1
            print("sleep_step:{}".format(self.__sleep_step))
            print("sleep_weight:{}".format(self.__sleep_weight))
            print("driver_counter:{}".format(self.__driver_counter))

            if self.__driver_counter >= self.__driver_away_FRAME:
                print("service_flag:{}".format(self.__sleep_service_flag))
                if self.__raise_sleep_step_flag == False:
                    self.raise_sleep_step()
                    self.__raise_sleep_step_flag = True

                    if self.__stretch_progress_flag==True:
                        self.stretch_shutdown()

                    return self.get_sleep_data(self.__C_DRIVER_AWAY)
                else:
                    return self.get_sleep_data(self.__C_NOMAL)

            else:
                self.__raise_sleep_step_flag = False

                #스트레칭 중에 운전자가 잠시 인식이 안된 경우
                if self.__stretch_progress_flag==True:
                    return self.get_stretch_data()

                return self.get_sleep_data(self.__C_NOMAL)

    def reset_nod(self):
        self.__nod=0

    def stretch_angle(self,angle_type):
        if self.__stretch_progress_flag==True:
            if angle_type=="pitch":
                if abs(self.__pitch)>=self.__stretch_pitch_threshold and abs(self.__roll)<=self.__stretch_roll_threshold and abs(self.__yaw)<= self.__stretch_yaw_threshold:
                    if self.__pitch>0:
                        self.__stretch_negative_frame=0
                        self.__stretch_positive_frame+=1

                        if self.__stretch_positive_frame>=self.__stretch_angle_threshold:
                            self.__positive_angle_flag=True
                            self.__stretch_positive_frame=0
                    else:
                        self.__stretch_positive_frame=0
                        self.__stretch_negative_frame+=1

                        if self.__stretch_negative_frame>=self.__stretch_angle_threshold:
                            self.__negative_angle_flag=True
                            self.__stretch_negative_frame=0

                elif abs(self.__pitch)<self.__stretch_pitch_threshold:
                    self.__stretch_negative_frame=0
                    self.__stretch_positive_frame=0

            elif angle_type=="roll":
                if abs(self.__roll)>=self.__stretch_roll_threshold and abs(self.__pitch)<=self.__stretch_pitch_threshold and abs(self.__yaw)<=self.__stretch_yaw_threshold:
                    if self.__roll > 0:
                        self.__stretch_negative_frame = 0
                        self.__stretch_positive_frame += 1

                        if self.__stretch_positive_frame >= self.__stretch_angle_threshold:
                            self.__positive_angle_flag = True
                            self.__stretch_positive_frame = 0
                    else:
                        self.__stretch_positive_frame = 0
                        self.__stretch_negative_frame += 1

                        if self.__stretch_negative_frame >= self.__stretch_angle_threshold:
                            self.__negative_angle_flag = True
                            self.__stretch_negative_frame = 0

                elif abs(self.__roll) < self.__stretch_roll_threshold:
                    self.__stretch_negative_frame = 0
                    self.__stretch_positive_frame = 0

            elif angle_type=="yaw":
                if abs(self.__yaw)>=self.__stretch_yaw_threshold and abs(self.__pitch)<=self.__stretch_pitch_threshold and abs(self.__roll)<=self.__stretch_roll_threshold:
                    if self.__yaw > 0:
                        self.__stretch_negative_frame = 0
                        self.__stretch_positive_frame += 1

                        if self.__stretch_positive_frame >= self.__stretch_angle_threshold:
                            self.__positive_angle_flag = True
                            self.__stretch_positive_frame = 0
                    else:
                        self.__stretch_positive_frame = 0
                        self.__stretch_negative_frame += 1

                        if self.__stretch_negative_frame >= self.__stretch_angle_threshold:
                            self.__negative_angle_flag = True
                            self.__stretch_negative_frame = 0

                elif abs(self.__yaw) < self.__stretch_yaw_threshold:
                    self.__stretch_negative_frame = 0
                    self.__stretch_positive_frame = 0

            if self.__positive_angle_flag == True and self.__negative_angle_flag == True:
                self.__stretch_positive_frame=0
                self.__stretch_negative_frame=0
                self.__stretch_end_flag=True
                return True

            else:
                return False

    def stretch_mouth(self):
        if self.__stretch_progress_flag==True:
            if self.__m_ear>=self.__stretch_mouth_threshold:
                self.__stretch_mouth_frame+=1

                if self.__stretch_mouth_frame>=self.__stretch_mouth_frame_threshold:

                    self.__stretch_end_flag=True
                    self.__stretch_mouth_frame=0
                    return True
            else:
                self.__stretch_mouth_frame=0

    def stretch_delay(self):
        if self.__stretch_delay_flag==True:
            self.__stretch_delay_frame+=1

            if self.__stretch_delay_frame>=self.__stretch_delay_threshold:
                self.__stretch_delay_flag=False
                self.__stretch_delay_frame=0
                return True
            return False

    def stretch_shutdown(self):
        if self.__stretch_shut_down_frame >= self.__stretch_shut_down_frame_threshold:
            self.__stretch_shut_down_frame=0

            self.__stretch_end_flag=True
            self.__stretch_order_number=3

            return self.get_stretch_data()

        else:
            self.__stretch_shut_down_frame=0

            self.reset_stretch_flag(True)

            return True

    def do_stretch(self):
        self.__driver_counter=0

        #스트레칭이 종료되지 않고 오래 유지될 시 강제 종료
        if self.__stretch_progress_flag==True:
            self.__stretch_shut_down_frame+=1

            if self.__stretch_shut_down_frame>=self.__stretch_shut_down_frame_threshold:
                return self.stretch_shutdown()

        #스트레칭 첫 시작 시
        if self.__stretch_progress_flag==False and self.__stretch_order_number==0:
            #스트레칭 순서 랜덤으로 초기화
            """
            self.__stretch_order=random.sample(self.__stretch_order,3)
            """

            #스트레칭 타입 랜덤 선택
            #self.__selected_stretch_type=random.choice(self.__stretch_type)

            #스트레칭 타입 각도로 고정
            self.__selected_stretch_type="angle"

            #스트레칭 타입 입으로 고정
            #self.__selected_stretch_type="mouth"

            #스트레칭 진행 중 flag
            self.__stretch_progress_flag=True

            #스트레칭 시작 시점 flag
            self.__stretch_start_flag=True


        if self.__stretch_progress_flag==True and self.__stretch_delay_flag==True:
            #스트레칭 루프 이후에 스트레칭에 딜레이를 줌
            self.stretch_delay()

        #angle_type=self.__stretch_order[self.__stretch_order_number]

        #딜레이 아닌 경우
        if self.__stretch_delay_flag==False:
            # 랜덤으로 선택된 스트레칭 타입에 따라 스트레칭 수행
            if self.__selected_stretch_type == "angle":
                # 얼굴 각도 스트레칭
                self.stretch_angle("roll")
            elif self.__selected_stretch_type == "mouth":
                # 입 스트레칭
                self.stretch_mouth()

        return self.get_stretch_data()

    def get_stretch_data(self):
        stretch_data=dict()
        stretch_type=self.__selected_stretch_type

        stretch_data["stretch_type"]=self.__selected_stretch_type
        #스트레칭 종료 시작 여부
        stretch_data["start"] = self.__stretch_start_flag
        stretch_data["end"] = self.__stretch_end_flag

        #시작 플래그는 스트레칭 루프 시작 순간만 True가 되도록 설정
        if self.__stretch_start_flag==True:
            self.__stretch_start_flag=False

        #스트레칭 딜레이 여부
        stretch_data["delay"] = self.__stretch_delay_flag

        if stretch_type=="angle":
            stretch_data["left"] = self.__positive_angle_flag
            stretch_data["right"] = self.__negative_angle_flag
            stretch_data["angle"] = self.__roll

        # 스트레칭 루프 한 단계 수행 종료 시
        if self.__stretch_end_flag:
            # 스트레칭 횟수 증가
            if self.__stretch_order_number<3:
                self.__stretch_order_number = self.__stretch_order_number + 1

            stretch_data["count"] = self.__stretch_order_number

            # 스트레칭 시작 알림
            self.__stretch_start_flag=True

            # 스트레칭 딜레이
            self.__stretch_delay_flag=True

            # 스트레칭 루프 종료 시
            if self.__stretch_order_number == 3:
                # 스트레칭 관련 플래그 및 횟수 초기화
                self.reset_stretch_flag(True)
            else:
                #스트레칭 루프가 한 단계 종료되고 완전 종료는 아닌 경우
                self.reset_stretch_flag()
        # 스트레칭 루프가 종료 되지않은 경우
        else:
            stretch_data["count"] = self.__stretch_order_number

        return stretch_data

    def get_sleep_data(self,code=None):
        sleep_data = dict()
        sleep_data["sleep_step"] = self.__sleep_step
        sleep_data["blink"] = self.__blink
        sleep_data["yawn"] = self.__yawn
        sleep_data["pitch"] = self.__pitch
        sleep_data["yaw"] = self.__yaw
        sleep_data["roll"] = self.__roll
        if code is None:
            sleep_data["status_code"] = self.__status_code
            sleep_data["left_ear"] = self.__l_ear
            sleep_data["right_ear"] = self.__r_ear
            sleep_data["m_ear"] = self.__m_ear

            return sleep_data

        else:
            self.__status_code = code
            sleep_data["status_code"] = code

            if code != self.__C_NOMAL:
                self.__sleep_service_flag = True

                if code == self.__C_BLIND or code == self.__C_BLINK:
                    sleep_data["left_ear"] = self.__last_l_ear
                    sleep_data["right_ear"] = self.__last_r_ear
                    sleep_data["m_ear"] = self.__m_ear

                elif code == self.__C_YAWN:
                    sleep_data["left_ear"] = self.__l_ear
                    sleep_data["right_ear"] = self.__r_ear
                    sleep_data["m_ear"] = self.__last_m_ear

                else:
                    sleep_data["left_ear"] = self.__l_ear
                    sleep_data["right_ear"] = self.__r_ear
                    sleep_data["m_ear"] = self.__m_ear

                return sleep_data

            else:
                sleep_data["left_ear"] = self.__l_ear
                sleep_data["right_ear"] = self.__r_ear
                sleep_data["m_ear"] = self.__m_ear

            return sleep_data

    def print_sleep_data(self):
        sleep_data=self.get_sleep_data()
        print(sleep_data)

