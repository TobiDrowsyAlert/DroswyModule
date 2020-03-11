from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import math
from sleep_step_calcaulation import sleep_step_calc

#눈 좌표의 비율을 계산하는 함수
def eye_aspect_ratio(eye):
    #눈의 종방향 좌표의 차를 계산
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    #눈의 횡방향 좌표의 차를 계산
    C = dist.euclidean(eye[0], eye[3])

    #ear 공식을 이용하여 계산
    ear = (A + B) / (2.0 * C)

    return ear

#입 크기 비율 계산 함수
def mouth_aspect_ratio(mouth):
    A=dist.euclidean(mouth[14],mouth[18])
    B=dist.euclidean(mouth[12],mouth[16])

    ear=A/B

    return ear

#얼굴 각도 계산 함수
def pose_aspect_angle(rect, landmarks):

    #2D points
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

    #3D model points
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),             # nose tip
            (0.0, -330.0, -65.0),        # chin
            (-165.0, 170.0, -135.0),     # left eye left corner
            (165.0, 170.0, -135.0),      # right eye right corner
            (-150.0, -150.0, -125.0),    # left mouth corner
            (150.0, -150.0, -125.0)      # right mouth corner
        ]
    )


    (x,y,w,h)=face_utils.rect_to_bb(rect)

    #print("in func :: x:%f,y:%f",x + w / 2, y + h / 2)
    center=(x+w/2,y+h/2)
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
    _, r_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    r_vector_matrix = cv2.Rodrigues(r_vec)[0]

    project_matrix = np.hstack((r_vector_matrix, trans_vec))
    euler_angles = cv2.decomposeProjectionMatrix(project_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return int(pitch), int(roll), int(yaw)

#dlib에서 얼굴 식별을 위한 함수 호출
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
#68 랜드마크 좌표를 얻기 위한 dat파일 경로 설정
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#눈의 ear 값 임계치를 결정
EYE_AR_THRESH = 0.21
#눈이 감겨있는 동안 최소 프레임
EYE_AR_CONSEC_FRAMES = 3
#눈 감은 동안의 프레임 수
E_COUNTER = 0
#눈 깜빡임 횟수
E_TOTAL = 0
blink=0

#입 크기 비율 임계치 결정
MOUTH_AR_THRESH = 0.4
#입이 열려있는 동안의 최소 프레임 결정
MOUTH_AR_CONSEC_FRAMES = 60

#입이 열려있는 동안 프레임 저장
M_COUNTER = 0
#하품 횟수
M_TOTAL = 0

#양 눈의 좌표를 얻음
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#입 좌표 값 설정
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#비디오 쓰레드 시작, 웹 캠에서 영상 얻음
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

#비디오 쓰레드가 동작하는 동안 루프
while True:
    #출력 영상을 frame에 저장
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    #별개로 회색조 영상을 얻음
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #회색조 영상에서 얼굴 식별
    rects = detector(gray, 0)

    #얼굴이 식별되는 동안 루프
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 두 눈의 좌표를 얻은 후 눈 크기 비율 계산
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 입 좌표를 얻은 뒤 크기 계산
        mouth = shape[mStart:mEnd]
        m_ear = mouth_aspect_ratio(mouth)

        #얼굴 각도 계산
        pitch,roll,yaw=pose_aspect_angle(rect,shape)

        # 양 눈이 임계치 보다 작은 동안의 프레임 수를 측정
        if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
            E_COUNTER += 1

        # 양 눈이 임계치보다 큰 조건에 수행
        else:
            # 눈이 감겨있던 동안의 프레임을 검사하여 눈 깜빡임 계산
            if E_COUNTER >= EYE_AR_CONSEC_FRAMES:
                E_TOTAL += 1
                blink+=1
            # 프레임 수 초기화
            E_COUNTER = 0

         # 입이 임계치보다 큰 동안 프레임 수 측정
        if m_ear > MOUTH_AR_THRESH:
            M_COUNTER += 1

        # 입이 임계치보다 작은 조건 하에 수행
        else:
            # 입이 열려있던 동안의 프레임을 측정하여 하품 수 계산
            if M_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                 M_TOTAL += 1

            # 프레임 수 초기화
            M_COUNTER = 0

        #텍스트 출력
        cv2.putText(frame, "PTICH: {}".format(pitch), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "ROLL: {}".format(roll), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAW: {}".format(yaw), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "BLINK: {}".format(E_TOTAL), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {}".format(M_TOTAL), (250, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 눈 깜빡임 감지
        if sleep_step_calc.blink_detection(sleep_step_calc, blink, 23):
            blink = 0

        # 하품 감지
        sleep_step_calc.yawn_detection(sleep_step_calc, M_TOTAL)
        # 졸음 단계에 따른 서비스 수행
        sleep_step_calc.event_sleep_step(sleep_step_calc)

    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #q 입력 시 종료
    if key == ord("q"):
        break

#마무리
cv2.destroyAllWindows()
vs.stop()