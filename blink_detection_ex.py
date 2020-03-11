from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
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


#눈의 ear 값 임계치를 결정
EYE_AR_THRESH = 0.21
#눈이 감겨있는 동안 최소 프레임
EYE_AR_CONSEC_FRAMES = 3
#눈이 감겨있는 동안 최대 프레임
EYE_AR_OVER_FRAMES= 10


#눈 감은 동안의 프레임 수
COUNTER = 0
#눈 깜빡임 횟수
TOTAL = 0
blink=0
#최근 눈 감은 동안의 프레임 수
LATEST_COUNTER=0

#dlib에서 얼굴을 식별하기 위한 함수 호출
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
#68랜드마크를 찾는 dat 파일 경로 설정
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#양 눈의 좌표를 얻음
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#비디오 스트림 쓰레드 시작, 웹 캠으로 부터 영상을 얻음
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

#비디오 스트림이 동작하는 동안 루프
while True:
    #출력 영상을 frame에 저장
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    #출력 영상과 별개로 회색조 영상 저장
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #얼굴을 회색조 영상에서 얻음
    rects = detector(gray, 0)

    #얼굴이 식별되는 동안 루프
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #두 눈의 좌표를 얻은 후 눈 크기 비율 계산
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #양 눈의 평균 값을 사용하지 않고 두 눈에서 각각 측정되는 값을 사용
        ear = (leftEAR + rightEAR) / 2.0

        #양 눈을 표시
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #양 눈이 임계치 보다 작은 동안의 프레임 수를 측정
        if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
            COUNTER += 1

        #양 눈이 임계치보다 큰 조건에 수행
        else:
            #눈이 감겨있던 동안의 프레임을 검사하여 눈 깜빡임 계산
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                LATEST_COUNTER = COUNTER
                TOTAL += 1
                blink+=1
            #프레임 수 초기화
            COUNTER = 0

        #텍스트 표시
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Counter: {}".format(LATEST_COUNTER),(150,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
        cv2.putText(frame, "LEFT_EAR: {:.2f}".format(leftEAR), (10,320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "RIGHT_EAR: {:.2f}".format(rightEAR), (250, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #눈 깜빡임 감지
        if sleep_step_calc.blink_detection(sleep_step_calc,blink,23):
            blink=0

        #졸음 단계에 따른 서비스 수행
        sleep_step_calc.event_sleep_step(sleep_step_calc)

    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q 입력 시 종료
    if key == ord("q"):
        break

#마무리
cv2.destroyAllWindows()
vs.stop()