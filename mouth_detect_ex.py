from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
from sleep_step_calcaulation import sleep_step_calc

#입 크기 비율 계산 함수
def mouth_aspect_ratio(mouth):
    A=dist.euclidean(mouth[14],mouth[18])
    B=dist.euclidean(mouth[12],mouth[16])

    ear=A/B

    return ear



#입 크기 비율 임계치 결정 
MOUTH_AR_THRESH = 0.4
#입이 열려있는 동안의 최소 프레임 결정
MOUTH_AR_CONSEC_FRAMES = 60

#입이 열려있는 동안 프레임 저장
COUNTER = 0
#하품 횟수
TOTAL = 0

#dlib에서 얼굴을 식별하기 위한 함수 호출
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
#68랜드마크를 읽기 위한 dat 파일 경로 설정
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#입 좌표 값 설정
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#비디오 쓰레드 시작, 웹 캠으로부터 영상을 얻음
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

#비디오 쓰레드가 동작하는 동안 루프
while True:
    #출력 영상을 frame에 저장
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    #출력 영상 별개로 회색조 영상 저장
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #얼굴을 회색조 영상에서 얻음
    rects = detector(gray, 0)

    #얼굴이 식별되는 동안 루프
    for rect in rects:   
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
       
        #입 좌표를 얻은 뒤 크기 계산
        mouth=shape[Start:End]
        ear=mouth_aspect_ratio(mouth)

        #입을 표시
        MouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [MouthHull], -1, (0, 255, 0), 1)

        #입이 임계치보다 큰 동안 프레임 수 측정
        if ear > MOUTH_AR_THRESH:
            COUNTER += 1

        #입이 임계치보다 작은 조건 하에 수행
        else:
            #입이 열려있던 동안의 프레임을 측정하여 하품 수 계산
            if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                TOTAL += 1

            #프레임 수 초기화
            COUNTER = 0

        #텍스트 출력
        cv2.putText(frame, "Yawns: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Counter: {}".format(COUNTER), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #하품 감지
        sleep_step_calc.yawn_detection(sleep_step_calc,TOTAL)
        #졸음 단계에 따른 서비스 수행
        sleep_step_calc.event_sleep_step(sleep_step_calc)

    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #q 입력시 종료
    if key == ord("q"):
        break

#마무리
cv2.destroyAllWindows()
vs.stop()
