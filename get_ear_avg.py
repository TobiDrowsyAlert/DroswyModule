from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from matplotlib import pyplot as plt
import numpy as np
import imutils
import time
import dlib
import cv2

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


#그래프 배열
arrayNum = []
arrayREar = []
arrayLEar= []

#총 프레임 수
COUNTER = 0

#평균EAR값
Left_EAR=0
Right_EAR=0

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

        Left_EAR+=leftEAR
        Right_EAR+=rightEAR

        COUNTER=COUNTER+1


        #그래프 출력
        arrayNum.append(COUNTER)
        arrayLEar.append(leftEAR)
        arrayREar.append(rightEAR)

        if COUNTER>0 and (COUNTER % 300)==0:
            plt.plot(arrayNum, arrayLEar,arrayREar)
            plt.show()


        #양 눈을 표시
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #텍스트 표시
        #현재 ear 값 출력
        cv2.putText(frame, "LEFT_EAR: {:.2f}".format(leftEAR), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "RIGHT_EAR: {:.2f}".format(rightEAR), (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    L_EAR_AVG=Left_EAR/COUNTER
    R_EAR_AVG=Right_EAR/COUNTER
    #평균 ear 값 출력
    cv2.putText(frame, "L_EAR_AVG: {:.2f}".format(L_EAR_AVG), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "R_EAR_AVG: {:.2f}".format(R_EAR_AVG), (250, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    """
    #현재 프레임 출력
    cv2.putText(frame, "Counter: {}".format(COUNTER), (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    """

    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q 입력 시 종료
    if key == ord("q"):
        break

#눈 감았을 때 EAR 값, 임계치를 구하기 위함
T_Rigth_EAR=0
T_Left_EAR=0
T_COUNTER=0

T_arrayNum=[]
T_arrayREar=[]
T_arrayLEar=[]

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

        if leftEAR<0.2 and rightEAR<0.2 :
            T_Left_EAR+=leftEAR
            T_Rigth_EAR+=rightEAR
            T_COUNTER=T_COUNTER+1
            
            # 그래프 출력
            T_arrayNum.append(T_COUNTER)
            T_arrayLEar.append(leftEAR)
            T_arrayREar.append(rightEAR)

        if T_COUNTER>0 and (T_COUNTER % 180)==0:
            plt.plot(T_arrayNum, T_arrayLEar,T_arrayREar)
            plt.show()


        #양 눈을 표시
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #텍스트 표시
        #현재 ear 값 출력
        cv2.putText(frame, "LEFT_EAR: {:.2f}".format(leftEAR), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "RIGHT_EAR: {:.2f}".format(rightEAR), (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if T_COUNTER>0:
        T_L_EAR_AVG=T_Left_EAR/T_COUNTER
        T_R_EAR_AVG=T_Rigth_EAR/T_COUNTER
        #평균 ear 값 출력
        cv2.putText(frame, "L_EAR_AVG: {:.2f}".format(T_L_EAR_AVG), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "R_EAR_AVG: {:.2f}".format(T_R_EAR_AVG), (250, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    """
    #현재 프레임 출력
    cv2.putText(frame, "Counter: {}".format(T_COUNTER), (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    """

    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q 입력 시 종료
    if key == ord("q"):
        break

#마무리
cv2.destroyAllWindows()
vs.stop()