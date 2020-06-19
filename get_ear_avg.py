from pymysql.cursors import Cursor
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from matplotlib import pyplot as plt
import numpy as np
import imutils
import time
import dlib
import cv2
import pymysql
import datetime
import queue

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
arrayBlink=[]
arrayBlind=[]
arrayRSub=[]
arrayLsub=[]
arrayAVG=[]


#총 프레임 수
COUNTER = 0
BK_COUNTER=0
BD_COUNTER=0
reset_BD=0
ear_THRESH=0.2
subtracted_ear_THRESH=0.08
blink_THRESH=1
blind_THRESH=30
blink=0
blind=False

#평균EAR값
Left_EAR=0
Right_EAR=0

last_leftear=0
last_rightear=0

#누적 ear 큐
ear_queue=queue.Queue(maxsize=5)

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

# 데이터베이스와 연결
#conn = pymysql.connect(host='localhost', user='root', password='root',charset='utf8')

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
        EAR=(leftEAR+rightEAR)/2
        arrayAVG.append(EAR)

        if ear_queue.full():
            ear_queue.get()
            ear_queue.put(EAR)

            ear_list=list(ear_queue.queue)
            ear_list.sort()
            ear_back=ear_list[2]
            print("frame:{}, ear_back:{}".format(COUNTER,ear_back))

            #눈 깜빡임
            if ear_back > ear_THRESH:
                blind = False
                if BD_COUNTER>0:
                    reset_BD+=1
                    if reset_BD==5:
                        BD_COUNTER=0
                        reset_BD=0

                arrayBlind.append(0)

                if EAR <= ear_THRESH:
                    BK_COUNTER += 1
                    arrayBlink.append(0)
                else:
                    if BK_COUNTER >= blink_THRESH:
                        blink += 1
                        arrayBlink.append(blink)

                    else:
                        arrayBlink.append(0)

                    BK_COUNTER = 0
            #눈 감음
            else:
                BD_COUNTER += 1
                arrayBlink.append(0)
                if BD_COUNTER >= blind_THRESH:
                    blind = True
                    arrayBlind.append(1)
                else:
                    arrayBlind.append(0)
        else:
            ear_queue.put(EAR)
            arrayBlink.append(0)
            arrayBlind.append(0)



        last_leftear=leftEAR
        last_rightear=rightEAR
        #그래프 입력
        arrayNum.append(COUNTER)
        arrayLEar.append(leftEAR)
        arrayREar.append(rightEAR)


        #양 눈을 표시
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #텍스트 표시
        #현재 ear 값 출력
        cv2.putText(frame, "LEFT_EAR: {:.2f}".format(leftEAR), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "RIGHT_EAR: {:.2f}".format(rightEAR), (350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    """
    #db에 기록하기 위한 dict
    sleep_data=dict()
    sleep_data['right_ear']=float(rightEAR)
    sleep_data['left_ear']=float(leftEAR)
    sleep_data['frame']=int(COUNTER)

    c_time = datetime.datetime.now()

    #db기록
    curs = conn.cursor()
    sql = 'call face_data_db.input_eye_data(%s,%s,%s,%s,%s)'
    curs.execute(sql, (c_time, None, sleep_data['right_ear'], sleep_data['left_ear'], sleep_data['frame']))
    conn.commit()
    """

    """
    L_EAR_AVG=Left_EAR/COUNTER
    R_EAR_AVG=Right_EAR/COUNTER
    #평균 ear 값 출력
    cv2.putText(frame, "L_EAR_AVG: {:.2f}".format(L_EAR_AVG), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "R_EAR_AVG: {:.2f}".format(R_EAR_AVG), (250, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    """

    #현재 프레임 출력
    cv2.putText(frame, "Counter: {}".format(COUNTER), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Blink: {}".format(blink), (380, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if blind:
        cv2.putText(frame, "Blind!!!", (250, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q 입력 시 종료
    if key == ord("q"):
        plt.subplot(4,1,1)
        plt.plot(arrayNum, arrayLEar,'ob',label="Left_ear")
        plt.plot(arrayNum,arrayREar,'or',label="Right_ear")
        plt.axhline(y=0.2,color='k')
        plt.xlabel('frame')
        plt.ylabel('ear')

        plt.subplot(4,1,2)
        plt.plot(arrayNum,arrayBlink,'ok',label="Blink")
        plt.xlabel('frame')
        plt.ylabel('blink')

        plt.subplot(4,1,3)
        plt.plot(arrayNum,arrayAVG,'or',label="EAR_AVG")
        plt.xlabel('frame')
        plt.ylabel('avg_ear')

        plt.subplot(4, 1, 4)
        plt.plot(arrayNum, arrayBlind, 'ok', label="Blind")
        plt.xlabel('frame')
        plt.ylabel('blind')

        """
        plt.subplot(3,1,3)
        plt.plot(arrayNum,arrayLsub,'ob',label='Left_sub_ear')
        plt.plot(arrayNum,arrayRSub,'or',label='right_sub_ear')
        plt.xlabel('frame')
        plt.ylabel('sub_ear')
        """

        plt.show()
        break

#마무리
#conn.close()
cv2.destroyAllWindows()
vs.stop()
print("종료")