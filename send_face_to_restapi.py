from imutils.video import VideoStream
from imutils import face_utils
#import numpy as np
import imutils
import dlib
import cv2
#import json
import requests


#dlib에서 얼굴 식별을 위한 함수 호출
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
#68 랜드마크 좌표를 얻기 위한 dat파일 경로 설정
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


#비디오 쓰레드 시작, 웹 캠에서 영상 얻음
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

#얼굴 좌표 저장 변수 선언
face_landmarks=[]
face_rect=[]

#확인용 프레임 수 변수 선언
frames=0
#json 저장 딕셔너리
face_location=dict()
driver=False
sleep_correct=True

#로그인 데이터 전송
userid="user"
userdata=dict()
userdata["userId"]=userid
requests.post('http://127.0.0.1:5000/login', json=userdata)

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

        driver=True
        frames=frames+1

        #얼굴을 표시하는 직사각형
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        #얼굴 좌표 변수 저장
        face_landmarks = shape.tolist()
        face_rect=list((x,y,w,h))

        #얼굴을 표시하는 직사각형 출력
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #직사각형 가운데 좌표 점으로 출력
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (255, 0, 0), -1)

        cv2.putText(frame, "Frame: {}".format(frames), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #print(face_rect)
    #print(face_landmarks)


    #dict 저장
    face_location["userId"]=userid
    face_location["landmarks"] = face_landmarks
    face_location["rect"] = face_rect
    face_location["frame"] = frames
    face_location["driver"]=driver

    #print(face_location)
    #rest API로 얼굴 좌표 전송
    #res=requests.post('http://15.165.116.82:1234/set_face',json=face_location)
    #res = requests.post('http://127.0.0.1:5000/set_face', json=face_location)
    res=requests.post('http://127.0.0.1:5000/stretch',json=face_location)
 #   print(type(res.json()))
    #sleep_data=res.json()
    #print(sleep_data)
    stretch_data=res.json()
    print(stretch_data)
    angle_type=stretch_data["angle_type"]
    if stretch_data["start"]:
        if not stretch_data["end"]:
            if angle_type == "pitch":
                if stretch_data["positive"] == False and stretch_data["negative"] == False:
                    cv2.putText(frame, "stretch {} angle!!".format(angle_type), (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif stretch_data["positive"]==True and stretch_data["negative"]==False:
                    cv2.putText(frame, "stretch oppsite direction!!", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Please stretch downward direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif stretch_data["negative"]==True and stretch_data["positive"]==False:
                    cv2.putText(frame, "stretch oppsite direction!!", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Please stretch upward direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif angle_type == "roll" or angle_type=="yaw":
                if stretch_data["positive"] == False and stretch_data["negative"] == False:
                    cv2.putText(frame, "stretch {} angle!!".format(angle_type), (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif stretch_data["positive"]==True and stretch_data["negative"]==False:
                    cv2.putText(frame, "stretch oppsite direction!!", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Please stretch right direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif stretch_data["negative"]==True and stretch_data["positive"]==False:
                    cv2.putText(frame, "stretch oppsite direction!!", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Please stretch left direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if stretch_data["count"]==3:
                break

    #print(face_location)


    #인식 중이 아니면 초기화
    driver=False

    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #q 입력 시 종료
    if key == ord("q"):
        break

requests.post('http://127.0.0.1:5000/logout', json=userdata)
#마무리
cv2.destroyAllWindows()
vs.stop()