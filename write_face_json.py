from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import json
import requests

# dlib에서 얼굴 식별을 위한 함수 호출
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# 68 랜드마크 좌표를 얻기 위한 dat파일 경로 설정
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# 비디오 쓰레드 시작, 웹 캠에서 영상 얻음
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

# 얼굴 좌표 저장 변수 선언
face_landmarks = []
face_rect = []

# 확인용 프레임 수 변수 선언
frames = 0
# json 저장 딕셔너리
face_location = dict()

# 비디오 쓰레드가 동작하는 동안 루프
while True:
    # 출력 영상을 frame에 저장
    frame = vs.read()
    frame = imutils.resize(frame, width=450)

    # 별개로 회색조 영상을 얻음
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 회색조 영상에서 얼굴 식별
    rects = detector(gray, 0)

    # 얼굴이 식별되는 동안 루프
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        frames = frames + 1

        # 얼굴을 표시하는 직사각형
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # 얼굴 좌표 변수 저장
        face_landmarks = shape.tolist()
        face_rect = list((x, y, w, h))

        # 얼굴을 표시하는 직사각형 출력
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 직사각형 가운데 좌표 점으로 출력
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (255, 0, 0), -1)

        cv2.putText(frame, "Frame: {}".format(frames), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # print(face_rect)
    print(face_landmarks)

    # dict 저장
    face_location["landmarks"] = face_landmarks
    face_location["rect"] = face_rect
    face_location["frame"] = frames

    # json 쓰기
    with open('face.json', 'w') as make_file:
        json.dump(face_location, make_file,indent="\t")

    # 인식 중이 아니면 초기화
    face_landmarks = []
    face_rect = []

    # 영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q 입력 시 종료
    if key == ord("q"):
        break

# 마무리
cv2.destroyAllWindows()
vs.stop()