from landmark_process import landmark_process
from matplotlib import pyplot as plt
from imutils.video import VideoStream
from imutils import face_utils
import dlib
import imutils
import cv2

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

roll=0
pitch=0
yaw=0

frames=0
nodriver=0
exception_nodriver=0

nod_THRESH=-15
nod=0
nod_delay=0
nod_flag=False

lp=landmark_process()
#비디오 스트림이 동작하는 동안 루프
while True:
    #출력 영상을 frame에 저장
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    #출력 영상과 별개로 회색조 영상 저장
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #얼굴을 회색조 영상에서 얻음
    rects = detector(gray, 0)

    if rects:
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            frames = frames + 1

            if nod_flag:
                nod_delay+=1
                if nod_delay==20:
                    nod_flag=False
                    nod_delay=0

            # 얼굴을 표시하는 직사각형
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # 얼굴 좌표 변수 저장
            face_landmarks = shape.tolist()
            face_rect = list((x, y, w, h))

            lp.set_face_landmarks(face_landmarks)
            lp.set_face_rect(face_rect)

            pitch,roll,yaw=lp.get_pose_angle_aspect()

            # 얼굴을 표시하는 직사각형 출력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 직사각형 가운데 좌표 점으로 출력
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (255, 0, 0), -1)

            cv2.putText(frame, "Frame: {}".format(frames), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    else:
        nod_delay = 0
        if (abs(pitch)<=30 and abs(roll)) and abs(yaw)>=30:
            exception_nodriver+=1

            cv2.putText(frame, "EXCEPT_NO_DRIVER: {}".format(exception_nodriver), (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif pitch<nod_THRESH:
            if not nod_flag:
                nod+=1
                nod_flag=True

        else:
            nodriver = nodriver + 1

            cv2.putText(frame, "NO_DRIVER: {}".format(nodriver), (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "nod: {}".format(nod), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "PITCH: {}".format(pitch), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "ROLL: {}".format(roll), (160, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "YAW: {}".format(yaw), (300, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q 입력 시 종료
    if key == ord("q"):
        break

# 마무리
cv2.destroyAllWindows()
vs.stop()

