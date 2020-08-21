from landmark_process import landmark_process
from imutils.video import VideoStream
from imutils import face_utils
import dlib
import imutils
import cv2
import random

def stretch_angle(angle,threshold,output,flag,angle_type):
    if not flag[1]:
        if not flag[0]:
            cv2.putText(output, "stretch {} angle!!".format(angle_type), (150, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if abs(angle) >= threshold:
                flag[0] = True
                if angle > 0:
                    flag[2] = True
                else:
                    flag[3] = True
        else:
            if flag[2]:
                cv2.putText(output, "stretch oppsite direction!!", (150, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if angle_type=="pitch":
                    cv2.putText(output, "Please stretch downward direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif angle_type=="yaw":
                    cv2.putText(output, "Please stretch right direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif angle_type=="roll":
                    cv2.putText(output, "Please stretch right direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if angle <= -threshold:
                    flag[3] = True
                    flag[1] = True
                    return

            elif flag[3]:
                cv2.putText(output, "Stretch oppsite direction!!", (150, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if angle_type=="pitch":
                    cv2.putText(output, "Please stretch upward direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif angle_type=="yaw":
                    cv2.putText(output, "Please stretch left direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif angle_type=="roll":
                    cv2.putText(output, "Please stretch left direction!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if angle >= threshold:
                    flag[2] = True
                    flag[1] = True
                    return
            else:
                return
    else:
        return

def reset_stretch_flag(flag):
    for i in range(len(flag)):
        flag[i]=False

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

#첫번째 변수 : 스트레칭 시작 여부
#두번째 변수 : 스트레칭 끝 여부
#세번째 변수 : 각도가 양수 일 때 스트레칭 여부
#네번째 변수 : 각도가 음수 일 때 스트레칭 여부
stretch_flag=[False,False,False,False]
stretch_order=["pitch","yaw","roll"]

stretch_order=random.sample(stretch_order,3)
order=0

frames=0

roll=0
pitch=0
yaw=0

roll_threshold=15
pitch_threshold=15
yaw_threshold=20

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

            if order<3:
                angle_name=stretch_order[order]
                if not stretch_flag[1]:
                    if angle_name == "pitch":
                        print("Start {} stretch".format(angle_name))
                        stretch_angle(pitch, pitch_threshold, frame, stretch_flag, angle_name)
                    elif angle_name == "roll":
                        print("Start {} stretch".format(angle_name))
                        stretch_angle(roll, roll_threshold, frame, stretch_flag, angle_name)
                    elif angle_name == "yaw":
                        print("Start {} stretch".format(angle_name))
                        stretch_angle(yaw, yaw_threshold, frame, stretch_flag, angle_name)
                else:
                    print("End {} stretch".format(angle_name))
                    order=order+1
                    reset_stretch_flag(stretch_flag)
            else:
                break

    cv2.putText(frame, "PITCH: {}".format(pitch), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "ROLL: {}".format(roll), (160, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "YAW: {}".format(yaw), (300, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if order==3:
        break
    # q 입력 시 종료
    if key == ord("q"):
        break

# 마무리
cv2.destroyAllWindows()
vs.stop()

