from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import math


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


#비디오 쓰레드 시작, 웹 캠에서 영상 얻음
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

#얼굴 인식이 벗어났을 때 최근 인식한 각도
l_pitch=l_roll=l_yaw=0

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
        #얼굴을 표시하는 직사각형
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        #얼굴 각도 계산
        pitch,roll,yaw=pose_aspect_angle(rect,shape)
        l_pitch=pitch
        l_roll=roll
        l_yaw=yaw

        #print("in for :: x:%f,y:%f", x + w / 2, y + h / 2)

        #얼굴 각도를 얻기 위한 좌표 지정
        dots=np.array([
            (shape[30][0], shape[30][1]),  # nose tip
            (shape[8][0], shape[8][1]),  # chin
            (shape[36][0], shape[36][1]),  # left eye left corner
            (shape[45][0], shape[45][1]),  # right eye right corner
            (shape[48][0], shape[48][1]),  # left mouth corner
            (shape[54][0], shape[54][1])  # right mouth corner
        ])

        #얼굴을 표시하는 직사각형 출력
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #직사각형 가운데 좌표 점으로 출력
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (255, 0, 0), -1)

        #위에서 지정한 좌표 값 출력
        for (x, y) in dots:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        #텍스트 출력
        cv2.putText(frame, "PTICH: {}".format(pitch), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "ROLL: {}".format(roll), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAW: {}".format(yaw), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "L_PITCH: {}".format(l_pitch), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "L_ROLL: {}".format(l_roll), (160, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "L_YAW: {}".format(l_yaw), (300, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #영상 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #q 입력 시 종료
    if key == ord("q"):
        break

#마무리
cv2.destroyAllWindows()
vs.stop()