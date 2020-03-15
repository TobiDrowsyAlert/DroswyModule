from landmark_process import landmark_process
import sleep_step_calcaulation
import socket
import json

#졸음 정보 딕셔너리 생성
sleep_data=dict()

#눈의 ear 값 임계치를 결정
EYE_AR_THRESH = 0.21
#눈 깜빡임 인식 임계 프레임
EYE_AR_CONSEC_FRAMES = 3
#눈 감은 동안의 프레임 수
E_COUNTER = 0
#눈 깜빡임 횟수
blink=0


#입 크기 비율 임계치 결정
MOUTH_AR_THRESH = 0.4
#입이 열려있는 동안의 최소 프레임 결정
MOUTH_AR_CONSEC_FRAMES = 60
#입이 열려있는 동안 프레임 저장
M_COUNTER = 0
#하품 횟수
yawn = 0

#서버의 주소와 포트 설정
HOST='127.0.0.1'
PORT=54321

#소켓 설정 및 연결
clnt_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clnt_socket.connect((HOST, PORT))

while True:
    #socket 데이터 받음
    recv_data=clnt_socket.recv(1024)
    face_data=recv_data.decode()
    face_data=json.loads(face_data)
    #print(face_data)
    print(face_data["frame"])

    landmark=face_data['landmarks']
    rect=face_data['rect']

    # 얼굴 좌표 set
    landmark_process.set_face_landmarks(landmark_process, landmark)
    landmark_process.set_face_rect(landmark_process, rect)

    #운전자가 인식 중일때만 동작
    if landmark_process.detect_driver(landmark_process):
        # 졸음 징후 정보 계산
        rear, lear = landmark_process.get_eye_aspect_ratio(landmark_process)
        mear = landmark_process.get_mouth_aspect_ratio(landmark_process)
        pitch, roll, yaw = landmark_process.get_pose_angle_aspect(landmark_process)

        # 양 눈이 임계치 보다 작은 동안의 프레임 수를 측정
        if lear < EYE_AR_THRESH and rear < EYE_AR_THRESH:
            E_COUNTER += 1

        # 양 눈이 임계치보다 큰 조건에 수행
        else:
            # 눈이 감겨있던 동안의 프레임을 검사하여 눈 깜빡임 계산
            if E_COUNTER >= EYE_AR_CONSEC_FRAMES:
                blink += 1
            # 프레임 수 초기화
            E_COUNTER = 0

        # 입이 임계치보다 큰 동안 프레임 수 측정
        if mear > MOUTH_AR_THRESH:
            M_COUNTER += 1

        # 입이 임계치보다 작은 조건 하에 수행
        else:
            # 입이 열려있던 동안의 프레임을 측정하여 하품 수 계산
            if M_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                yawn += 1

            # 프레임 수 초기화
            M_COUNTER = 0

        # 졸음 정보 딕셔너리 저장
        sleep_data["rightEAR"] = rear
        sleep_data["leftEAR"] = lear
        sleep_data["mouthEAR"] = mear
        sleep_data["pitch"] = pitch
        sleep_data["roll"] = roll
        sleep_data["yaw"] = yaw
        sleep_data["blink"]=blink
        sleep_data["yawn"]=yawn
        sleep_data["driver"] = True

    else:
        sleep_data["driver"]=False

    print(sleep_data)

    #졸음 정보를 소켓을 통해 전송
    data=json.dumps(sleep_data)
    clnt_socket.sendall(data.encode())
    #break

clnt_socket.close()

