from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restplus import Api
from landmark_process import landmark_process,sleep_data_calc

app = Flask(__name__)

CORS(app)

api = Api(app, version='1.0', title='myProject')

#졸음 정보 딕셔너리 생성
sleep_data=dict()

#졸음 정보 변수
rear=lear=mear=0
pitch=roll=yaw=0

@app.route("/set_face",methods=['POST'])
def set_landmark():
    if request.method == 'POST':
        #서버로부터 좌표 정보 획득
        face_location=request.json

        #운전자 인식 여부 변수 받음
        driver = face_location["driver"]

        #운전자 인식 여부
        if driver:
            landmark_process.set_face_landmarks(landmark_process, face_location['landmarks'])
            landmark_process.set_face_rect(landmark_process, face_location['rect'])

            sleep_data_calc.set_data(sleep_data_calc,landmark_process.return_sleep_data(landmark_process))
            sleep_data=sleep_data_calc.calc_sleep_data(sleep_data_calc)
            print(sleep_data)

            return jsonify(sleep_data)
        else:
            sleep_data=sleep_data_calc.no_driver(sleep_data_calc)
            print(sleep_data)

            return jsonify(sleep_data)


@app.route("/feedback",methods=['POST'])
def get_feedback():
    if request.method=='POST':
        # 피드백으로 졸음이 아닌 경우 단계 롤백
        sleep_data_calc.cancle_sleep_step(sleep_data_calc)

    return "a"


@app.route("/reset",methods=['POST'])
def reset_data():
    if request.method =='POST':
        sleep_data_calc.reset_data(sleep_data_calc)
        sleep_data_calc.print_sleep_data(sleep_data_calc)

    return "a"


@app.route("/drop",methods=['POST'])
def drop_sleep_step():
    if request.method =='POST':
        sleep_data_calc.drop_sleep_step(sleep_data_calc)
        sleep_data_calc.print_sleep_data(sleep_data_calc)

    return "a"

@app.route("/timer",methods=['POST'])
def get_timer():
    if request.method=='POST':
        sleep_data=sleep_data_calc.get_sleep_data(sleep_data_calc)
        sleep_data_calc.reset_blink(sleep_data_calc)

        return jsonify(sleep_data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port="5000")