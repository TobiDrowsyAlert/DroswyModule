from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restplus import Api
from landmark_process import landmark_process,sleep_data_calc

app = Flask(__name__)

CORS(app)

api = Api(app, version='1.0', title='myProject')

userlist=dict()

@app.route("/login",methods=['POST'])
def check_userid():
    global userlist
    if request.method =='POST':
        appdata=request.json
        userid=appdata["userId"]
        print("login ID : {}".format(userid))
        messeage=dict()
        if not userid in userlist:

            userlist[userid]=[landmark_process(),sleep_data_calc()]
            msg="Login of {} was successful!".format(userid)
            print(msg)
            messeage["msg"]=msg
            return jsonify(messeage)

        else:
            msg="Login of {} failed!"
            print(msg)
            print("ID {} already exists".format(userid))
            messeage["msg"]=msg
            return jsonify(messeage)

@app.route("/personal",methods=['POST'])
def set_personal_data():
    global userlist
    if request.method=='POST':
        data=request.json
        userid=data["userId"]
        print(data)
        message=dict()
        if userid in userlist:
            user=userlist[userid]

            user[1].set_persnal_data(data)
            msg=" ID {} set personal data".format(userid)
            print(msg)
            message["msg"]=msg
            return jsonify(message)

@app.route("/logout",methods=['POST'])
def delete_userid():
    global userlist
    if request.method =='POST':
        appdata = request.json
        userid = appdata["userId"]
        print("logout ID : {}".format(userid))
        message=dict()

        if userid in userlist:
            del userlist[userid]
            msg="Logout of {} was successful!".format(userid)
            print(msg)
            message["msg"]=msg
            return jsonify(message)
        else:
            msg="Logout of {} was failed!".format(userid)
            print(msg)
            print("ID {} does not exist".format(userid))
            message["msg"]=msg
            return jsonify(message)

@app.route("/set_face",methods=['POST'])
def set_landmark():
    global userlist
    if request.method == 'POST':
        #서버로부터 좌표 정보 획득
        app_data=request.json
        #운전자 인식 여부 변수 받음
        userid=app_data["userId"]
        driver = app_data["driver"]

        #해당 id가 존재하면 수행
        if userid in userlist:
            print("sleep calcaultion of {}".format(userid))
            userdata=userlist[userid]

            # 운전자 인식 여부
            if driver:
                userdata[0].set_face_landmarks(app_data["landmarks"])
                userdata[0].set_face_rect(app_data['rect'])

                landmark_data=userdata[0].return_sleep_data()
                userdata[1].set_data(landmark_data)
                sleep_data=userdata[1].calc_sleep_data()
                print(sleep_data)
                return jsonify(sleep_data)

            else:
                sleep_data=userdata[1].no_driver()
                return jsonify(sleep_data)

@app.route("/stretch",methods=['POST'])
def do_stretch():
    global userlist
    if request.method=='POST':
        app_data = request.json
        userid = app_data["userId"]
        driver = app_data["driver"]

        if userid in userlist:
            print("do stertch of {}".format(userid))
            userdata=userlist[userid]

            if driver:
                userdata[0].set_face_landmarks(app_data["landmarks"])
                userdata[0].set_face_rect(app_data["rect"])

                landmark_data=userdata[0].return_sleep_data()
                userdata[1].set_data(landmark_data)
                stretch_data=userdata[1].do_stretch()
                return jsonify(stretch_data)

            else:
                sleep_data=userdata[1].no_driver()
                return jsonify(sleep_data)


@app.route("/feedback",methods=['POST'])
def get_feedback():
    global userlist
    if request.method=='POST':
        # 피드백으로 졸음이 아닌 경우 단계 롤백
        appdata = request.json
        userid = appdata["userId"]
        message = dict()

        if userid in userlist:
            print("rollback sleep step of {}".format(userid))
            userdata=userlist[userid]

            userdata[1].cancle_sleep_step()


            msg="Feedback of ID {} was successful".format(userid)
            print(msg)
            message["msg"]=msg

            return jsonify(message)



@app.route("/reset",methods=['POST'])
def reset_data():
    global userlist
    if request.method == 'POST':
        #졸음 단계 리셋
        appdata = request.json
        userid = appdata["userId"]
        message=dict()
        if userid in userlist:
            print("reset sleep data of {}".format(userid))
            userdata = userlist[userid]

            userdata[1].reset_data()
            userdata[1].print_sleep_data()

            msg="Reset of ID {} was successful".format(userid)
            print(msg)
            message["msg"] = msg

            return jsonify(message)

@app.route("/drop",methods=['POST'])
def drop_sleep_step():
    global userlist
    if request.method =='POST':
        #졸음 단계 하락
        appdata = request.json
        userid = appdata["userId"]
        message=dict()
        if userid in userlist:
            print("drop sleep step of {}".format(userid))
            userdata=userlist[userid]

            userdata[1].drop_sleep_step()
            userdata[1].print_sleep_data()

            msg = "Drop sleep step of ID {} was successful".format(userid)
            print(msg)
            message["msg"] = msg

            return jsonify(message)


@app.route("/timer",methods=['POST'])
def get_timer():
    global userlist
    if request.method=='POST':
        app_data = request.json
        userid = app_data["userId"]

        if userid in userlist:
            userdata = userlist[userid]
            # 취약시간 여부
            if app_data["isWeakTime"]:
                userdata[1].senstitive_sleep_symptom()
            else:
                userdata[1].non_weak_time()

            print("timer actuation of {}".format(userid))


            sleep_data=userdata[1].get_sleep_data()
            print(sleep_data)
            userdata[1].reset_blink()
            userdata[1].reset_nod()

            return jsonify(sleep_data)

@app.route("/lookuser",methods=['POST'])
#접속한 유저 리스트 확인
def lookup_user():
    global userlist
    if request.method=='POST':
        userids=list(userlist.keys())
        print(userids)

        return jsonify(userids)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port="5000")