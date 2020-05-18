import requests
userid="admin"
def reset(userid):
    requests.post('http://127.0.0.1:5000/reset', json=userid)

def drop(userid):
    requests.post('http://127.0.0.1:5000/drop', json=userid)

def timer(userid):
    requests.post('http://127.0.0.1:5000/timer', json=userid)

def lookup(userid):
    requests.post('http://127.0.0.1:5000/lookuser',json=userid)
#drop(userid)
#reset(userid)
#timer(userid)
lookup(userid)