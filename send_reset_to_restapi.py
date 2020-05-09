import requests

def reset():
    requests.post('http://127.0.0.1:5000/reset', json="reset")

def drop():
    requests.post('http://127.0.0.1:5000/drop', json="drop")

#drop()
reset()