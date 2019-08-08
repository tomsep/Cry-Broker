from flask import Flask
import time

app = Flask(__name__)


@app.route('/')
def timestamp():
    return str(int(time.time()))



if __name__ == '__main__':
    Flask.run(app, host='localhost', port=27420)