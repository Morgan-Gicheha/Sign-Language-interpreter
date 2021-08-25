from flask import Flask
from camera import main_camera
app = Flask(__name__)

@app.route('/')
def home():
    
    return '<h1>Welcome,</h1> <br> <p>navigate to "/golive" to activate the camera and predict</p>'

@app.route('/golive')
def golive():
    main_camera()


if __name__ == '__main__':
    app.run()