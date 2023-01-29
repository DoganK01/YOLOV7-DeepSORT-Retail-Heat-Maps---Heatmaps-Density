#Created by Augmented Startups
#YOLOv7 Flask Application
#Enroll at www.augmentedstartups.com/store
from cProfile import label
from decimal import ROUND_HALF_UP, ROUND_UP
from wsgiref.validate import validator
from flask import Flask, render_template, Response,jsonify,request,session,make_response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
from flask_bootstrap import Bootstrap
import cv2
from time import time
import requests
from hubconfCustom import video_detection
app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = 'daniyalkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    # text = StringField(u'Conf: ', validators=[InputRequired()])
    conf_slide = IntegerRangeField('Confidence:  ', default=25,validators=[InputRequired()])
    submit = SubmitField("Run")
    




detectedObjects = 0
def generate_frames(path_x = '',conf_= 0.25):
    yolo_output = video_detection(path_x,conf_)
    
    for detection_,detection in yolo_output:
        # # print(detection_)
        ref,buffer=cv2.imencode('.jpg',detection_)


        global detectedObjects
        detectedObjects = str(detection)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        



@app.route("/",methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])

def home():
    session.clear()
    return render_template('root.html')


@app.route('/FrontPage',methods=['GET','POST'])
def front():
    # session.clear()
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        # conf_ = form.text.data
        conf_ = form.conf_slide.data
        
        # # print(round(float(conf_)/100,2))
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        session['conf_'] = conf_
    return render_template('video.html',form=form)


@app.route('/video')
def video():
    temp_res = generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2))
    if(temp_res):
        return Response(temp_res,mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        global detectedObjects
        detectedObjects = 0

        session['video_path'] = ''
        session['conf_'] = 0.25
        session.clear()



@app.route('/detectionCount',methods = ['GET'])
def detect_fun():
    global detectedObjects
    return jsonify(detectCount=detectedObjects)







if __name__ == "__main__":
    app.run(debug=True)
