from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
from flask import Flask, Response
import imutils
import time
import cv2
from keras.preprocessing import image
from gpiozero import Buzzer
from time import sleep
#import telepot
import time
import os
import telegram
import telegram_send
buzzer = Buzzer(17)

# telegram_send.send(messages=["Wow that was easy!"])
# 
bot = telegram.Bot(token='1728241325:AAGy6Ooyd-gq7UQxYFPSwyKGvcQaienbGOc')
# print(bot.get_me())
# 
# updates = bot.get_updates()
# print([u.message.text for u in updates])
# 
# chat_id = bot.get_updates()[-1].message.chat_id
# print(chat_id)
# 
# bot.send_photo(chat_id=1269931525, photo=open('/home/pi/SmartGlass/dataset/Gudiya/IMG-20180623-WA0016.jpg', 'rb'))

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

app = Flask(__name__)
video = cv2.VideoCapture(0)
model = load_model('facefeatures_new_model.h5')
i=0
@app.route('/')
def index():
    return render_template('index.html')
def gen(video):
    i=0
    while True:
        ret,frame = video.read()
        face=face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
            img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            #print(pred)
                     
            name="None matching"
        
            if(pred[0][0]>0.5):
                name='Tanisha'
            cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            if name == "None matching":
                buzzer.on()
                sleep(2)
                buzzer.off()
                
            
        else:
            cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            buzzer.on()
            sleep(2)
            buzzer.off()
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 2 )
        i=i+1
        cv2.imwrite('frame'+str(i)+'.jpg',frame)
        bot.send_photo(chat_id=1269931525, photo=open('frame'+str(i)+'.jpg', 'rb'))
        os.remove('frame'+str(i)+'.jpg')
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host='localhost', port=5000)