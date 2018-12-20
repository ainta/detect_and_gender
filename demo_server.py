
from flask import Flask, request, send_from_directory
import tornado.wsgi
import tornado.httpserver

import cv2
import base64
import numpy as np

from SSH.test import detect_cv2, detect
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe

import h5py
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense

app = Flask(__name__)

def img2tag(img):
    return '<img src="data:image/png;base64, {}" />'.format(base64.b64encode(cv2.imencode('.png', img)[1]))

def addlink(name):
    return '<a href=sample/{0}>{0}</a><br/>'.format(name)

@app.route('/sample/<path:path>')
def sample(path):
    return send_from_directory('sample', path)

@app.route('/', methods=['GET', 'POST'])
def main():
    re = "Examples <br/>"
    re += addlink('1.jpg')
    re += addlink('2.jpg')
    re += addlink('3.jpg')
    re += addlink('4.jpg')
    re += addlink('5.png')
    re += addlink('6.png')
    re += addlink('7.jpg')
    re += addlink('8.png')
    re += addlink('9.png')
    re += addlink('11.png')
    re += addlink('12.png')
    re += addlink('13.png')
    re += addlink('14.png')
    re += addlink('15.png')
    re += addlink('16.png')
    re += '''<br/><form action="/" method="POST" enctype="multipart/form-data">
    <input type="file" name="img" id="img" accept="image/*" />
    <input type="submit" value="demo image" />
    </form>'''
    if request.method == 'POST':
        image = np.asarray(bytearray(request.files.get('img').stream.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #(detect_cv2(net, image))
        
        boxes = detect_cv2(app.ssh.net, image)
        
        inds = np.where(boxes[:, -1] >= 0.5)[0]
        boxes = boxes[inds]
        
        faces = np.empty((boxes.shape[0], 224, 224, 3))
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            w0 = box[3] - box[1]
            w1 = box[2] - box[0]
            
            i0s = max([int(box[1] - w0*0.3), 0])
            i0e = min([int(box[3] + w0*0.3), image.shape[0]-1])
            i1s = max([int(box[0] - w1*0.3), 0])
            i1e = min([int(box[2] + w1*0.3), image.shape[1]-1])

            faces[i,:,:,:] = cv2.resize(image[i0s:i0e, i1s:i1e], (224, 224))
            re += img2tag(image[i0s:i0e, i1s:i1e])

        result = app.model.predict(faces)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            val = result[i]
            re += str(val)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255.*val[1], 0,255.*val[0]), 2)

        re += img2tag(image)

    return re

class SSHNet(object):

    def __init__(self):
        cfg_from_file('SSH/configs/wider.yml')
        cfg.GPU_ID = 0

        # Loading the network
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.net = caffe.Net('SSH/models/test_ssh.prototxt', 'SSH-FL-OHEM-ver2_iter_40000.caffemodel', caffe.TEST)
        self.net.name = 'SSH'

def start_tornado(app, port=8880):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    app.ssh = SSHNet()

    app.vgg = VGG16(weights='imagenet')
    app.x = app.vgg.get_layer('fc2').output
    app.prediction = Dense(2, activation='softmax', name='predictions')(app.x)
    app.model = Model(input=app.vgg.input, outputs=app.prediction)
    app.model.load_weights("./gender/pretrained_weight.hdf5")

    start_tornado(app)

