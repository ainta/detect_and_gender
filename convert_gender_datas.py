import cv2
import numpy as np

from SSH.test import detect_cv2, detect
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe
import os

cfg_from_file('SSH/configs/wider.yml')
cfg.GPU_ID = 0

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net('SSH/models/test_ssh.prototxt', 'SSH-FL-OHEM-ver2_iter_40000.caffemodel', caffe.TEST)
net.name = 'SSH'

in_path = 'gender/img_align_celeba/train/male'
out_path = 'gender/datas/train/male'

c = 0
N = len(os.listdir(in_path))
for fn in os.listdir(in_path):
    c = c + 1
    if c%1000 == 0:
        print "{}/{}".format(c, N)
    image = cv2.imread(os.path.join(in_path, fn))
    boxes = detect_cv2(net, image)
    
    inds = np.where(boxes[:, -1] >= 0.5)[0]
    boxes = boxes[inds]

    basename = os.path.splitext(fn)[0]

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        w0 = box[3] - box[1]
        w1 = box[2] - box[0]
        
        i0s = max([int(box[1] - w0*0.3), 0])
        i0e = min([int(box[3] + w0*0.3), image.shape[0]-1])
        i1s = max([int(box[0] - w1*0.3), 0])
        i1e = min([int(box[2] + w1*0.3), image.shape[1]-1])

        cv2.imwrite(os.path.join(out_path, basename+'_'+str(i)+'.jpg'), image[i0s:i0e, i1s:i1e])

