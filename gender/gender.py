import shutil
import numpy as np
gender=np.genfromtxt('./gender.txt')

for i in range(1, 40001):
    gen = 'female'
    if gender[i-1]==1:
        gen = 'male'
    f = "%06d.jpg" % i
    src = './img_align_celeba/'
    dir = './img_align_celeba/train/' + gen + '/'
    shutil.move(src + f, dir + f)


for i in range(40001, 50001):
    gen = 'female'
    if gender[i-1]==1:
        gen = 'male'
    f = "%06d.jpg" % i
    src = './img_align_celeba/'
    dir = './img_align_celeba/validation/' + gen + '/'
    shutil.move(src + f, dir + f)

