# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 9:49
# @Author  : weic
# @FileName: view_result.py
# @Software: PyCharm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_highloss_name_label():
    with open('diff_init_5_20/acc_result.txt') as f:
        results = f.readlines()

    with open('model_foot/test_result.txt') as f_name:
        names = f_name.readlines()

    what = np.load('result.npy')
    num = what.shape[0]

    f_img_name = open('over100.txt', 'w')

    for index, line in enumerate(results):
        list_line = [float(x) for x in line.split(' ')[:-1]]
        # if list_line[15] > 100:
        #     f_img_name.write(names[index])
        #     a_result = what[index]
        #     map = a_result[15]
        #     position = np.argmax(map)
        #     y, x = divmod(position, 64)
        #     print(map[x][y])
        #     plt.matshow(map)
        #     plt.show()
        all_x=[]
        all_y=[]
        img_path=names[index].replace(' \n','')
        img = Image.open(img_path)
        width,height=img.size
        a_result = what[index]
        for i in range(1):
            position = np.argmax(a_result[i])
            y, x = divmod(position, 64)
            all_x.append(x*width/64)
            all_y.append(y*height/64)
            print(np.max(a_result[i]))
            plt.matshow(a_result[i])
            plt.show()
            # print('x,y',x,y)
            # print(a_result[i][y][x])
            # print('where',np.where(a_result[i]==np.max(a_result[i])))
        print('*'*20)
        plt.imshow(img)
        plt.plot(all_x,all_y,'r*')
        plt.show()




get_highloss_name_label()

def checkout_high_loss_picture(images_file):
    names=open(images_file).readlines()
    for name in names:
        img_path=name.replace(' \n','')
        img=Image.open(img_path)
        plt.imshow(img)
        plt.show()
checkout_high_loss_picture('low_pro.txt')


def get_low_pro_name():
    what = np.load('result.npy')
    num = what.shape[0]
    f=open('low_pro.txt','w')
    with open('diff_init_5_20/test_result.txt') as f_name:
        names = f_name.readlines()
    for i in range(num):
        a_result=what[i]
        map = a_result[15]
        position = np.argmax(map)
        y, x = divmod(position, 64)

        if np.abs(map[x][y])<0.00005:
            f.write(names[i])
    f.close()

# get_low_pro_name()








