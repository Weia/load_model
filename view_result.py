# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 9:49
# @Author  : weic
# @FileName: view_result.py
# @Software: PyCharm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
with open('./result.txt') as f:
    results=f.readlines()

    for result in results:
        #print(type(result))
        #print(result)
        list_result=result.split(' ')
        #print(list_result)
        imgPath=list_result[0]
        #print(list_result[1].split(' '))
        label=np.asarray([float(x) for x in list_result[1:-1]]).reshape(-1,2)


        x=label[:,0]
        y=label[:,1]

        print(label)
        print(x)
        print(y)
        img=Image.open(imgPath)
        plt.imshow(img)
        plt.plot(x,y,'r*')
        plt.show()

