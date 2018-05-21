import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt

def cal_distance(pointA,pointB):
    diff=pointA-pointB

    return np.sqrt((np.sum(np.power(diff,2))))

def justify_is_full_body(label_file,save_file):
    contents= open(label_file).readlines()
    f_save=open(save_file,'w')
    for line in contents:
        list_line=line.split(' ')
        img_path=list_line[0]
        labels=np.asarray([float(x) for x in list_line[1:-1]]).reshape(-1,2)
        height=labels[15][1]-labels[0][1]
        left_arm=cal_distance(labels[12],labels[4])
        right_arm=cal_distance(labels[13],labels[5])
        shoulders=cal_distance(labels[5],labels[4])
        cal_height=left_arm+right_arm+shoulders


        if (height/cal_height)>1.8:
            f_save.write(img_path)
            f_save.write('\n')
            print(height/cal_height)


    pass




def checkout_high_loss_picture(images_file):
    names=open(images_file).readlines()
    for name in names:
        img_path=name.replace('\n','')
        img=Image.open(img_path)
        plt.imshow(img)
        plt.show()

# justify_is_full_body('/home/weic/project/load_model/diff_init_5_20/model520_result.txt','error_imgs.txt')
# checkout_high_loss_picture('error_imgs.txt')