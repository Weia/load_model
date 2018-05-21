import numpy as np
import os
from PIL import Image
import csv
import matplotlib.pyplot as plt


#截取图片的部分，作为下个模型的输入
def _crop_image(img,label,posi_to_path):

    img_width,img_height=img.size
    ##print('img_width',img_width,img_height)
    part_images=[]
    marge=[]
    for position in posi_to_path:

        valid_label = label[position, :]
        no_zero_label=np.asarray([x for x in valid_label if not np.array_equal(x,[0.0,0.0])]).reshape(-1,2)
        if len(no_zero_label)==0:
            raise IndexError
        else:
            x_min,y_min=np.min(no_zero_label,axis=0)
            x_max,y_max=np.max(no_zero_label,axis=0)
            if x_max>img_width or y_max>img_height:
                raise ValueError
            width=x_max-x_min
            height=y_max-y_min
            width=(800 if width<20 else width)
            height=(800 if height<20 else height)
            #print('width',width,height)
            x_min=(1 if x_min-width*0.5<0 else x_min-width*0.5)
            y_min=(1 if y_min-height*0.5<0 else y_min-height*0.5)
            x_max = (img_width-1 if x_max + width * 0.5>img_width else x_max + width * 0.5)
            y_max = (img_height-1 if y_max + height * 0.5>img_height else y_max + height * 0.5)



            part_image=img.crop((x_min,y_min,x_max,y_max))
            part_images.append(part_image)
            marge.append([x_min,y_min])
    return part_images,marge


# img_dir='/media/weic/新加卷/数据集/数据集/学生照片/test'
# label='/media/weic/新加卷/标注文件/test.txt'
# content=open(label).readlines()
# posi_to_path = [
#         [0, 1, 2],
#         [3, 4, 5, 6, 7, 8],
#         [3, 9, 11, 13],
#         [4, 10, 12, 14],
#         [15]]
# for name in os.listdir(img_dir):
#     img=Image.open(img_dir+'/'+name)
#     for line in content:
#         label_name=line.split(' ')[0]
#         label=np.asarray([float(x) for x in line.split(' ')[1:-1]]).reshape(-1,2)
#         if name==label_name:
#             print(label)
#
#             parts,marge=_crop_image(img,label,posi_to_path)
#             print(len(parts))
#             for i in range(len(parts)):
#                 width,height=parts[i].size
#                 print(width,' ',height)
#                 plt.imshow(parts[i].resize((64,46)))
#                 plt.show()


