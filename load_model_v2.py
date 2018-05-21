import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
#加载模型，进行预测
path=r'/media/weic/新加卷/数据集/数据集/学生照片/test'#预测图片文件夹
result=open('diff_init_5_20/520_result.txt','w+')
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('diff_init_5_20')  # 通过检查文件锁定最新模型,时间
    if ckpt and ckpt.model_checkpoint_path:#ckpt.model_checkpoint_path最新的模型
        new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')  # 载入图结构
        new_saver.restore(sess,ckpt.model_checkpoint_path)
        # for val in tf.trainable_variables():
        #     print(val.name, val.value)
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name('input/input_images:0')
        # loss=graph.get_tensor_by_name('loss/cross_entropy_loss:0')
        output = graph.get_tensor_by_name('inference/output:0')

        imgNames=os.listdir(path)
        hotmap_result=[]
        try:
            for name in imgNames:
                imgPath = os.path.join(path, name)
                try:
                    image = Image.open(imgPath)
                except Exception as info:
                    print(info)
                    continue
                or_width, or_height = image.size
                image = image.resize((256, 256), Image.ANTIALIAS)

                images = np.expand_dims(image, 0)
                print(images.shape)

                test = sess.run(output, feed_dict={input: images})
                hotmap_result.append(test[0])
                result.write(imgPath + ' ')
                # 将width 和height 写入文件
                # result.write(str(or_width)+' '+str(or_height)+' ')
                (width,height)=test[0][0].shape
                # print(width,height)
                for i in range(16):
                    position=np.argmax(test[0][i])
                    y,x=divmod(position,width)
                    result.write(str(x*or_width/width)+' ')
                    result.write(str(y*or_height/height)+' ')
                    result.write(str(test[0][i][y][x])+' ')
                result.write('\n')
        except Exception as info:
            #np.save('result.npy', np.asarray(final_result))
            pass
        finally:
            np.save('result/result.npy', np.asarray(hotmap_result))



    #loss=sess.run(graph.get_tensor_by_name('loss/train_loss:0'),feed_dict={'input_image':image,'labels':label})
    # print(loss)
