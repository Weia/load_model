import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import crop_image
#加载第一阶段模型，进行预测
path=r'/media/weic/新加卷/数据集/数据集/学生照片/test'#预测图片文件夹
result=open('final_model/final_result.txt','w+')

posi_to_path = [
        [0, 1, 2],
        [3, 4, 5, 6, 7, 8],
        [3, 9, 11, 13],
        [4, 10, 12, 14],
        [15]]

def load_a_model(model_path):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:

        with sess.as_default():
            with sess.graph.as_default():
                new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')  # 载入图结构
                new_saver.restore(sess,ckpt.model_checkpoint_path)

        #graph = tf.get_default_graph()
                input = graph.get_tensor_by_name('input/input_images:0')
        # loss=graph.get_tensor_by_name('loss/cross_entropy_loss:0')
                output = graph.get_tensor_by_name('inference/output:0')
        print('load model  '+ckpt.model_checkpoint_path+' success')
    else:
        raise FileNotFoundError('model not found %s'%(model_path))
    return sess,input,output


# graph=tf.Graph()
# graph1=tf.Graph()
# sess=tf.Session(graph=graph)
# sess1=tf.Session(graph=graph1)
model_path=['model1.1','model_head','model_body','model_left','model_right','model_foot']
sesses = []
inputs =[]
outputs = []
try:
    for i in range(len(model_path)):
        part_sess,part_input,part_output=load_a_model(model_path[i])
        sesses.append(part_sess)
        inputs.append(part_input)
        outputs.append(part_output)

except FileNotFoundError as info:

    print(info)
    exit()



imgNames=os.listdir(path)
for name in imgNames:
    imgPath = os.path.join(path, name)
    try:
        or_image = Image.open(imgPath)
    except Exception as info:
        print(info)
        continue
    or_width, or_height = or_image.size
    result.write(imgPath + ' ')
    result.write(str(or_width)+' '+str(or_height)+' ')


    image = or_image.resize((256, 256), Image.ANTIALIAS)
    images = np.expand_dims(image, 0)
    print(images.shape)
    result_1 = []

    with sesses[0].as_default():
        with sesses[0].graph.as_default():
            test_full = sesses[0].run(outputs[0], feed_dict={inputs[0]: images})
        (width, height) = test_full[0][0].shape

        for i in range(16):
            position = np.argmax(test_full[0][i])
            y, x = divmod(position, width)
            result_1.append((x * or_width / width, y * or_height / height))
        print(result_1[15])
    plt.imshow(or_image)
    label = np.asarray(result_1).reshape(-1, 2)
    plt.plot(label[:, 0], label[:, 1], 'r*')
    plt.show()

    np_result=np.asarray(result_1).reshape(-1,2)
    print(result_1)
    parts_images,parts_marge=crop_image._crop_image(or_image,np_result,posi_to_path)

    parts_results=[]
    for i in range(5):
        part_image=np.expand_dims(parts_images[i].resize((64,64),Image.ANTIALIAS),0)
        or_part_width,or_part_height=parts_images[i].size
        with sesses[i+1].as_default():
            with sesses[i+1].graph.as_default():
                test=sesses[i+1].run(outputs[i+1],feed_dict={inputs[i+1]: part_image})
                (width, height) = test[0][0].shape
                one_result=[]
                all_x=[]
                all_y=[]
                for j in range(len(posi_to_path[i])):
                    print('len',len(posi_to_path[i]))
                    position = np.argmax(test[0][j])
                    y, x = divmod(position, width)
                    print(y, x)
                    print('posi',posi_to_path[i][j])
                    one_result.append([x * or_part_width / width + parts_marge[i][0],
                                                      y * or_part_height / height + parts_marge[i][1]])
                    all_x.append(x*or_part_width/width)
                    all_y.append(y*or_part_height/height)
                plt.imshow(parts_images[i])
                plt.plot(all_x,all_y,'r*')
                plt.show()
        parts_results.append(one_result)
    print(parts_results)
    new_result=[0]*16
    for i in range(5):
        for j in range(len(posi_to_path[i])):
            index=posi_to_path[i][j]
            if new_result[index] != 0:
                print('new_result',new_result[index])
                new_x=(new_result[index][0]+parts_results[i][j][0])/2
                new_y=(new_result[index][1]+parts_results[i][j][1])/2
                new_result[index] =[new_x,new_y]
            else:
                new_result[index] = parts_results[i][j]
    print(new_result)




    # plt.imshow(or_image)
    # label = np.asarray(new_result).reshape(-1, 2)
    # plt.plot(label[:, 0], label[:, 1], 'r*')
    # plt.show()

    for i in range(len(result_1)):
        result.write(str(result_1[i])+' ')
    result.write('\n')

















    #loss=sess.run(graph.get_tensor_by_name('loss/train_loss:0'),feed_dict={'input_image':image,'labels':label})
    # print(loss)
