#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 02:52:35 2019

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:33:56 2019

@author: root
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import os
import matplotlib.pyplot as plt
import random
from glob import glob
from tflearn.layers.conv import global_avg_pool
#import vgg16
import efficientnet_builder
tf.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1,0"
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

''' if you want to run this code, you need to put this code under the 'efficientnet' document.'''

'''SE-block
	U can delete this blcok and the result should be same 
	Maybe not the same but i donot think there is an obvious difference between with and without 		it'''

def squeeze_and_excitation(input_x,ratio=16):
    x=global_avg_pool(input_x)
    x=tf.layers.dense(x,units=input_x.shape[3]//ratio)
    x=tf.nn.relu(x)
    x=tf.layers.dense(x,units=input_x.shape[3])
    x=tf.nn.sigmoid(x)
    x=tf.reshape(x,[-1,1,1,input_x.shape[3]])
    scale=x*input_x
    return scale

'''leaky_relu'''

def leaky_relu(input_x, negative_slop=0.2):
    return tf.maximum(negative_slop*input_x, input_x)

'''PReLU'''

def PReLU(x,number,name=None):
        if name is None:
            name="alpha_{}".format(number)
            _alpha=tf.get_variable(name,shape=x.get_shape(),
                               initializer=tf.constant_initializer(0.0),
                               dtype=x.dtype)
            return tf.maximum(_alpha*x,x)
'''there is no function for this two activation function in tensorflow
    if tensorflow has then u can replace it with official function'''
    
''' pixel_shuffle_layer change H*W*r*r to (H*r)*(W*r)*C
    r is the ratio for upsampling
                                                '''
def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, [0,1,2,4,3])
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_,[1]) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_,[1]) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)

''' ESRGAN consists of dense block. Each dense block is unit of RRDB'''

def dense_block(input_x,i):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer = tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer = None,
                        activation_fn=None,
                        normalizer_fn=None):
        input_x=input_x
        x1 = leaky_relu(slim.conv2d(input_x, 64, 3, 1, scope='d_conv1_{}'.format(i)))
        x1=tf.concat([input_x,x1],axis=3)
        x2 = leaky_relu(slim.conv2d(x1, 64, 3, 1, scope='d_conv2_{}'.format(i)))
        x2=tf.concat([input_x,x1,x2],axis=3)
        x3 = leaky_relu(slim.conv2d(x2, 64, 3, 1, scope='d_conv3_{}'.format(i)))
        x3=tf.concat([input_x,x1,x2,x3],axis=3)
        x4 = leaky_relu(slim.conv2d(x3, 64, 3, 1, scope='d_conv4_{}'.format(i)))
        x4=tf.concat([input_x,x1,x2,x3,x4],axis=3)
        x5 = leaky_relu(slim.conv2d(x4, 64, 3, 1, scope='d_conv5_{}'.format(i)))
        x5=squeeze_and_excitation(x5,ratio=16)
        x5=x5*0.2
        x=x5+input_x
        return x
    
'''RRDB block 
    The number is for differentiating each dense block'''
    
def RRDB(input_x,number):
    input_x=input_x
    x=dense_block(input_x,number-2)
    x=dense_block(x,number-1)
    x=dense_block(x,number)
    x=0.2*x
    out=x+input_x
    return out

'''discriminator is a simple VGG structure'''

def discriminator(input_x,batch_size,reuse=False):
    with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                weights_regularizer = None,
                                activation_fn=None,
                                normalizer_fn=None):
                                
                 conv1 = leaky_relu(slim.conv2d(input_x, 64, 3, 1, scope='d_conv1'))
                 conv1_1 = leaky_relu(slim.conv2d(conv1, 64, 3, 2, scope='d_conv1_1'))

                 conv2 = leaky_relu(slim.conv2d(conv1_1, 128, 3, 1, scope='d_conv2')) 
                 conv2_1 = leaky_relu(slim.conv2d(conv2, 128, 3, 2, scope='d_conv2_1'))
                
                 conv3 = leaky_relu(slim.conv2d(conv2_1, 256, 3, 1, scope='d_conv3'))
                 conv3_1 = leaky_relu(slim.conv2d(conv3, 256, 3, 2, scope='d_conv3_1'))

                 conv4 = leaky_relu(slim.conv2d(conv3_1, 512, 3, 1, scope='d_conv4'))
                 conv4_1 = leaky_relu(slim.conv2d(conv4, 512, 3, 2, scope='d_conv4_1'))                 

                 conv_flat = tf.reshape(conv4_1, [batch_size, -1])
                 dense1 = leaky_relu(slim.fully_connected(conv_flat, 1024, scope='d_dense1'))
                 dense2 = slim.fully_connected(dense1, 1, scope='d_dense2')
                 global d_vars
                 d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                 return dense2, tf.nn.sigmoid(dense2) 

'''generator Use 23 RRDBs'''

def generator(input_x, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()
            # down_sample here
            # input_x = down_sample_layer(input_x)
            
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            weights_regularizer= None,
                            activation_fn=None,
                            normalizer_fn=None, ##tf.nn.relu
                            padding='SAME'):
            conv1 = leaky_relu(slim.conv2d(input_x, 64, 9, 1, scope='g_conv1'))
            #print(conv1)
            shortcut = conv1
            g1=RRDB(conv1,3)
            g2=RRDB(g1,6)
            g3=RRDB(g2,9)
            g4=RRDB(g3,12)
            g5=RRDB(g4,15)
            g6=RRDB(g5,18)
            g7=RRDB(g6,21)
            g8=RRDB(g7,24)
            g9=RRDB(g8,27)
            g10=RRDB(g9,30)
            g11=RRDB(g10,33)
            g12=RRDB(g11,36)
            g13=RRDB(g12,39)
            g14=RRDB(g13,42)
            g15=RRDB(g14,45)
            g16=RRDB(g15,48)
            g17=RRDB(g16,51)
            g18=RRDB(g17,54)
            g19=RRDB(g18,57)
            g20=RRDB(g19,60)
            g21=RRDB(g20,63)
            g22=RRDB(g21,66)
            g23=RRDB(g22,69)

            conv2 = leaky_relu(slim.conv2d(g23, 64, 3, 1, scope='g_conv2'))

            conv2_out = shortcut*0.2+conv2
           
            conv3 = leaky_relu(slim.conv2d(conv2_out, 256, 3, 1, scope='g_conv3'))
            
            shuffle1 = PReLU(pixel_shuffle_layer(conv3, 2, 64),2) 
            
            conv4 = leaky_relu(slim.conv2d(shuffle1, 256, 3, 1, scope='g_conv4'))
            shuffle2 = PReLU(pixel_shuffle_layer(conv4, 2, 64),3)
            
            conv5 = leaky_relu(slim.conv2d(shuffle2, 3, 9, 1, scope='g_conv5'))
            global g_vars
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
            return tf.nn.tanh(conv5)

''' adversarial_loss for generator and discriminator 
    efficientloss is perceptual loss
    and of coure RaGAN is used for improving image quality'''
    
def inference_loss(real, fake,real_feature,fake_feature):
    def inference_mse_content_loss(real, fake):
        return tf.reduce_mean(tf.abs(real-fake))
    def inference_adversarial_loss(x, y, w=1, type_='gan'):
        if type_=='gan':
            try:
                return w*tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return w*tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        elif type_=='lsgan':
            return w*(x-y)**2
        else:
            raise ValueError('no {} loss type'.format(type_))
    
#    phi_real=VGG16(real,reuse=False)
#    phi_fake=VGG16(fake,reuse=True)
#    phi_real=convo1
#    phi_fake=convo2
    efficientloss=tf.reduce_mean(tf.square(real_feature-fake_feature))
#    vgg16loss=tf.reduce_mean(tf.square(phi_real-phi_fake))#inference_mse_content_loss(real, fake)+
    content_loss = inference_mse_content_loss(real, fake)+efficientloss
    d_real_logits, d_real_sigmoid = discriminator(real, batch_size,reuse=False)
    d_fake_logits, d_fake_sigmoid = discriminator(fake, batch_size,reuse=True)
    d_fake_loss = tf.reduce_mean(inference_adversarial_loss((d_real_logits-tf.reduce_mean(d_fake_logits)), tf.ones_like(d_real_sigmoid)))
    d_real_loss = tf.reduce_mean(inference_adversarial_loss(d_fake_logits-tf.reduce_mean(d_real_logits), tf.zeros_like(d_fake_sigmoid)))
    g_fake_loss=tf.reduce_mean(inference_adversarial_loss((d_fake_logits-tf.reduce_mean(d_real_logits)), tf.ones_like(d_fake_sigmoid)))
    g_real_loss=tf.reduce_mean(inference_adversarial_loss((d_real_logits-tf.reduce_mean(d_fake_logits)), tf.zeros_like(d_real_sigmoid)))
    d_loss = 0.001*(d_fake_loss+d_real_loss)
    g_loss = 0.001*(g_fake_loss+g_real_loss)
    return d_loss, g_loss, content_loss

'''Optimizer for GAN
    cop is for pretrain using L1 Loss
    gop and dop are for generator and discriminator, respectively'''
    
def model_optimizer(d_loss,g_loss,content_loss,learning_rate1,learning_rate,beta1,beta2,global_step):
    cop=tf.train.AdamOptimizer(learning_rate=learning_rate1,beta1=beta1,beta2=beta2).minimize(content_loss,var_list=g_vars)
    gop=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2).minimize(g_loss+content_loss,var_list=g_vars)
    dop=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2).minimize(d_loss,var_list=d_vars)
    return dop,gop,cop

''' This PSNR is only for testing outputs if u want a precise evaluation, matlabl PSNR function is a good choice'''

def PSNR(real, fake):
    mse = tf.reduce_mean(tf.square(255*((real-fake)/2+0.5)),axis=(-3,-2,-1))
    psnr = tf.reduce_mean(10 * (tf.log(255*255 / tf.sqrt(mse)) / np.log(10)))
    return psnr

'''train_highlist is 224*224*3; train_lowlist is 56*56*3;test_list is 56*56*3
    all data are saved in documents in png or jpg or any other format for picture
    In preprocessing, all data are normalized to [-1,1] '''
    
train_highlist=glob(os.path.join('data','train','HE','high','*.*'))
train_lowlist=glob(os.path.join('data','train','HE','low','*.*'))
test_list=glob(os.path.join('newtest','*.*'))
batch_size=2
test=np.zeros([0,56,56,3])
for n in range(batch_size):
    imga=scipy.misc.imread(test_list[n], mode='RGB')
    imga=np.reshape(imga,[1,56,56,3])
    test=np.concatenate([test,imga],axis=0)
test=2*(test/255-0.5)

input_x=tf.placeholder(tf.float32,[batch_size,224,224,3],name='input_x')
low=tf.placeholder(tf.float32,[batch_size,56,56,3],name='low')
global_step=tf.Variable(0,trainable=False,name='global_step')
start_learning_rate=0.0001
learning_rate=tf.train.exponential_decay(start_learning_rate,global_step=global_step,decay_steps=10000,decay_rate=0.87,staircase=False)
learning_rate1=tf.train.exponential_decay(start_learning_rate,global_step=global_step,decay_steps=10000,decay_rate=0.87,staircase=False)
beta1=0.9
beta2=0.999
real=input_x
fake=generator(low,reuse=False)
all_im=tf.concat([real,fake],axis=0)
features,endpoints=efficientnet_builder.build_model_base(all_im,'efficientnet-b4',training=False)
picture=tf.multiply(tf.add(tf.divide(fake,2),0.5),255,name='picture')

noise=PSNR(real,fake)
all_feature=endpoints['reduction_5']
real_feature=all_feature[0:2]
fake_feature=all_feature[2:4]

d_loss,g_loss,content_loss=inference_loss(real,fake,real_feature,fake_feature)
loss=tf.add(tf.add(d_loss,g_loss),content_loss)
dop,gop,cop=model_optimizer(d_loss,g_loss,content_loss,learning_rate1,learning_rate,beta1,beta2,global_step)
init=tf.global_variables_initializer()
tf.get_collection("nodes")
tf.add_to_collection("nodes",input_x)
tf.add_to_collection("nodes",picture)
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        num=random.randint(0,len(train_highlist)-batch_size)
        batch_highdata=train_highlist[num:num+batch_size]
        batch_lowdata=train_lowlist[num:num+batch_size]
        highdata=np.zeros([0,224,224,3])
        for m in range(batch_size):
            imga=scipy.misc.imread(batch_highdata[m], mode='RGB')
            imga=np.reshape(imga,[1,224,224,3])
            highdata=np.concatenate([highdata,imga],axis=0)
        highdata=2*(highdata/255-0.5)
        lowdata=np.zeros([0,56,56,3])
        for n in range(batch_size):
            imga=scipy.misc.imread(batch_lowdata[n], mode='RGB')
            imga=np.reshape(imga,[1,56,56,3])
            lowdata=np.concatenate([lowdata,imga],axis=0)
        lowdata=2*(lowdata/255-0.5)

        sess.run(cop,feed_dict={input_x:highdata,low:lowdata,global_step:i})
        if (i+1)%10000==0:
            im=sess.run(fake,feed_dict={low:test,global_step:i})
            plt.imsave('/cptjack/totem/kaixiangjin/sample/sample/20190722sample/{}.png'.format(i+1),np.uint8(255*(im[0]/2+0.5)))
            plt.imsave('/cptjack/totem/kaixiangjin/sample/sample/20190722sample/{}.png'.format(i+2),np.uint8(255*(im[1]/2+0.5)))
    for i in range(200000):
        num=random.randint(0,len(train_highlist)-batch_size)
        batch_highdata=train_highlist[num:num+batch_size]
        batch_lowdata=train_lowlist[num:num+batch_size]
        highdata=np.zeros([0,224,224,3])
        for m in range(batch_size):
            imga=scipy.misc.imread(batch_highdata[m], mode='RGB')
            imga=np.reshape(imga,[1,224,224,3])
            highdata=np.concatenate([highdata,imga],axis=0)
        highdata=2*(highdata/255-0.5)
        lowdata=np.zeros([0,56,56,3])
        for n in range(batch_size):
            imga=scipy.misc.imread(batch_lowdata[n], mode='RGB')
            imga=np.reshape(imga,[1,56,56,3])
            lowdata=np.concatenate([lowdata,imga],axis=0)
        lowdata=2*(lowdata/255-0.5)

        sess.run(gop,feed_dict={input_x:highdata,low:lowdata,global_step:i})
        sess.run(dop,feed_dict={input_x:highdata,low:lowdata,global_step:i})
#        sess.run(cop,feed_dict={input_x:highdata,low:lowdata})
        if (i+1)%10000==0:
            print(sess.run(d_loss,feed_dict={input_x:highdata,low:lowdata,global_step:i}))
            print(sess.run(g_loss,feed_dict={input_x:highdata,low:lowdata,global_step:i}))
            im=sess.run(fake,feed_dict={low:test,global_step:i})

            plt.imsave('/cptjack/totem/kaixiangjin/sample/sample/20190722sample/{}.png'.format(i+3),np.uint8(255*(im[0]/2+0.5)))
            plt.imsave('/cptjack/totem/kaixiangjin/sample/sample/20190722sample/{}.png'.format(i+4),np.uint8(255*(im[1]/2+0.5)))
        if (i+1)%10000==0:
            save_path=saver.save(sess,'Model/SE-ESRGAN/model.ckpt')

    
    


