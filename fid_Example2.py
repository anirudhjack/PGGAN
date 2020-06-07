import tensorflow as tf
import os
import glob
from scipy.misc import imresize
from scipy.misc import imread
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
datapath1 = '/home/parimala/Aa_current/Current_GAN_work/SSIM_Expicit_regularization/Models_ssim_regularization_cifar/niqe_91k_4'
datapath2 = './train'
files = glob.glob(os.path.join(datapath1, '*.jpg'))
#files2 = files[:20000]
images1 = np.array([imread(str(fn),mode='RGB').astype(np.float32) for fn in files])
#images22 = images1[0:3000,:,:,:]
images222 = tf.convert_to_tensor(images1)
print("yes")
files = glob.glob(os.path.join(datapath2, '*.png'))
#files3 = files[:20000]
#images2 = np.array([imresize(imread(str(fn),mode='RGB'),(48,48)).astype(np.float32) for fn in files])
images2 = np.array([imread(str(fn),mode='RGB').astype(np.float32) for fn in files])
#images22_2 = images2[0:3000,:,:,:]
images222_2 = tf.convert_to_tensor(images2)
print("yes")
def frechet_process(x):
    INCEPTION_FINAL_POOL = 'pool_3:0'
    #x = generator(z, reuse=True)
    x = tf.image.resize_bilinear(x, [299, 299])
    return tf.contrib.gan.eval.run_inception(x,output_tensor=INCEPTION_FINAL_POOL)
                                                          

#print(type(images1))
fid = tf.contrib.gan.eval.frechet_classifier_distance(images222,images222_2,classifier_fn =frechet_process,num_batches=200)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(fid))
    
    
