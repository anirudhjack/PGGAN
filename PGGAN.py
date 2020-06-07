from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import adler.tensorflow as atf
import tensordata
from ops import lrelu, conv2d, fully_connect, upscale, Pixl_Norm, downscale2d, MinibatchstateConcat
from utils import save_images
import numpy as np
from scipy.ndimage.interpolation import zoom
from tensorflow.python import pywrap_tensorflow
#from IPython.core.debugger import Pdb
#pdb = Pdb()
import cPickle as pickle
import os
import urllib
import math
import numpy as np
import numpy.linalg
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import scipy.io
import skimage.transform
#from ops import *
from glob import glob
#from ops import *
from glob import glob
import numpy as np
import tarfile
import pickle
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensordata.augmentation import random_flip
import adler.tensorflow as atf
import sys
from scipy.optimize import linear_sum_assignment
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from scipy.spatial import distance
slim = tf.contrib.slim
import tensorflow as tf
import numpy as np
import tensordata
import functools
def generalized_gaussian_ratio(alpha):
    return (gamma(2.0/alpha)**2) / (gamma(1.0/alpha) * gamma(3.0/alpha))


"""
Generalized Gaussian ratio function inverse (numerical approximation)
Cite: Dominguez-Molina 2001, pg 13
"""
def generalized_gaussian_ratio_inverse(k):
    a1 = -0.535707356
    a2 = 1.168939911
    a3 = -0.1516189217
    b1 = 0.9694429
    b2 = 0.8727534
    b3 = 0.07350824
    c1 = 0.3655157
    c2 = 0.6723532
    c3 = 0.033834

    if k < 0.131246:
        return 2 * math.log(27.0/16.0) / math.log(3.0/(4*k**2))
    elif k < 0.448994:
        return (1/(2 * a1)) * (-a2 + math.sqrt(a2**2 - 4*a1*a3 + 4*a1*k))
    elif k < 0.671256:
        return (1/(2*b3*k)) * (b1 - b2*k - math.sqrt((b1 - b2*k)**2 - 4*b3*(k**2)))
    elif k < 0.75:
        third_term = (c2**2 + 4*c3*math.log((3-4*k)/(4*c1)))
        #if(third_term < 0.0):
            #third = (-1)*third_term         
            #print "%f %f %f" % (k, ((3-4*k)/(4*c1)), c2**2 + 4*c3*math.log((3-4*k)/(4*c1)) )
            #return (1/(2*c3)) * (c2 - math.sqrt(third))
        #else:
        return (1/(2*c3)) * (c2 - math.sqrt(third_term))
    else:
        print("warning: GGRF inverse of %f is not defined" %(k))
        return numpy.nan

"""
Estimate the parameters of an asymmetric generalized Gaussian distribution
"""
def estimate_aggd_params(x):
    x_left = x[x < 0]
    x_right = x[x >= 0]
    stddev_left = math.sqrt((1.0/(x_left.size - 1)) * numpy.sum(x_left ** 2))
    stddev_right = math.sqrt((1.0/(x_right.size - 1)) * numpy.sum(x_right ** 2))
    if stddev_right == 0:
        return 1, 0, 0 # TODO check this
    r_hat = numpy.mean(numpy.abs(x))**2 / numpy.mean(x**2)
    y_hat = stddev_left / stddev_right
    R_hat = r_hat * (y_hat**3 + 1) * (y_hat + 1) / ((y_hat**2 + 1) ** 2)
    gam = np.arange(0.2,10,0.001)
    rgam = (gamma(2.0/gam)**2) / (gamma(1.0/gam) * gamma(3.0/gam))
    intermediate_array = (rgam -R_hat)**2
    array_position = numpy.argmin(intermediate_array)
    #print(type(array_position))
    #print(array_position)
    #print(type(gam))
    #pdb.set_trace()
    alpha = gam[array_position]
    #print(alpha)
    #pdb.set_trace()
    beta_left = stddev_left * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
    beta_right = stddev_right * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
    return alpha, beta_left, beta_right

def compute_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)

    features.extend([ alpha, (beta_left+beta_right)/2 ])

    for x_shift, y_shift in ((0,1), (1,0), (1,1), (1,-1)):
        img_pair_products  = img_norm * numpy.roll(numpy.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0/alpha) / gamma(1.0/alpha))
        features.extend([ alpha, eta, beta_left, beta_right ])

    return features

def normalize_image(img, sigma=7/6):
    mu  = gaussian_filter(img, sigma, mode='nearest')
    mu_sq = mu * mu
    sigma = numpy.sqrt(numpy.abs(gaussian_filter(img * img, sigma, mode='nearest') - mu_sq))
    img_norm = (img - mu) / (sigma + 1)
    return img_norm


def niqe_normal(img):
    model_mat = scipy.io.loadmat('modelparameters_normal_8.mat')
    model_mu = model_mat['mu_distparam_normal_8']
    model_cov = model_mat['cov_distparam_normal_8']

    features = None
    img_scaled = img
    for scale in [1,2]:

        if scale != 1:
            img_scaled = skimage.transform.rescale(img, 1/scale)
            #img_scaled = scipy.misc.imresize(img_norm, 0.5)

        # print img_scaled
        img_norm = normalize_image(img_scaled)

        scale_features = []
        block_size = 8//scale
        print(img.shape[1])
        for block_col in range(img_norm.shape[0]//block_size):
            for block_row in range(img_norm.shape[1]//block_size):
                block_features = compute_features( img_norm[block_col*block_size:(block_col+1)*block_size, block_row*block_size:(block_row+1)*block_size] )
                scale_features.append(block_features)
                #print(len(block_features))
        # print "len(scale_features)=%f" %(len(scale_features))
        if np.all(features) == None:
            features = numpy.vstack(scale_features)
            # print features.shape
        else:
            features = numpy.hstack([features, numpy.vstack(scale_features)])
            # print features.shape
        
    features_mu = numpy.nanmean(features, axis=0)
    maskedarr = numpy.ma.array(features)

    features_cov = numpy.ma.cov((maskedarr.astype(float)).T)

    pseudoinv_of_avg_cov  = numpy.linalg.pinv((model_cov + features_cov)/2)
    niqe_quality = math.sqrt( (model_mu - features_mu).dot( pseudoinv_of_avg_cov.dot( (model_mu - features_mu).T ) ) )

    return np.float32(niqe_quality)

def niqe_grad(img):
    model_mat = scipy.io.loadmat('modelparameters_grad_16.mat')
    model_mu = model_mat['mu_distparam_grad_16']
    model_cov = model_mat['cov_distparam_grad_16']

    features = None
    img_scaled = img
    for scale in [1,2]:

        if scale != 1:
            img_scaled = skimage.transform.rescale(img, 1/scale)
            #img_scaled = scipy.misc.imresize(img_norm, 0.5)

        # print img_scaled
        img_norm = normalize_image(img_scaled)

        scale_features = []
        block_size = 16//scale
	print(img.shape)
        for block_col in range(img_norm.shape[0]//block_size):
            for block_row in range(img_norm.shape[1]//block_size):
                block_features = compute_features( img_norm[block_col*block_size:(block_col+1)*block_size, block_row*block_size:(block_row+1)*block_size] )
                scale_features.append(block_features)
                #print(len(block_features))
        # print "len(scale_features)=%f" %(len(scale_features))
        if np.all(features) == None:
            features = numpy.vstack(scale_features)
            # print features.shape
        else:
            features = numpy.hstack([features, numpy.vstack(scale_features)])
            # print features.shape
        
    features_mu = numpy.nanmean(features, axis=0)
    maskedarr = numpy.ma.array(features)

    features_cov = numpy.ma.cov((maskedarr.astype(float)).T)

    pseudoinv_of_avg_cov  = numpy.linalg.pinv((model_cov + features_cov)/2)
    niqe_quality = math.sqrt( (model_mu - features_mu).dot( pseudoinv_of_avg_cov.dot( (model_mu - features_mu).T ) ) )

    return np.float32(niqe_quality)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)
def tf_ssim_modified(img1, img2, cs_map=False, mean_metric=True, size=4, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
    #if cs_map:
       # value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
         #           (sigma1_sq + sigma2_sq + C2)),
         #       (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
   # else:
    S1 = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    S2 = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    value = tf.reduce_mean(tf.sqrt(2-S1-S2),[1,2,3])
        #value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                 #   (sigma1_sq + sigma2_sq + C2))

    #if mean_metric:
        #value = tf.reduce_mean(value)
    		
    return value
        
class PGGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, read_model_path, data, sample_size, sample_path,  sample_path2,log_dir,
                 learn_rate, lam_gp, lam_eps, PG, t, use_wscale, is_celeba):
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gan_model_path = model_path
        self.read_model_path = read_model_path
        self.data_In = data
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.sample_path2 = sample_path2
        self.log_dir = log_dir
        self.learning_rate = learn_rate
        self.lam_gp = lam_gp
        self.lam_eps = lam_eps
        self.pg = PG
        self.trans = t
        self.log_vars = []
        self.channel = self.data_In.channel
        self.output_size = 4 * pow(2, PG - 1)
        self.use_wscale = use_wscale
        self.is_celeba = is_celeba
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False,name='alpha_tra')

    def build_model_PGGan(self):
        self.fake_images = self.generate(self.z,reuse=False, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.D_pro_logits = self.discriminate(self.images, reuse=False, pg = self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True,pg= self.pg, t=self.trans, alpha_trans=self.alpha_tra)

        # the defination of loss for D and G
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # gradient penalty from WGAN-GP
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits= self.discriminate(interpolates, reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        gradients = tf.gradients(discri_logits, [interpolates])[0]
        print(gradients)
        print(self.output_size)
        #pdb.set_trace()
        # 2 norm
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)
        #atf.image_grid_summary('x_true', self.images )
        #atf.image_grid_summary('x_generated',self.fake_images )
        self.D_origin_loss = self.D_loss
        self.D_loss +=  self.gradient_penalty
        self.D_loss += self.lam_eps * tf.reduce_mean(tf.square(self.D_pro_logits))
        res = self.output_size
        #if (res >= 32):
        #    niqe_score_grad = tf.py_func(niqe_grad,[gradients], tf.float32)
        #    niqe_score_mean_grad = tf.reduce_mean(niqe_score_grad)
        #    self.D_loss += niqe_score_mean_grad
        #d_regularizer_mean = tf.reduce_mean(tf.square(d_true))
        #d_regularizer_ssim_total  = d_regularizer_ssim 
        #d_regularizer_ssim_total  = d_regularizer_ssim 
        #added_reg = tf.placeholder(tf.float32, shape=None)
        #added_regularizer = 0.1*niqe_score_mean_grad + d_regularizer1
        #SSIM_modified= tf_ssim_modified(tf.image.rgb_to_grayscale(self.images),tf.image.rgb_to_grayscale(self.fake_images))
        #SSIM_x = tf.reduce_mean(SSIM_modified,[1,2,3])
        _, self.D_pro_logits2 = self.discriminate(self.images, reuse=True, pg = self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.G_pro_logits2 = self.discriminate(self.fake_images, reuse=True,pg= self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        #difference = tf.abs(self.D_pro_logits2 -self.G_pro_logits2)
        #SSIM_modified2 = tf.where(tf.is_nan(SSIM_modified), 1.414*tf.ones_like(SSIM_modified), SSIM_modified)
        #summary_ssim2 = tf.reduce_mean(SSIM_modified2)
        #print(difference.get_shape())
        #pdb.set_trace()
        #ddx_2 = tf.divide(difference[:,0],SSIM_modified2)
        #self.d_regularizer2 = 0.1*tf.reduce_mean(tf.square(ddx_2)-1)
        
        #tf.summary.scalar("d_regularizer",self.d_regularizer2)
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]

        total_para = 0
        for variable in self.d_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        print ("The total para of D", total_para)

        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        total_para2 = 0
        for variable in self.g_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para2 += variable_para
        print ("The total para of G", total_para2)

        #save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.output_size) not in var.name]

        print ("d_vars", len(self.d_vars))
        print ("g_vars", len(self.g_vars))

        print ("self.d_vars_n_read", len(self.d_vars_n_read))
        print ("self.g_vars_n_read", len(self.g_vars_n_read))

        print ("d_vars_n_2_rgb", len(self.d_vars_n_2_rgb))
        print ("g_vars_n_2_rgb", len(self.g_vars_n_2_rgb))

        # for n in self.d_vars:
        #     print (n.name)

        self.g_d_w = [var for var in self.d_vars + self.g_vars if 'bias' not in var.name]

        print ("self.g_d_w", len(self.g_d_w))

        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)
        self.saver_mine = tf.train.Saver(self.d_vars + self.g_vars,max_to_keep=40)
        #self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)
        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)
        
        

    def generate_and_classify(self,z_2):
        INCEPTION_OUTPUT = 'logits:0'
        xx = self.generate(z_2,reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        #pdb.set_trace()
        #x = generator(z_2, reuse=True)
        xx = tf.image.resize_bilinear(xx, [299, 299])
        return tf.contrib.gan.eval.run_inception(xx, output_tensor=INCEPTION_OUTPUT)
    # do train
    def train(self):
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #inception_z = tf.constant(np.random.randn(12800, 512), dtype='float32')
        #inception_score = tf.contrib.gan.eval.classifier_score(inception_z,classifier_fn=self.generate_and_classify,num_batches=200)
        #fid_score = tf.contrib.gan.eval.frechet_classifier_distance(self.images,self.fake_images,classifier_fn=self.generate_and_classify,num_batches=1)
        #tf.summary.scalar("inception_score",inception_score)
        #tf.summary.scalar("fid_inception_score",inception_score)
        with tf.Session(config=config) as sess:
            sess.run(init)
            summary_op = tf.summary.merge_all()
            other=self.log_dir+"/"+str(self.pg)
            summary_writer = tf.summary.FileWriter(other, sess.graph)
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            #print(self.read_model_path)
            #print(self.trans)
            #pdb.set_trace()
            if self.pg != 1 and self.pg != 8:
                if self.trans:
                    variables_read = self.d_vars_n_read + self.g_vars_n_read
                    variables_rgb_read = self.d_vars_n_2_rgb + self.g_vars_n_2_rgb
                    #print(variables_read)
                    #print(variables_rgb_read)
                    #pdb.set_trace()
                    #restored_vars  = self.get_tensors_in_checkpoint_file(file_name=self.read_model_path)
                    #tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                    #loader = tf.train.Saver(tensors_to_load)
                    #loader.restore(sess, self.read_model_path)
                    name =  './output/PGGAN_without_regularizer/model_pggan_0/6/'                    
                    self.r_saver.restore(sess, name)
                    self.rgb_saver.restore(sess, name)
                    #self.saver_mine.restore(sess, name)
                    reader = pywrap_tensorflow.NewCheckpointReader(name)                    
                    var_to_shape_map = reader.get_variable_to_shape_map()
                    var_keys = var_to_shape_map.keys()
                    var_names = []
                    for j in variables:
                        s = j.name
                        other = str(s)
                        foo = other[:-2]
                        var_names.append(foo)
                    #var_list = checkpoint_utils.list_variables(self.read_model_path)
                    all_list =[]
                    #print(var_names)
                    #print(var_keys)
                    #pdb.set_trace()
                    for k in variables:
                        s = k.name
                        other = str(s)
                        foo = other[:-2]
                        
                        if foo in var_keys:
                    #        print("y")
                            all_list.append(k)   
                        #ss= tf.variable(key)
                    #print(len(all_list))
                    #self.saver.restore(sess, self.read_model_path)
                    saver = tf.train.Saver(all_list)
                    saver.restore(sess, name)

                else:
                    self.saver.restore(sess, self.read_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iters:
                # optimization D
                n_critic = 1
                if self.pg >= 5:
                    n_critic = 1

                for i in range(n_critic):
                    sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
                    if self.is_celeba:
                        train_list = self.data_In.getNextBatch(batch_num, self.batch_size)
                        realbatch_array = self.data_In.getShapeForData(train_list, resize_w=self.output_size)
                    else:
                        realbatch_array = self.data_In.getNextBatch(self.batch_size, resize_w=self.output_size)
                        realbatch_array = np.transpose(realbatch_array, axes=[0, 3, 2, 1]).transpose([0, 2, 1, 3])

                    if self.trans and self.pg != 0:
                        alpha = np.float(step) / self.max_iters
                        low_realbatch_array = zoom(realbatch_array, zoom=[1, 0.5, 0.5, 1], mode='nearest')
                        low_realbatch_array = zoom(low_realbatch_array, zoom=[1, 2, 2, 1], mode='nearest')
                        realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.z: sample_z})
                    batch_num += 1

                # optimization G
                sess.run(opti_G, feed_dict={self.z: sample_z})
                #################### Fake images
                #inception_z = np.random.normal(size=[self.batch_size, self.sample_size])
                
                #inception_summary = tf.summary.merge([tf.summary.scalar('inception_score', inception_score)])
                #full_summary = tf.summary.merge([summary_op,inception_score ])
                #tf.summary.scalar("inception_score",inception_score)
                
                # the alpha of fake_in process
                sess.run(alpha_tra_assign, feed_dict={step_pl: step})

                if step % 10 == 0:
                    summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.z: sample_z})
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.add_summary(summary_str, step)
                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss,self.alpha_tra], feed_dict={self.images: realbatch_array, self.z: sample_z})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    print(self.batch_size/2)
                    print(self.batch_size)
                    save_images(realbatch_array[0:self.batch_size], [2, int(self.batch_size/2)],
                                '{}/{:02d}_real.jpg'.format(self.sample_path, step))

                    if self.trans and self.pg != 0:
                        low_realbatch_array = np.clip(low_realbatch_array, -1, 1)
                        save_images(low_realbatch_array[0:self.batch_size], [2, int(self.batch_size / 2)],
                                    '{}/{:02d}_real_lower.jpg'.format(self.sample_path, step))
                   
                    fake_image = sess.run(self.fake_images,
                                          feed_dict={self.images: realbatch_array, self.z: sample_z})
                                          
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, int(self.batch_size/2)], '{}/{:02d}_train.jpg'.format(self.sample_path, step))
                    #######################
                    for ii in range(self.batch_size): 
                        dd=fake_image[ii,:,:,:]
                        #fn = "{:0>6d}.png".format(global_step)
                        fn2='{}/{:02d}_{}_train_individual.jpg'.format(self.sample_path2, step,ii)
                        scipy.misc.imsave(os.path.join(fn2), dd) 
                if np.mod(step, 1000) == 0 and step != 0:
                    self.saver.save(sess, self.gan_model_path)
                    gan_model2_path_mine = self.gan_model_path + 'PG_GAN_NIQE' +str(step)
                    save_path2 = self.saver_mine.save(sess,gan_model2_path_mine)

                step += 1

            save_path = self.saver.save(sess, self.gan_model_path)
            print ("Model saved in file: %s" % save_path)

        tf.reset_default_graph()

    def discriminate(self, conv, reuse=False, pg=1, t=False, alpha_trans=0.01):
        #dis_as_v = []
        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [self.batch_size, -1])

            #for D
            output = fully_connect(conv, output_size=1, use_wscale=self.use_wscale, gain=1, name='dis_n_fully')

            return tf.nn.sigmoid(output), output

    def generate(self, z_var, reuse=False,pg=1, t=False, alpha_trans=0.0):
        with tf.variable_scope('generator') as scope:
            if reuse == True:
                scope.reuse_variables()

            de = tf.reshape(Pixl_Norm(z_var), [self.batch_size, 1, 1, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [self.batch_size, 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))))
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))))

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            if pg == 1: return de
            if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            else: de = de

            return de

    def get_nf(self, stage):
        return min(1024 / (2 **(stage * 1)), 512)
    
    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps
    def test(self):
    #step_pl = tf.placeholder(tf.float32, shape=None)
    #alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

    #opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
    #    self.D_loss, var_list=self.d_vars)
    #opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
    #    self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #inception_z = tf.constant(np.random.randn(12800, 512), dtype='float32')
        #inception_score = tf.contrib.gan.eval.classifier_score(inception_z,classifier_fn=self.generate_and_classify,num_batches=200)
        #fid_score = tf.contrib.gan.eval.frechet_classifier_distance(self.images,self.fake_images,classifier_fn=self.generate_and_classify,num_batches=1)
        #tf.summary.scalar("inception_score",inception_score)
        #tf.summary.scalar("fid_inception_score",inception_score)
        with tf.Session(config=config) as sess:
            sess.run(init)
            print(self.read_model_path)
            name =  './output/PGGAN_without_regularizer/model_pggan_0/7/'
            self.saver_mine.restore(sess, name)
            self.r_saver.restore(sess, name)
            #self.rgb_saver.restore(sess, name)
            print(self.read_model_path)
            #pdb.set_trace()
            for xx in range(1,500,1):
                sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
                if self.is_celeba:
                    train_list = self.data_In.getNextBatch(xx, self.batch_size)
                    realbatch_array = self.data_In.getShapeForData(train_list, resize_w=self.output_size)                    
                realbatch_array = np.clip(realbatch_array, -1, 1)
                fake_image = sess.run(self.fake_images,feed_dict={self.images: realbatch_array, self.z: sample_z})
                #fake_image = np.clip(fake_image, -1, 1)
                #save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.jpg'.format(self.sample_path, xx+100))
                fake_image = np.clip(fake_image, -1, 1)
                fake_image = ((fake_image + 1.)* 127.5).astype(np.uint8)
                #print(fake_image.shape)
                #pdb.set_trace()
                dirname = '/home/parimala/Desktop/progressive_growing_of_gans_tensorflow-master_NIQE_Based_approach/All_generated_samples_256_7000/'
                for i in range(64): 
                    dd=fake_image[i,:,:,:]
                    #fn = "{:0>6d}.png".format(global_step)
                    fn2=dirname+'sample'+str(i)+'_'+str(xx)+'_'+'.jpg';
                    scipy.misc.imsave(os.path.join(fn2), dd) 
            #save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.jpg'.format(self.sample_path, step))
                #if np.mod(step, 1000) == 0 and step != 0:
                #    self.saver.save(sess, self.gan_model_path)

                #step += 1

            #save_path = self.saver.save(sess, self.gan_model_path)
            #print ("Model saved in file: %s" % save_path)

        #tf.reset_default_graph()

















