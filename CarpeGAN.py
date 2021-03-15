from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pdb
import matplotlib.pyplot as plt

from ops import *
from CarpeUtils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, dataset, dataset_name,
          use_labels=True,z_dim=100,batch_size=64,sample_num=64,
          gf_dim=64, df_dim=64,gfc_dim=1024, dfc_dim=1024,
          xhparams=""):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    
    self.dataset_name = dataset_name
    self.data = dataset
    self.use_labels = use_labels
    self.xhparams = xhparams

    self.input_height, self.input_width, self.c_dim = dataset.output_shapes[0].as_list()
    self.output_height,self.output_width = self.input_height, self.input_width
    self.grayscale = (self.c_dim == 1)
    
    self.y_dim = None if use_labels==False else dataset.output_shapes[1].as_list()[0]

    self.z_dim = z_dim
    self.batch_size = batch_size
    self.sample_num = batch_size
#     self.sample_num = max([x*x for x in range(4,10) if x*x <= self.batch_size])

    self.gf_dim = gf_dim
    self.df_dim = df_dim
    
    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')
    
    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None
    
    image_dims = [self.output_height, self.output_width, self.c_dim]
    self.inputs = tf.placeholder(tf.float32, [None] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
#     self.g_loss = tf.reduce_mean(
#       sigmoid_cross_entropy_with_logits(self.D_logits, tf.zeros_like(self.D)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, epochs, d_learning_rate, g_learning_rate, beta1, resume_checkpoint=None):
    d_optim = tf.train.AdamOptimizer(d_learning_rate, beta1=beta1).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(g_learning_rate, beta1=beta1).minimize(self.g_loss, var_list=self.g_vars)
    
    tf.global_variables_initializer().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    
    self.hparam_str = make_hparam_string(learning_rate=g_learning_rate,beta1=beta1,image_size=self.input_height,data=self.dataset_name.split('.')[0],z_dim=self.z_dim,batch_size=self.batch_size,labels=self.use_labels,other=self.xhparams)
    self.checkpoint_dir = os.path.join('checkpoint',self.hparam_str)
    self.sample_dir = os.path.join('samples',self.hparam_str)
    self.log_dir = os.path.join('logs',self.hparam_str)
    self.writer = SummaryWriter(self.log_dir, self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir) if not resume_checkpoint else self.load(resume_checkpoint)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    early_stop = False
    for epoch in xrange(epochs):
      if early_stop: break
        
      train_data = self.data.shuffle(10000)
      train_data = train_data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
      iterator = train_data.make_one_shot_iterator()
      batch_images_iter, batch_labels_iter = iterator.get_next()
      idx = -1
       
      while True:
        try:
          batch_images = batch_images_iter.eval()
          batch_labels = batch_labels_iter.eval()

          idx+=1
          batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
  
          # Update D network
          if self.y_dim:
            _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict={ self.inputs: batch_images, self.z: batch_z, self.y: batch_labels })
          else:
            _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
  
          # Update G network
          if self.y_dim:
            _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z, self.y: batch_labels })
          else:
            _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
  
          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          if self.y_dim:
            _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z, self.y: batch_labels })
          else:
            _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z })
            self.writer.add_summary(summary_str, counter)
          
          if self.y_dim:
            errD_fake = self.d_loss_fake.eval({ self.z: batch_z, self.y: batch_labels })
            errD_real = self.d_loss_real.eval({ self.inputs: batch_images, self.y: batch_labels })
            errG = self.g_loss.eval({self.z: batch_z, self.y: batch_labels})
          else:
            errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
            errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
            errG = self.g_loss.eval({self.z: batch_z })            
          
          counter += 1
          idx = counter
#           batch_idxs = int(math.ceil(config.train_size/self.batch_size))
          batch_idxs = 0
          print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, epochs, idx, batch_idxs,
              time.time() - start_time, errD_fake+errD_real, errG))
       
          if (errD_fake+errD_real) < .000001 or errG == 0.0:
            print('Discriminator loss -> 0 detected, stopping early...')
            early_stopping = True
            break
  
          if np.mod(counter, 100) == 1:
            try:
              if self.y_dim:
                samples, d_loss, g_loss = self.sess.run(
                  [self.sampler, self.d_loss, self.g_loss],
                  feed_dict={
                      self.z: batch_z,
                      self.inputs: batch_images,
                      self.y: batch_labels
                  },
                )
              else:
                samples, d_loss, g_loss = self.sess.run(
                  [self.sampler, self.d_loss, self.g_loss],
                  feed_dict={
                      self.z: batch_z,
                      self.inputs: batch_images
                  },
                )
              if not os.path.isdir(self.sample_dir): os.makedirs(self.sample_dir)
              results = save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except Exception as e:
              print("one pic error!...")
              print(e)
              raise(e)
  
          if np.mod(counter, 500) == 2:
            self.save(self.checkpoint_dir, counter)
        except:
            break

        
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      shape = [self.batch_size]+image.get_shape().as_list()[1:]
      image.set_shape(shape)
    
      if True:
        if self.y_dim:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        if self.y_dim: h0 = conv_cond_concat(h0,yb)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        if self.y_dim: h1 = conv_cond_concat(h1,yb)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        if self.y_dim: h2 = conv_cond_concat(h2,yb)
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        if self.y_dim: h3 = conv_cond_concat(h3,yb)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if True:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        
        if self.y_dim:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [self.batch_size, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))
        if self.y_dim: h0 = conv_cond_concat(h0,yb)

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))
        if self.y_dim: h1 = conv_cond_concat(h1,yb)

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))
        if self.y_dim: h2 = conv_cond_concat(h2,yb)            

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))
        if self.y_dim: h3 = conv_cond_concat(h3,yb)            

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
        return tf.nn.tanh(h4)

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if True:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        if self.y_dim:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
            
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))
        if self.y_dim: h0 = conv_cond_concat(h0,yb)

        h1 = deconv2d(h0, [self.sample_num, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))
        if self.y_dim: h1 = conv_cond_concat(h1,yb)

        h2 = deconv2d(h1, [self.sample_num, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))
        if self.y_dim: h2 = conv_cond_concat(h2,yb)            

        h3 = deconv2d(h2, [self.sample_num, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))
        if self.y_dim: h3 = conv_cond_concat(h3,yb)            

        h4 = deconv2d(h3, [self.sample_num, s_h, s_w, self.c_dim], name='g_h4')
        return tf.nn.tanh(h4)
    

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0