import os
import scipy.misc
import numpy as np
import tensorflow as tf
from CarpeGAN import *
from CarpeUtils import *
import wikiart


# command line flags
flags = tf.app.flags
flags.DEFINE_string("data_path", "genre-label.128.tfrecords", "Name of tfrecords file")
flags.DEFINE_integer("epochs", 10, "Epochs to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("z_dim", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS


# main
def main(_):
#   pp.pprint(flags.FLAGS.__flags)

  for i in range(1):   
    g_lr = np.random.normal(loc=.0002,scale=1e-2)
    d_lr = np.random.normal(loc=.0002,scale=1e-2)
    beta1 = np.random.normal(loc=0.5,scale=.1)
  
    dataset = wikiart.get_genre_data()  
    run_dcgan(dataset,'genre-label',g_lr=g_lr,d_lr=d_lr,beta1=beta1)


    
def run_dcgan(dataset,dataset_name,z_dim=100,batch_size=64,epochs=5,g_lr=.0002,d_lr=.00004,beta1=0.5,xhparams=""):  
  tf.reset_default_graph()  
  
  with tf.Session() as sess:
    dcgan = DCGAN(
            sess,
            dataset=dataset,
            dataset_name=dataset_name,
            z_dim=z_dim,
            batch_size=batch_size,
            xhparams=xhparams)
    dcgan.train(epochs=epochs,g_learning_rate=g_lr,d_learning_rate=d_lr,beta1=beta1)

    
    
if __name__ == '__main__':
  tf.app.run()