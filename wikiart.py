import tensorflow as tf
import numpy as np
import pdb

def get_genre_data():
    data_path = 'genre-label.128.tfrecords'
    y_dim = 7
    x_dim = 128
    dataset = tf.data.TFRecordDataset(data_path)
#     dataset = dataset.shuffle(10000)
    dataset = dataset.map(lambda serialized_example,img_dim=x_dim: _parse_image_and_label(serialized_example,img_dim))
    dataset = dataset.map(lambda image,label,y_dim=y_dim: tuple(tf.py_func(_label_to_yvec,[image,label,y_dim],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label: tuple(tf.py_func(_transform,[image,label],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label,ish=[x_dim,x_dim,3],lsh=y_dim: _set_shapes(image,label,ish,lsh))
    return dataset

def get_landscape_data(img_dim=128):
    data_path = 'landscape-area.'+str(img_dim)+'.tfrecords'
    y_dim = 1
    x_dim = img_dim
    dataset = tf.data.TFRecordDataset(data_path)
#     dataset = dataset.shuffle(10000)
    dataset = dataset.map(lambda serialized_example,img_dim=x_dim: _parse_image_and_label(serialized_example,img_dim))
    dataset = dataset.map(lambda image,label: tuple(tf.py_func(_transform,[image,label],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label,ish=[x_dim,x_dim,3],lsh=y_dim: _set_shapes(image,label,ish,lsh))
    return dataset

def get_landscape_data_augmented(img_dim=128):
    data_path = 'landscape-area.'+str(img_dim)+'.tfrecords'
    y_dim = 1
    x_dim = img_dim
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(2)
    dataset = dataset.map(lambda serialized_example,img_dim=x_dim: _parse_image_and_label(serialized_example,img_dim))
    dataset = dataset.map(lambda image,label: _random_augmentation(image,label))
    dataset = dataset.map(lambda image,label: tuple(tf.py_func(_transform,[image,label],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label,ish=[x_dim,x_dim,3],lsh=y_dim: _set_shapes(image,label,ish,lsh))
    return dataset

def get_artist_data():
    data_path = 'artists.128.tfrecords'
    y_dim = 15
    x_dim = 128
    dataset = tf.data.TFRecordDataset(data_path)
#     dataset = dataset.shuffle(10000)
    dataset = dataset.map(lambda serialized_example,img_dim=x_dim: _parse_image_and_label(serialized_example,img_dim))
    dataset = dataset.filter(_filter)
    dataset = dataset.map(lambda image,label,y_dim=y_dim: tuple(tf.py_func(_label_to_yvec,[image,label,y_dim],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label: tuple(tf.py_func(_transform,[image,label],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label,ish=[x_dim,x_dim,3],lsh=y_dim: _set_shapes(image,label,ish,lsh))
    return dataset

def get_vangogh_data():
    data_path = 'vangogh.64.tfrecords'
    y_dim = 1
    x_dim = 64
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(lambda serialized_example,img_dim=x_dim: _parse_image_and_label(serialized_example,img_dim))
    dataset = dataset.map(lambda image,label: tuple(tf.py_func(_transform,[image,label],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label,ish=[x_dim,x_dim,3],lsh=y_dim: _set_shapes(image,label,ish,lsh))
    return dataset    

def get_celebA_data():
    data_path = 'celebA.108.tfrecords'
    x_dim = 108
    y_dim = 1
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(lambda serialized_example,img_dim=x_dim: _parse_image_and_label(serialized_example,img_dim))
    dataset = dataset.map(lambda image,label: tuple(tf.py_func(_transform,[image,label],[tf.float32,tf.float32])))
    dataset = dataset.map(lambda image,label,ish=[x_dim,x_dim,3],lsh=y_dim: _set_shapes(image,label,ish,lsh))
    return dataset

def _random_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image,label

def _transform(image,label):
  return np.array(image)/127.5 - 1., label

def _set_shapes(image, label, img_shape, lbl_shape):
    image.set_shape(img_shape)
    label.set_shape(lbl_shape)
    return image, label

def _parse_image_and_label(serialized_example,img_dim):
  feature = {'image': tf.FixedLenFeature([], tf.string),'label': tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(serialized_example, features=feature)
  image = tf.decode_raw(features['image'], tf.float32)
  label = tf.cast(features['label'], tf.float32)
  label = tf.reshape(label,[1])
  image = tf.reshape(image, [img_dim, img_dim, 3])
  return image, label

def _label_to_yvec(image,label,y_dim):
  yvec = np.zeros(shape=(y_dim),dtype=np.float32)
  yvec[int(label)] = 1.0
  return image,yvec

use_artists = ['Rembrandt','Pierre-Auguste Renoir','Pablo Picasso','Claude Monet','Salvador Dali','Utagawa Kuniyoshi','Vincent van Gogh','Edgar Degas','Ivan Shishkin','Ivan Aivazovsky','Paul Cezanne','Raphael Kirchner','Nicholas Roerich','Theophile Steinlen','Fernand Leger']
use_genres = ['abstract','cityscape','flower painting','landscape','nude painting','portrait','religious painting']

def label_to_artist(label):
    return use_artists[np.where(label == 1.0)[0][0]]

def _parse_image_and_labels(serialized_example):
  feature = {'image': tf.FixedLenFeature([], tf.string),
             'artist': tf.FixedLenFeature([], tf.string),
             'genre': tf.FixedLenFeature([], tf.string),
             'pixelsx': tf.FixedLenFeature([], tf.int64),
             'pixelsy': tf.FixedLenFeature([], tf.int64),
             'style': tf.FixedLenFeature([], tf.string)
            }
  features = tf.parse_single_example(serialized_example, features=feature)
  artist = tf.cast(features['artist'], tf.string)
  genre = tf.cast(features['genre'], tf.string)
  style = tf.cast(features['style'], tf.string)
  image = tf.decode_raw(features['image'], tf.float32)
  image = tf.reshape(image, [128, 128, 3])
  return image, artist, genre, style

def _artist_labels(image,artist,genre,style):
  label = -1 if artist not in use_artists else use_artists.index(artist)
  label = np.int32(label)
  return image,label

# def filter_artists(dataset):
#     return dataset.

def _filter(image,label):
  return tf.reshape(tf.less(label,5),[])

def get_data(data_path):
    dataset = tf.data.TFRecordDataset([data_path])
    dataset = dataset.map(parser)
    dataset = dataset.map(lambda image,artist,genre,style: 
                          tuple(tf.py_func(_artist_labels,[image,artist,genre,style],[tf.float32, tf.int32])))
    dataset = dataset.filter(_filter)
    return dataset


# class WikiartData():
#     def get_genre_data():
#         data_path = 'genre-label.128.tfrecords'
#         y_dim = 7
#         x_dim = 128
#         dataset = tf.data.TFRecordDataset(data_path)
#         dataset = dataset.map(lambda serialized_example,img_dim=x_dim: self._parse_image_and_label(serialized_example,img_dim))
#         dataset = dataset.map(lambda image,label,y_dim=y_dim: tuple(tf.py_func(self._label_to_yvec,[image,label,y_dim],[tf.float32,tf.float32])))
#         dataset = dataset.map(lambda image,label,ish=[x_dim,x_dim,3],lsh=y_dim: self._set_shapes(image,label,ish,lsh))
#         return dataset
    
#     def _set_shapes(image, label, img_shape, lbl_shape):
#         image.set_shape(img_shape)
#         label.set_shape(lbl_shape)
#         return image, label
        
#     def _parse_image_and_label(serialized_example,img_dim):
#       feature = {'image': tf.FixedLenFeature([], tf.string),'label': tf.FixedLenFeature([], tf.int64)}
#       features = tf.parse_single_example(serialized_example, features=feature)
#       image = tf.decode_raw(features['image'], tf.float32)
#       label = tf.cast(features['label'], tf.float32)
#       label = tf.reshape(label,[1])
#       image = tf.reshape(image, [img_dim, img_dim, 3])
#       return image, label

#     def _label_to_yvec(image,label,y_dim):
#       yvec = np.zeros(shape=(y_dim),dtype=np.float32)
#       yvec[int(label)] = 1.0
#       return image,yvec