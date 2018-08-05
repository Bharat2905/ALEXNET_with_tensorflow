import numpy as np
import os
import tensorflow as tf
from skimage import io
from skimage.transform import resize
from keras.utils import to_categorical
from tqdm import tqdm
class create_tfrecord(object):
 	def __init__(self,image_directory,number_cate,save_directory):
 		self.image_size = 227
 		self.image_directory=image_directory
 		self.save_directory=save_directory
 		self.number_cate=number_cate


 	def load_data(self):
 		with tf.python_io.TFRecordWriter(self.save_directory) as writer:
 			for idx,cate in enumerate(os.listdir(self.image_directory)):
 				for i in tqdm(os.listdir(os.path.join(self.image_directory,cate))):
 					path = os.path.join(self.image_directory,cate,i)
 					img=resize(io.imread(path),(self.image_size,self.image_size,3))
 					label=to_categorical(idx,self.number_cate)
 					image_raw=img.tostring()
 					label_raw=label.tostring()
 					example=tf.train.Example(features=tf.train.Features(feature={
 						'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
 						'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
 						}))
 					writer.write(example.SerializeToString())
 		print('saved the tfrecord file')

if __name__=='__main__':
	file=create_tfrecord(path_to_data_folder,number_of_categories,path_to_save_record_file)
	file.load_data()
