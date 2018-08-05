import numpy as np
import tensorflow as tf
from skimage.transform import resize
from skimage import io
import sys
import os
import time
class infernece(object):
	def __init__(self,image_directory,checkpoint_directory,metafile_name,indexfile_name):
		self.image_directory=image_directory
		self.checkpoint_directory=checkpoint_directory
		self.metafile_name=metafile_name
		self.indexfile_name= indexfile_name

	def load_model(self):
		self.metafile=self.checkpoint_directory+self.metafile_name
		self.indexfile=self.checkpoint_directory+self.indexfile_name
		self.sess = tf.Session()
		saver = tf.train.import_meta_graph(self.metafile)
		saver.restore(self.sess,self.indexfile)
		self.graph = tf.get_default_graph()
		self.image = self.graph.get_tensor_by_name("image:0")
		self.y = self.graph.get_tensor_by_name("y:0")


	def infer(self):
		self.input=np.array(resize(io.imread(self.image_directory),(227,227,3))).reshape(1,227,227,3)
		self.ouput=self.sess.run([self.y],feed_dict={self.image:self.input})
		self.sess.close()
		print(np.argmax(self.ouput))


if __name__=='__main__':
	inf=infernece(image_directory,checkpoint_directory,metafile_name,indexfile_name)
	inf.load_model()
	t1=time.time()
	inf.infer()
	t2=time.time()
	print(t2-t1)
