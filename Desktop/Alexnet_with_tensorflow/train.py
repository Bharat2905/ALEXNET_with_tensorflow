import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class train(object):
	def __init__(self,record_path,no_classes,batch_size,num_iterations,checkpoint_dir):
                self.no_classes=no_classes
		self.record_path=record_path
		self.batch_size=batch_size
		self.num_iterations = num_iterations
		self.checkpoint_dir = checkpoint_dir


	


	def read_data(self):
		def _parse_function(example_proto):
			features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),"label": tf.FixedLenFeature((), tf.string, default_value="")}
			parsed_features = tf.parse_single_example(example_proto,features)
			return parsed_features["image"], parsed_features["label"]
		dataset = tf.data.TFRecordDataset(self.record_path)
		dataset =  dataset.map(_parse_function)
		dataset = dataset.shuffle(buffer_size=100)
		dataset = dataset.batch(self.batch_size)
		dataset = dataset.repeat()
		print('data loaded')
		return dataset


	def read_example(self):
		dataset=self.read_data()
		iterator = dataset.make_initializable_iterator()
		with tf.Session() as sess:
			sess.run(iterator.initializer)
			for i in range(1):
				X, Y = iterator.get_next()
				x,y=sess.run([X,Y])
				image_raw=tf.decode_raw(x, tf.float64)
				image = tf.cast(tf.reshape(image_raw, [self.batch_size, 227, 227, 3]), tf.float32)
				image_np=sess.run([image])
		image1=np.array(image_np)
		image_o=image1.reshape(self.batch_size,227,227,3)
		plt.imshow(image_o[8])
		plt.show()
		print(image_o[0].shape)
	def load_model(self):
		self.w_conv1=tf.Variable(tf.truncated_normal([11,11,3,96]),'w_conv1')
		self.b_conv1=tf.Variable(tf.truncated_normal([96]),'b_conv1')
		self.h_conv1=tf.nn.relu(tf.nn.conv2d(self.image,self.w_conv1, strides=[1, 4, 4, 1], padding='VALID')+self.b_conv1,'h_conv1')
		self.h_pool1=tf.nn.max_pool(self.h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID',name='h_pool1')
		self.w_conv2=tf.Variable(tf.truncated_normal([5,5,96,256]),'w_conv2')
		self.b_conv2=tf.Variable(tf.truncated_normal([256]),'b_conv2')
		self.h_conv2=tf.nn.relu(tf.nn.conv2d(self.h_pool1,self.w_conv2, strides=[1, 1, 1, 1], padding='SAME')+self.b_conv2,'h_conv2')
		self.h_pool2=tf.nn.max_pool(self.h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID',name='h_pool2')
		self.w_conv3=tf.Variable(tf.truncated_normal([3,3,256,384]),'w_conv3')
		self.b_conv3=tf.Variable(tf.truncated_normal([384]),'b_conv3')
		self.h_conv3=tf.nn.relu(tf.nn.conv2d(self.h_pool2,self.w_conv3, strides=[1, 1, 1, 1], padding='SAME')+self.b_conv3,'h_conv3')
		self.w_conv4=tf.Variable(tf.truncated_normal([3,3,384,256]),'w_conv4')
		self.b_conv4=tf.Variable(tf.truncated_normal([256]),'b_conv4')
		self.h_conv4=tf.nn.relu(tf.nn.conv2d(self.h_conv3,self.w_conv4, strides=[1, 1, 1, 1], padding='SAME')+self.b_conv4,'h_conv4')
		self.h_pool4=tf.nn.max_pool(self.h_conv4, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID',name='h_pool4')
		self.W_fc1 = tf.Variable(tf.truncated_normal([9216,4096]),'W_fc1')
		self.b_fc1 = tf.Variable(tf.truncated_normal([4096]),'b_fc1')
		self.h_pool2_flat = tf.reshape(self.h_pool4, [-1, 9216],'h_pool2_flat')
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat,self.W_fc1) + self.b_fc1,'h_fc1')
		self.W_fc2 = tf.Variable(tf.truncated_normal([4096,4096]),'W_fc2')
		self.b_fc2 = tf.Variable(tf.truncated_normal([4096]),'b_fc2')
		self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1,self.W_fc2) + self.b_fc2,'h_fc2')
		self.W_fc3 = tf.Variable(tf.truncated_normal([4096,self.no_classes]),'W_fc3')
		self.b_fc3 = tf.Variable(tf.truncated_normal([self.no_classes]),'b_fc3')
		self.y = tf.add(tf.matmul(self.h_fc2, self.W_fc3),self.b_fc3,'y')
		print('prediction done')
		print(self.y.shape,self.label.shape,self.image.shape)

	def loss(self):
		self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.y))
		self.optimize = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)

	def accuracy(self):
		self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.label, 1))
		self.accu = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



	def train(self):
		self.image=tf.placeholder(tf.float32, shape=[None,227,227,3],name='image')
		self.label=tf.placeholder(tf.float32, shape=[None, self.no_classes])
		#self.image=tf.Variable(tf.truncated_normal([self.batch_size,227,227,3]),trainable=False,name='image')
		#self.label=tf.Variable(tf.truncated_normal([self.batch_size, 102]),trainable=False)
		self.load_model()
		self.loss()
		self.accuracy()
		dataset=self.read_data()
		iterator=dataset.make_initializable_iterator()
		saver=tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(iterator.initializer)
			X, Y = iterator.get_next()
			for i in range(self.num_iterations):
				x,y=sess.run([X,Y])
				image_raw=tf.decode_raw(x, tf.float64)
				label_raw=tf.decode_raw(y,tf.float32)
				image_np=sess.run(tf.cast(tf.reshape(image_raw, [self.batch_size, 227, 227, 3]), tf.float32))
				label_np=sess.run(tf.cast(tf.reshape(label_raw, [self.batch_size, 102]), tf.float32))
				#sess.run(tf.assign(self.image,tf.cast(tf.reshape(image_raw, [self.batch_size, 227, 227, 3]), tf.float32)))
				#sess.run(tf.assign(self.label,tf.cast(tf.reshape(label_raw, [self.batch_size, 102]), tf.float32)))
				#self.y.eval(feed_dict={self.image:image_np})
				#self.optimize.eval(feed_dict={self.label:label_np})
				#sess.run([self.optimize,self.y])
				sess.run([self.optimize,self.y],feed_dict={self.image:image_np,self.label:label_np})
				if (i%10==0):
					#print('loss:',sess.run([self.cross_entropy]))
					#print('accuracy:',sess.run([self.accu]))
					print('loss:',sess.run([self.cross_entropy],feed_dict={self.image:image_np,self.label:label_np}))
					print('accuracy:',sess.run([self.accu],feed_dict={self.image:image_np,self.label:label_np}))
				if (i%50==0):
					saver.save(sess,self.checkpoint_dir,global_step=i)





if __name__=='__main__':
	trainer=train(path_to_tfrecordfile,no_of_classes,bach_size,number_iterations,path_to_save_checkpoint)
	trainer.train()
