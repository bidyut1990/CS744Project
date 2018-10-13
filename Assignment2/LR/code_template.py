import tensorflow as tf
import numpy as np
import numpy.random as rnd

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
	"worker" : [
		"10.10.1.2:2222"
	]
})

clusterSpec_cluster = tf.train.ClusterSpec({
	"ps" : [
		"10.10.1.2:3333"
	],
	"worker" : [
		"10.10.1.2:2222",
		"10.10.1.4:2222"
	]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
	"ps" : [
		"10.10.1.2:3333"
	],
	"worker" : [
		"10.10.1.2:2222",
		"10.10.1.4:2222",
		"10.10.1.3:2222",
		"10.10.1.1:2222"
	]
})

clusterSpec = {
	"single": clusterSpec_single,
	"cluster": clusterSpec_cluster,
	"cluster2": clusterSpec_cluster2
}



clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
#create session on server
sess = tf.Session(server.target)

if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":
	#put your code here
	train_X, train_y = (mnist.train.images,mnist.train.labels)
	test_X, test_y = (mnist.test.images,mnist.test.labels)
	classCount = train_y.shape[1]
	featureCount = train_X.shape[1]
	trainSize = train_X.shape[0]
	print("Num features:",featureCount,"Num classes:", classCount)
	print("Train size:",trainSize)
	
	W = tf.Variable(tf.random_normal(shape=[featureCount, classCount]))
	b = tf.Variable(tf.random_normal(shape=[1, classCount]))	
	
	data = tf.placeholder(dtype=tf.float32, shape=[None, featureCount])
	target = tf.placeholder(dtype=tf.float32, shape=[None, classCount])

	init = tf.global_variables_initializer()
	#sess = tf.Session()
	sess.run(init)

	prediction  = tf.nn.softmax(tf.matmul(data, W) + b)
	loss = tf.reduce_mean(-tf.reduce_sum(target*tf.log(prediction), reduction_indices=1))

	learning_rate = 0.5
	batch_size = int(trainSize*0.01)# 1% of examples
	num_batches = 100
	iter_num = 1
	
	opt = tf.train.GradientDescentOptimizer(learning_rate)
	goal = opt.minimize(loss)

	correct = tf.cast(tf.equal(tf.round(prediction), target), dtype=tf.float32)
	accuracy = tf.reduce_mean(correct)

	#loss_trace = []
	#train_acc = []
	#test_acc = []
	for epoch in range(iter_num):
		for index in range(num_batches)
			# Generate continuos batch index as images already in random seq
			batch = range(index*batch_size, min(trainSize, (index+1)*batch_size))
			batch_X,batch_y = train_X[batch], train_y[batch]
			sess.run(goal, feed_dict={data: batch_X, target: batch_y})
			#temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
			# convert into a matrix, and the shape of the placeholder to correspond
			#temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_y)})
			#temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_y)})
			# recode the result
			#loss_trace.append(temp_loss)
			#train_acc.append(temp_train_acc)
			#test_acc.append(temp_test_acc)
			# output
			#if (epoch + 1) % 3 == 0:
			#	print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
			#		temp_train_acc, temp_test_acc))
	test_acc = sess.run(accuracy, feed_dict={data: test_X, target: test_y})
	test_loss = sess.run(loss, feed_dict={data: test_X, target: test_y})
	print("Test Set accuracy: ",test_acc)
	print("Test Set loss: ",test_loss)