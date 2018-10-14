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

	learning_rate = 0.05
	num_batches = 550
	batch_size = int(trainSize/num_batches)# 1% of examples
	iter_num = 1
	
	opt = tf.train.GradientDescentOptimizer(learning_rate)
	#Provide the global step cunt as the second argument to track no. of steps taken
	goal = opt.minimize(loss)

	correct = tf.cast(tf.equal(tf.round(prediction), target), dtype=tf.float32)
	accuracy = tf.reduce_mean(correct)

	#train_acc = []
	test_acc = [sess.run(accuracy, feed_dict={data: test_X, target: test_y})]
	prev_loss = np.inf
	train_loss = sess.run(loss, feed_dict={data: train_X, target: train_y})
	loss_trace = [train_loss]
	#epsilon = 0.01
	#epoch = 0
	max_epochs = 40
	#Run until convergence
	#while abs(prev_loss - train_loss)>epsilon:
	for epoch in range(max_epochs):
		#TODO randomize data
		#prev_loss = train_loss
		#train_loss = 0
		#for index in range(num_batches):
		# Generate continuos batch index as images already in random seq
		#batch = range(index*batch_size, min(trainSize, (index+1)*batch_size))
		#batch_X,batch_y = mnist.train.next_batch(batch_size)
		# optimize on this batch
		_,train_loss = sess.run([goal,loss], feed_dict={data: train_X, target: train_y})
		#train_loss += batch_loss/num_batches
		# make measurements on test set
		#train_loss = sess.run(loss, feed_dict={data: train_X, target: train_y})
		test_acc.append(sess.run(accuracy, feed_dict={data: test_X, target: test_y}))
		loss_trace.append(train_loss)
		#epoch += 1
		print("Current training loss: ", train_loss)
		print("Current test accuracy: ", test_acc[-1])
		print("epochs completed: ",epoch)
	
	print("final Test Set accuracy: ",test_acc[-1])
	print("final Train Set loss: ",train_loss)
	print("Total epochs: ",epoch)
