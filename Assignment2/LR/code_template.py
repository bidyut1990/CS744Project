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

classCount = 10
featureCount = 784

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
#create session on server
sess = tf.Session(server.target)

if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":
	#put your code here
	W = tf.Variable(tf.random_normal(shape=[featureCount, classCount]))
	b = tf.Variable(tf.random_normal(shape=[1, classCount]))	
	
	data = tf.placeholder(dtype=tf.float32, shape=[None, featureCount])
	target = tf.placeholder(dtype=tf.float32, shape=[None, classCount])

	init = tf.global_variables_initializer()
	#sess = tf.Session()
	sess.run(init)

	prediction  = tf.nn.softmax(tf.matmul(data, W) + b)
	loss = tf.reduce_mean(-tf.reduce_sum(target*tf.log(prediction), reduction_indices=1))

	learning_rate = 0.003
	batch_size = 30
	iter_num = 150

	train_X, train_y = (mnist.train.images,mnist.train.labels)
	test_X, test_y = (mnist.test.images,mnist.test.labels)

	opt = tf.train.GradientDescentOptimizer(learning_rate)

	goal = opt.minimize(loss)

	#prediction = tf.round(tf.sigmoid(mod))
	# Bool into float32 type
	correct = tf.cast(tf.equal(tf.round(prediction), target), dtype=tf.float32)
	# Average
	accuracy = tf.reduce_mean(correct)

	loss_trace = []
	train_acc = []
	test_acc = []

	for epoch in range(iter_num):
		# Generate random batch index
		batch_index = np.random.choice(len(train_X), size=batch_size)
		batch_train_X = train_X[batch_index]
		batch_train_y = np.matrix(train_y[batch_index])
		sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})
		temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
		# convert into a matrix, and the shape of the placeholder to correspond
		temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_y)})
		temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_y)})
		# recode the result
		loss_trace.append(temp_loss)
		train_acc.append(temp_train_acc)
		test_acc.append(temp_test_acc)
		# output
		if (epoch + 1) % 3 == 0:
			print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
				temp_train_acc, temp_test_acc))
