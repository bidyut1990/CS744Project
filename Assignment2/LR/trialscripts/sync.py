import tensorflow as tf
import numpy as np
import numpy.random as rnd
import time
from tensorflow.examples.tutorials.mnist import input_data

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

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
is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

REPLICAS_TO_AGGREGATE = 2
total_num_replicas = 2
learning_rate = 0.5
max_epochs = 20

if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":
    #Select data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_X, train_y = (mnist.train.images,mnist.train.labels)
	test_X, test_y = (mnist.test.images,mnist.test.labels)
	classCount = train_y.shape[1]
	featureCount = train_X.shape[1]
	trainSize = train_X.shape[0]
	print("Num features:",featureCount,"Num classes:", classCount)
	print("Train size:",trainSize)

    # Graph
    worker_device = "/job:%s/task:%d/cpu:0" % (FLAGS.job_name,FLAGS.task_index)
    with tf.device(tf.train.replica_device_setter(ps_tasks=1,worker_device=worker_device)):

        W = tf.Variable(tf.random_normal(shape=[featureCount, classCount]))
        b = tf.Variable(tf.random_normal(shape=[1, classCount]))
        
        global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        data = tf.placeholder(dtype=tf.float32, shape=[None, featureCount])
        target = tf.placeholder(dtype=tf.float32, shape=[None, classCount])
        
        prediction  = tf.nn.softmax(tf.matmul(data, W) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(target*tf.log(prediction), reduction_indices=1))
        
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        sync_opt = tf.train.SyncReplicasOptimizer(optimizer,replicas_to_aggregate=REPLICAS_TO_AGGREGATE, total_num_replicas=total_num_replicas)
        goal = opt.minimize(loss, global_step=global_step)
        
        correct = tf.cast(tf.equal(tf.round(prediction), target), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)
        
    # Session
    sync_replicas_hook = sync_opt.make_session_run_hook(is_chief)
    stop_hook = tf.train.StopAtStepHook(last_step=10)
    hooks = [sync_replicas_hook,stop_hook]

    # Monitored Training Session
    sess = tf.train.MonitoredTrainingSession(master = server.target, 
          is_chief=is_chief,
          hooks=hooks,
          stop_grace_period_secs=10)
    
    test_acc = []
    loss_trace = []
    print('Starting training on worker %d'%FLAGS.task_index)
    for epochs in range(max_epochs):
        _,l,gs=sess.run([goal,loss,global_step], feed_dict={data: train_X, target: train_y})
        print('loss: ',l,'step: ',gs,'worker: ',FLAGS.task_index)
        test_acc.append(sess.run(accuracy, feed_dict={data: test_X, target: test_y}))
        loss_trace.append(train_loss)
        if is_chief: time.sleep(1)
        time.sleep(1)
    print('Done',FLAGS.task_index)

    time.sleep(10) #grace period to wait before closing session
    sess.close()
    print('Session from worker %d closed cleanly'%FLAGS.task_index)