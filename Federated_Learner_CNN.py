# Copyright 2018 coMind. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# https://comind.org/
# ==============================================================================

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#CNN mode detection
from Mode_Detection_CNN import *

# Helper libraries
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt

# Import custom optimizer
import federated_averaging_optimizer

flags = tf.app.flags
flags.DEFINE_integer("task_index", 1,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task that performs the variable "
                     "initialization ")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "worker", "job name: worker or ps")

# You can safely tune these variables
BATCH_SIZE = 32
EPOCHS = 5
INTERVAL_STEPS = 100
# ----------------

FLAGS = flags.FLAGS

if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

# Only enable GPU for worker 1 (not needed if training with separate machines)
if FLAGS.task_index == 0:
    print('--- GPU Disabled ---')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

#Construct the cluster and start the server
ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")

# Get the number of workers.
num_workers = len(worker_spec)
print('{} workers defined'.format(num_workers))

cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
# Parameter server will block here
if FLAGS.job_name == "ps":
    print('--- Parameter Server Ready ---')
    server.join()

# Load dataset as Pandas dataframe
# X_train, X_test, Y_train_labels, Y_test_labels, predicted_label_train_df, predicted_label_df, prob_train_df, prob_test_df = load_dataset()
# X_train, X_test, Y_train, Y_test = load_dataset()


#parameters fefinition

num_channels_ensemble = [5]
num_filters_ensemble = []
filters_size_ensemble = []
num_stride_maxpool_ensemble = []
num_stride_conv2d_ensemble = []
maxpool_size_ensemble = []


####The data structures in the following data files are different from those in ensemble paper
X_train = np.load("D:/OneDrive - Concordia University - Canada/PycharmProjects/Itinerum-Deep-Neural-Network/data/augmenteddata_5channels/X_train_final.npy")

Y_train = np.load(
    "D:/OneDrive - Concordia University - Canada/PycharmProjects/Itinerum-Deep-Neural-Network/data/augmenteddata_5channels/Y_train_final.npy")

print(Y_train.shape)
Y_onehot = np.zeros((Y_train.shape[0], 4))

Y_onehot[np.arange(Y_train.shape[0]), Y_train] = 1

Y_train = np.copy(Y_onehot)
print(Y_train.shape)

print('Data loaded')

is_chief = (FLAGS.task_index == 0)

checkpoint_dir='logs_dir/federated_worker_{}/{}'.format(FLAGS.task_index, time())
print('Checkpoint directory: ' + checkpoint_dir)

worker_device = "/job:worker/task:%d" % FLAGS.task_index
print('Worker device: ' + worker_device + ' - is_chief: {}'.format(is_chief))

# Place all ops in the local worker by default
with tf.device(worker_device):
    global_step = tf.train.get_or_create_global_step()

    # Define input pipeline, place these ops in the cpu
    with tf.name_scope('dataset'), tf.device('/cpu:0'):
        # Placeholders for the iterator
        X_placeholder = tf.placeholder(tf.float32, shape=(None, seg_size, num_channels), name='X_placeholder')
        Y_placeholder = tf.placeholder(tf.float32, shape=[None, num_classes], name='Y_placeholder')
        minibatch_weights = tf.placeholder(tf.float32, shape=[None], name='minibatch_weights')
        batch_size = tf.placeholder(tf.int64, name='batch_size')
        shuffle_size = tf.placeholder(tf.int64, name='shuffle_size')

        # Create dataset from numpy arrays, shuffle, repeat and batch
        dataset = tf.data.Dataset.from_tensor_slices((X_placeholder, Y_placeholder))
        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
        dataset = dataset.repeat(EPOCHS)
        dataset = dataset.batch(batch_size)


        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        X, y = iterator.get_next()

    # Define our model

    num_layers_ensemble, filters_size_ensemble, num_filters_ensemble, maxpool_size_ensemble, num_stride_conv2d_ensemble, num_stride_maxpool_ensemble, weights_ensemble = parameters_weights()

    filters_size = filters_size_ensemble[0]
    num_filters = num_filters_ensemble[0]
    num_stride_conv2d = num_stride_conv2d_ensemble[0]
    num_stride_maxpool = num_stride_maxpool_ensemble[0]
    maxpool_size = maxpool_size_ensemble[0]
    weights = weights_ensemble[0]

    # Initialize parameters
    parameters = initialize_parameters(weights)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    predictions = forward_propagation(X, parameters, num_stride_conv2d, maxpool_size, num_stride_maxpool)


    # Object to keep moving averages of our metrics (for tensorboard)
    summary_averages = tf.train.ExponentialMovingAverage(0.9)

    # Define cross_entropy loss
    with tf.name_scope('loss'):
        #loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=predictions, weights=minibatch_weights)
        print("type logits:", type(predictions))
        print("dim logits:", predictions.shape)

        print("type labels:", type(y))
        print("dim labels:", y.shape)
        # loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, predictions))
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predictions)
        loss=tf.losses.softmax_cross_entropy(onehot_labels=y, logits=predictions, weights=1)
        loss_averages_op = summary_averages.apply([loss])
        # Store moving average of the loss
        tf.summary.scalar('cross_entropy', summary_averages.average(loss))

    # Define accuracy metric
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # Compare prediction with actual label
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.cast(tf.argmax(y, 1), tf.int64))
        # Average correct predictions in the current batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_metric')
        accuracy_averages_op = summary_averages.apply([accuracy])
        # Store moving average of the accuracy
        tf.summary.scalar('accuracy', summary_averages.average(accuracy))

    # Define optimizer and training op
    with tf.name_scope('train'):
        # Define device setter to place copies of local variables
        device_setter = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
        # Wrap optimizer in a FederatedAveragingOptimizer for federated training
        optimizer = federated_averaging_optimizer.FederatedAveragingOptimizer(tf.train.AdamOptimizer(0.001), replicas_to_aggregate=num_workers, interval_steps=INTERVAL_STEPS, is_chief=is_chief, device_setter=device_setter)
        # Make train_op dependent on moving averages ops. Otherwise they will be
        # disconnected from the graph
        with tf.control_dependencies([loss_averages_op, accuracy_averages_op]):
            train_op = optimizer.minimize(loss, global_step=global_step)
        # Define a hook for optimizer initialization
        federated_hook = optimizer.make_session_run_hook()

    n_batches = int(X_train.shape[0] / BATCH_SIZE)
    last_step = int(n_batches * EPOCHS)

    print('Graph definition finished')

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        operation_timeout_in_ms=2000000,
        device_filters=["/job:ps",
        "/job:worker/task:%d" % FLAGS.task_index])

    print('Training {} batches...'.format(last_step))

    # Logger hook to keep track of the training
    class _LoggerHook(tf.train.SessionRunHook):
      def begin(self):
          self._total_loss = 0
          self._total_acc = 0

      def before_run(self, run_context):
          return tf.train.SessionRunArgs([loss, accuracy, global_step])

      def after_run(self, run_context, run_values):
          loss_value, acc_value, step_value = run_values.results
          self._total_loss += loss_value
          self._total_acc += acc_value
          if (step_value + 1) % n_batches == 0:
              print("Epoch {}/{} - loss: {:.4f} - acc: {:.4f}".format(int(step_value / n_batches) + 1, EPOCHS, self._total_loss / n_batches, self._total_acc / n_batches))
              self._total_loss = 0
              self._total_acc = 0

    # Hook to initialize the dataset
    class _InitHook(tf.train.SessionRunHook):
        def after_create_session(self, session, coord):
            session.run(dataset_init_op, feed_dict={X_placeholder: X_train, Y_placeholder: Y_train, batch_size: BATCH_SIZE, shuffle_size: X_train.shape[0]})

    # Hook to save just trainable_variables
    class _SaverHook(tf.train.SessionRunHook):
        def begin(self):
            self._saver = tf.train.Saver(tf.trainable_variables())

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(global_step)

        def after_run(self, run_context, run_values):
            step_value = run_values.results
            if step_value % n_batches == 0 and not step_value == 0:
                self._saver.save(run_context.session, checkpoint_dir+'/model.ckpt', step_value)

        def end(self, session):
            self._saver.save(session, checkpoint_dir+'/model.ckpt', session.run(global_step))

    # Make sure we do not define a chief worker
    with tf.name_scope('monitored_session'):
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                checkpoint_dir=checkpoint_dir,
                hooks=[_LoggerHook(), _InitHook(), _SaverHook(), federated_hook],
                config=sess_config,
                stop_grace_period_secs=10,
                save_checkpoint_secs=None) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

if is_chief:
    print('--- Begin Evaluation ---')
    # Reset graph and load it again to clean tensors placed in other devices
    tf.reset_default_graph()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', clear_devices=True)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')
        graph = tf.get_default_graph()
        X_placeholder = graph.get_tensor_by_name('dataset/X_placeholder:0')
        Y_placeholder = graph.get_tensor_by_name('dataset/Y_placeholder:0')
        batch_size = graph.get_tensor_by_name('dataset/batch_size:0')
        #shuffle_size = graph.get_tensor_by_name('dataset/shuffle_size:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy_metric:0')
        predictions = graph.get_tensor_by_name('softmax/BiasAdd:0')
        dataset_init_op = graph.get_operation_by_name('dataset/dataset_init')
        sess.run(dataset_init_op, feed_dict={images_placeholder: test_images, labels_placeholder: test_labels, batch_size: test_images.shape[0], shuffle_size: 1})
        print('Test accuracy: {:4f}'.format(sess.run(accuracy)))
        predicted = sess.run(predictions)

    # Plot the first 25 test images, their predicted label, and the true label
    # Color correct predictions in green, incorrect predictions in red
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        predicted_label = np.argmax(predicted[i])
        true_label = test_labels[i]
        if predicted_label == true_label:
          color = 'green'
        else:
          color = 'red'
        plt.xlabel("{} ({})".format(class_names[predicted_label],
                                    class_names[true_label]),
                                    color=color)

    plt.show(True)
