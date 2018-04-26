import argparse
import helper
from distutils.version import LooseVersion
import os.path
import project_tests as tests
import tensorflow as tf
import time
from datetime import timedelta
import warnings


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# Set parameters
L2_REG = 1e-5
STDEV = 1e-3
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 8
IMAGE_SHAPE = (160, 576)  # (256, 512) for Cityscapes
NUM_CLASSES = 2  # 50 for Cityscapes

DATA_DIR = './data'
RUNS_DIR = './runs'
MODEL_DIR = './models'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input, keep_prob, layer3_out, layer4_out, layer7_out

print("Load VGG Model:\r")
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    ## Encoder

    # Set initialization and regularization methods
    init = tf.truncated_normal_initializer(stddev=STDEV)
    reg = tf.contrib.layers.l2_regularizer(L2_REG)

    # 1x1 convolutional layers on vgg l3, l4, and l7 outputs
    conv_l3 = tf.layers.conv2d(
        vgg_layer3_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    conv_l4 = tf.layers.conv2d(
        vgg_layer4_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    conv_l7 = tf.layers.conv2d(
        vgg_layer7_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    ## Decoder

    # upsample encoder output by 2
    deconv_1 = tf.layers.conv2d_transpose(
        conv_l7,
        num_classes,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    # skip connection from vgg layer 4
    deconv_1 = tf.add(deconv_1, conv_l4)

    # upsample by 2 again
    deconv_2 = tf.layers.conv2d_transpose(
        deconv_1,
        num_classes,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    # skip connection from vgg layer 3
    deconv_2 = tf.add(deconv_2, conv_l3)

    # nn_last_layer - upsample by 8 to return to original input image size
    deconv_3 = tf.layers.conv2d_transpose(
        deconv_2,
        num_classes,
        kernel_size=16,
        strides=8,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    return deconv_3

print("Layers Test:\r")
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

print("Optimize Test:\r")
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver, model_dir):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param saver: TF method for saving checkpoints
    :param model_dir: directory where checkpoints are saved
    """

    # TensorBoard integration
    # summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(LOGDIR + hparam)
    # writer.add_graph(sess.graph)

    counter = 1
    for epoch in range(epochs):
        start_time = time.time()
        for image, label in get_batches_fn(batch_size):
            if counter % 5 == 0:
                # Run optimizer and merge TB summary
                # TODO: to activate TB, add "s" variable after loss
                _, loss = sess.run([train_op, cross_entropy_loss], #, summary],
                                   feed_dict={input_image: image,
                                              correct_label: label,
                                              keep_prob: KEEP_PROB,
                                              learning_rate: LEARNING_RATE})
                # writer.add_summary(s, counter)
            # Run optimizer without TB summary
            else:
                _, loss = sess.run([train_op, cross_entropy_loss],
                                      feed_dict={input_image: image,
                                                 correct_label: label,
                                                 keep_prob: KEEP_PROB,
                                                 learning_rate: LEARNING_RATE})
            counter += 1
        # Print data on the learning process
        print("Epoch: {}".format(epoch+1), "/ {}".format(EPOCHS), " Loss: {:.3f}".format(loss), " Time: ",
              str(timedelta(seconds=(time.time()-start_time))))
        # Save checkpoint every N epochs
        if (epoch+1) % 5 == 0:
            save_path = saver.save(sess, os.path.join(model_dir, 'cfn_epoch_' + str(epoch) + '.ckpt'))

print("NN Train Test:")
tests.test_train_nn(train_nn)


def run():
    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    print("Start training...\r")
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Variable placeholders
        correct_label = tf.placeholder(tf.int32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUM_CLASSES)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, NUM_CLASSES)

        # Train NN using the train_nn function
        tf.set_random_seed(237)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver, MODEL_DIR)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

def predict():
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # Variable placeholders
        correct_label = tf.placeholder(tf.int32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUM_CLASSES)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, NUM_CLASSES)

        # Restore model from checkpoint
        tf.set_random_seed(47)
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('./models/cfn_epoch_9.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        help='what to do: train/predict/freeze/optimise/video',
                        type=str,
                        choices=['train', 'predict', 'freeze', 'optimise', 'video'])
    # parser.add_argument('-g', '--gpu', help='number of GPUs to use. default 0 (use CPU)', type=int, default=0)
    # parser.add_argument('-gm','--gpu_mem', help='GPU memory fraction to use. default 0.9', type=float, default=0.9)
    # parser.add_argument('-x','--xla', help='XLA JIT level. default None', type=int, default=None, choices=[1,2])
    # parser.add_argument('-ep', '--epochs', help='training epochs. default 0', type=int, default=0)
    # parser.add_argument('-bs', '--batch_size', help='training batch size. default 5', type=int, default=5)
    # parser.add_argument('-lr', '--learning_rate', help='training learning rate. default 0.0001', type=float, default=0.0001)
    # parser.add_argument('-kp', '--keep_prob', help='training dropout keep probability. default 0.9', type=float, default=0.9)
    # parser.add_argument('-rd', '--runs_dir', help='training runs directory. default runs', type=str, default='runs')
    # parser.add_argument('-cd', '--ckpt_dir', help='training checkpoints directory. default ckpt', type=str, default='ckpt')
    # parser.add_argument('-sd', '--summary_dir', help='training tensorboard summaries directory. default summaries', type=str, default='summaries')
    # parser.add_argument('-md', '--model_dir', help='model directory. default None - model directory is created in runs. needed for predict', type=str, default=None)
    # parser.add_argument('-fd', '--frozen_model_dir', help='model directory for frozen graph. for freeze', type=str, default=None)
    # parser.add_argument('-od', '--optimised_model_dir', help='model directory for optimised graph. for optimize', type=str, default=None)
    # parser.add_argument('-ip', '--images_paths', help="images path/file pattern. e.g. 'train/img*.png'", type=str, default=None)
    # parser.add_argument('-lp', '--labels_paths', help="label images path/file pattern. e.g. 'train/label*.png'", type=str, default=None)
    # parser.add_argument('-vi', '--video_file_in', help="mp4 video file to process", type=str, default=None)
    # parser.add_argument('-vo', '--video_file_out', help="mp4 video file to save results", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    if args.action == 'predict':
        predict()
    elif args.action == 'train':
        run()
    else:
        print('Error: Please provide an action.\r')
