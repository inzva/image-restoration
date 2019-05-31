# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import datetime
import time

#FILE PATHS#
input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'
#FILE PATHS#

#HYPERPARAMETERS#
network_size_factor = 1 #network size multiplier. minimum allowed value is 0.25.
#HYPERPARAMETERS#
nsf = int(network_size_factor * 4) #for easier implementation


# get test IDs

#we dont want all images that start with 1, that is too much.
#test_fns = glob.glob(gt_dir + '/1*.ARW')

#we dont want images starting with 0021. these are training images
#test_fns = glob.glob(gt_dir + '/0021*.ARW')

#we want images starting with 100
test_fns = glob.glob(gt_dir + '/100*.ARW')

test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output



def network(input):
    conv1 = slim.conv2d(input, 8*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 8*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 16*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 16*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 32*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 32*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 64*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 64*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 128*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 128*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 64*nsf, 128*nsf)
    conv6 = slim.conv2d(up6, 64*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 64*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 32*nsf, 64*nsf)
    conv7 = slim.conv2d(up7, 32*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 32*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 16*nsf, 32*nsf)
    conv8 = slim.conv2d(up8, 16*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 16*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 8*nsf, 16*nsf)
    conv9 = slim.conv2d(up9, 8*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 8*nsf, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])

gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = network(in_image)


test_loss = tf.reduce_mean(tf.image.ssim_multiscale(out_image, gt_image, max_val=255))


saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        #lets print this later 
        #print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
	

        start_time_camera_pipeline = time.time()
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        end_time_camera_pipeline = time.time()
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        camera_time = 	end_time_camera_pipeline - start_time_camera_pipeline



        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        start_time_network = time.time()


        output, mytest_loss = sess.run([out_image,test_loss], feed_dict={in_image: input_full, gt_image:gt_full})


        end_time_network = time.time()
        output = np.minimum(np.maximum(output, 0), 1)

        network_time = 	end_time_network - start_time_network




        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]


        
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the groundtruth






        print(in_fn, "{0} {1} {2}".format(network_time,camera_time,mytest_loss))





        current_time = datetime.datetime.now()
        cur = current_time.strftime("%d-%m-%Y_%H-%M-%S")

        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_out_%s.png' % (test_id, ratio,cur))
        scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_scale_%s.png' % (test_id, ratio, cur))
        scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_gt_%s.png' % (test_id, ratio, cur))
