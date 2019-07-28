"""
Exports the embeddings and labels of a directory of images as numpy arrays.

Typicall usage expect the image directory to be of the openface/facenet form
and the images to be aligned. Simply point to your model and your image
directory:
python facenet/contributed/export_embeddings.py
 ~/models/facenet/20170216-091149/ ~/datasets/lfw/mylfw

Output:
embeddings.npy -- Embeddings as np array, Use --embeddings_name to change name
labels.npy -- Integer labels as np array, Use --labels_name to change name
label_strings.npy -- Strings from folders names,
 --labels_strings_name to change name


Use --image_batch to dictate how many images to load in memory at a time.

If your images aren't already pre-aligned, use --is_aligned False

I started with compare.py from David Sandberg, and modified it to export
the embeddings. The image loading is done use the facenet library if the image
is pre-aligned. If the image isn't pre-aligned, I use the compare.py function.
I've found working with the embeddings useful for classifications models.

Charles Jekel 2017

"""
# MIT License
#
# Copyright (c) 2017, 2018 Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import imageio
from skimage.transform import resize
import time
import tensorflow as tf
import numpy as np
# import sys
import os
# import argparse
import tindetheus.facenet_clone.facenet as facenet
from tindetheus.facenet_clone.align import detect_face
# import glob

# I think this is a decent Python2 / Python3 fix w/o imports
xrange = range


def main(model_dir='20170512-110547', data_dir='database_aligned',
         is_aligned=True, image_size=160, margin=44, gpu_memory_fraction=1.0,
         image_batch=1000, embeddings_name='embeddings.npy',
         labels_name='labels.npy', labels_strings_name='label_strings.npy',
         return_image_list=False):
    train_set = facenet.get_dataset(data_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(data_dir)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    # get the label strings
    label_strings = [name for name in classes if
                     os.path.isdir(os.path.join(path_exp, name))]
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(model_dir)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")  # noqa: E501
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # noqa: E501
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")  # noqa: E501

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list)
            print('Number of images: ', nrof_images)
            batch_size = image_batch
            if nrof_images % batch_size == 0:
                nrof_batches = nrof_images // batch_size
            else:
                nrof_batches = (nrof_images // batch_size) + 1
            print('Number of batches: ', nrof_batches)
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches - 1:
                    n = nrof_images
                else:
                    n = i*batch_size + batch_size
                # Get images for the batch
                if is_aligned is True:
                    images = facenet.load_data(image_list[i*batch_size:n],
                                               False, False, image_size)
                else:
                    images = load_and_align_data(image_list[i*batch_size:n],
                                                 image_size, margin,
                                                 gpu_memory_fraction)
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}
                # Use the facenet model to calculate embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i*batch_size:n, :] = embed
                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            #   export embeddings and labels
            label_list = np.array(label_list)

            np.save(embeddings_name, emb_array)
            if emb_array.size > 0:
                labels_final = (label_list)-np.min(label_list)
                np.save(labels_name, labels_final)
                label_strings = np.array(label_strings)
                np.save(labels_strings_name, label_strings[labels_final])
                np.save('image_list.npy', image_list)
            if return_image_list:
                np.save('validation_image_list.npy', image_list)
                return image_list, emb_array


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)  # noqa: E501
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        print(image_paths[i])
        img = imageio.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet,
                                                    onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = resize(cropped, (image_size, image_size))
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


if __name__ == '__main__':
    main()
