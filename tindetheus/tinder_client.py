# MIT License
#
# Copyright (c) 2019 Charles Jekel
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

# The facenet implementation has been hard coded into tindetheus. This has been
# hardcoded into tindetheus for the following reasons: 1) there is no setup.py
# for facenet yet. 2) to prevent changes to facenet from breaking tindetheus.
#
# facenet is used to align the database, crop the faces in database, and
# to calculate the embeddings for the database. I've included the copyright
# from facenet below. The specific code that is in this file from facenet
# is within the like_or_dislike_users(self, users) function.

# facenet was created by David Sandberg and is available at
# https://github.com/davidsandberg/facenet with the following MIT license:

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

from builtins import input

import os
import time
import pynder
from pynder.errors import RecsTimeout

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib

from tindetheus import tindetheus_align
import tindetheus.facenet_clone.facenet as facenet
from tindetheus.export_embeddings import load_and_align_data
import tindetheus.image_processing as imgproc
import tindetheus.machine_learning as ml

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class client:
    # a class to manage the pynder api
    def __init__(self, facebook_token, distance, model_dir, likes_left=100,
                 tfsess=None, x_auth_token=None):
        self.session = self.login(facebook_token, x_auth_token)
        self.likes_left = likes_left
        # set facenet model dir
        self.model_dir = model_dir
        #   set your search distance in miles
        self.search_distance = distance
        self.session.profile.distance_filter = self.search_distance
        self.sess = tfsess
        # ensure that there is a temp_images dir
        if not os.path.exists('temp_images'):
            os.makedirs('temp_images')
        if not os.path.exists('database/like'):
            os.makedirs('database/like')
        if not os.path.exists('database/dislike'):
            os.makedirs('database/dislike')
        if not os.path.exists('al_database'):
            os.makedirs('al_database')

        # attempt to load database
        try:
            self.database = list(np.load('database.npy',
                                         allow_pickle=True))
            print('You have browsed', len(self.database), 'Tinder profiles.')
        except FileNotFoundError:
            self.database = []

        # attempt to load an auto liked or disliked database
        try:
            self.al_database = list(np.load('al_database.npy',
                                            allow_pickle=True))
            print('You have automatically liked or disliked ',
                  len(self.al_database), 'Tinder profiles.')
        except FileNotFoundError:
            self.al_database = []

    def login(self, facebook_token=None, x_auth_token=None):
        # login to Tinder using pynder
        session = pynder.Session(facebook_token=facebook_token,
                                 XAuthToken=x_auth_token)
        print('Hello ', session.profile)
        return session

    def look_at_users(self, users):
        # Browse user profiles one at a time. You will be presented with the
        # opportunity to like or dislike profiles. Your history will be
        # stored in a database that you can use for training.
        for user in users:
            print('********************************************************')
            print(user.name, user.age, 'Distance in km: ', user.distance_km)
            print('Schools: ', user.schools)
            print('Job: ', user.jobs)
            print(user.bio)
            print('--------------------------------------------------------')
            print('Do you like this user?')
            print('type l or s for like, or j or f for dislike   ')
            urls = user.get_photos(width='640')
            image_list = imgproc.download_url_photos(urls, user.id)
            imgproc.show_images(image_list)

            didILike = ml.like_or_dislike()
            plt.close('all')

            dbase_names = imgproc.move_images(image_list, user.id, didILike)

            if didILike == 'Like':
                print(user.like())
                self.likes_left -= 1
            else:
                print(user.dislike())
            userList = [user.id, user.name, user.age, user.bio,
                        user.distance_km, user.jobs, user.schools,
                        user.get_photos(width='640'), dbase_names, didILike]
            self.database.append(userList)
            np.save('database.npy', self.database)

    def like_or_dislike_users(self, users):
        # automatically like or dislike users based on your previously trained
        # model on your historical preference.

        # facenet settings from export_embeddings....
        data_dir = 'temp_images_aligned'
        embeddings_name = 'temp_embeddings.npy'
        # labels_name = 'temp_labels.npy'
        # labels_strings_name = 'temp_label_strings.npy'
        is_aligned = True
        image_size = 160
        margin = 44
        gpu_memory_fraction = 1.0
        image_batch = 1000
        prev_user = None
        for user in users:
            imgproc.clean_temp_images()
            urls = user.get_photos(width='640')
            image_list = imgproc.download_url_photos(urls, user.id,
                                                     is_temp=True)
            # align the database
            tindetheus_align.main(input_dir='temp_images',
                                  output_dir='temp_images_aligned')
            # export the embeddings from the aligned database

            train_set = facenet.get_dataset(data_dir)
            image_list_temp, label_list = facenet.get_image_paths_and_labels(train_set)  # noqa: E501

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")  # noqa: E501
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # noqa: E501
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")  # noqa: E501

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list_temp)
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
                    images = facenet.load_data(image_list_temp[i*batch_size:n],  # noqa: E501
                                                False, False,
                                                image_size)
                else:
                    images = load_and_align_data(image_list_temp[i*batch_size:n],  # noqa: E501
                                                    image_size, margin,
                                                    gpu_memory_fraction)
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}
                # Use the facenet model to calculate embeddings
                embed = self.sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i*batch_size:n, :] = embed
                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            # export embeddings and labels
            label_list = np.array(label_list)

            np.save(embeddings_name, emb_array)

            if emb_array.size > 0:
                # calculate the n average embedding per profiles
                X = ml.calc_avg_emb_temp(emb_array)
                # evaluate on the model
                yhat = self.model.predict(X)

                if yhat[0] == 1:
                    didILike = 'Like'
                    # check to see if this is the same user as before
                    if prev_user == user.id:
                        imgproc.clean_temp_images_aligned()
                        print('\n\n You have already liked this user!!! \n \n')
                        print('This typically means you have used all of your'
                              ' free likes. Exiting program!!! \n\n')
                        self.likes_left = -1
                        return
                    else:
                        prev_user = user.id
                else:
                    didILike = 'Dislike'
            else:
                # there were no faces in this profile
                didILike = 'Dislike'
            print('**************************************************')
            print(user.name, user.age, didILike)
            print('**************************************************')

            dbase_names = imgproc.move_images_temp(image_list, user.id)

            if didILike == 'Like':
                print(user.like())
                self.likes_left -= 1
            else:
                print(user.dislike())
            userList = [user.id, user.name, user.age, user.bio,
                        user.distance_km, user.jobs, user.schools,
                        user.get_photos(width='640'), dbase_names,
                        didILike]
            self.al_database.append(userList)
            np.save('al_database.npy', self.al_database)
            imgproc.clean_temp_images_aligned()

    def browse(self):
        # browse for Tinder profiles
        while self.likes_left > 0:
            try:
                users = self.session.nearby_users()
                self.look_at_users(users)
            except RecsTimeout:
                print('Likes left = ', self.likes_left)
                search_string = '''*** There are no users found!!! ***
Would you like us to increase the search distance by 5 miles?
Enter anything to quit, Enter l or s to increase the search distance.
'''
                print(search_string)
                stayOrQuit = input()
                if stayOrQuit == 'l' or stayOrQuit == 's':
                    # if self.search_distance < 100:
                    self.search_distance += 5
                    self.session.profile.distance_filter += 5
                    self.browse()
                else:
                    break

    def like(self):
        # like and dislike Tinder profiles using your trained logistic
        # model. Note this requires that you first run tindetheus browse to
        # build a database. Then run tindetheus train to train a model.

        # load the pretrained model
        self.model = joblib.load('log_reg_model.pkl')

        while self.likes_left > 0:
            try:
                users = self.session.nearby_users()
                self.like_or_dislike_users(users)
            except RecsTimeout:
                self.search_distance += 5
                self.session.profile.distance_filter += 5
                self.like()
