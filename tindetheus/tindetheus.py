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

import sys
import os
import shutil
import argparse
import pynder
import pandas as pd
from pynder.errors import RecsTimeout

import matplotlib.pyplot as plt
import imageio
import numpy as np
try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve

from tindetheus import export_embeddings
from tindetheus import tindetheus_align
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

import tindetheus.facenet_clone.facenet as facenet
from tindetheus.facenet_clone.facenet import to_rgb
import tensorflow as tf

from tindetheus.export_embeddings import load_and_align_data

# add version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()


def clean_temp_images():
    # delete the temp_images dir
    shutil.rmtree('temp_images')
    os.makedirs('temp_images')


def clean_temp_images_aligned():
    # delete the temp images aligned dir
    shutil.rmtree('temp_images_aligned')


def download_url_photos(urls, userID, is_temp=False):
    # define a function which downloads the pictures of urls
    count = 0
    image_list = []
    if is_temp is True:
        os.makedirs('temp_images/temp')
    for url in urls:
        if is_temp is True:
            image_list.append('temp_images/temp/'+userID+'.'+str(count)+'.jpg')
        else:
            image_list.append('temp_images/'+userID+'.'+str(count)+'.jpg')
        urlretrieve(url, image_list[-1])
        count += 1
    return image_list


def move_images_temp(image_list, userID):
    # move images from temp folder to al_database
    count = 0
    database_loc = []
    for i, j in enumerate(image_list):
        new_fname = 'al_database/'+userID+'.'+str(count)+'.jpg'
        try:
            os.rename(j, new_fname)
        except:
            print('WARNING: unable to save file, it may already exist!',
                  'file: ' + new_fname)
        database_loc.append(new_fname)
        count += 1
    return database_loc


def move_images(image_list, userID, didILike):
    # move images from temp folder to database
    if didILike == 'Like':
        fname = 'like/'
    else:
        fname = 'dislike/'
    count = 0
    database_loc = []
    for i, j in enumerate(image_list):
        new_fname = 'database/'+fname+userID+'.'+str(count)+'.jpg'
        try:
            os.rename(j, new_fname)
        except:
            print('WARNING: unable to save file, it may already exist!',
                  'file: ' + new_fname)
        database_loc.append(new_fname)
        count += 1
    return database_loc


def show_images(images, holdon=False, title=None, nmax=49):
    # use matplotlib to display profile images
    n = len(images)
    if n > nmax:
        n = nmax
        n_col = 7
    else:
        n_col = 3
    if n % n_col == 0:
        n_row = n // n_col
    else:
        n_row = n // 3 + 1
    if title is None:
        plt.figure()
    else:
        plt.figure(title)
    plt.tight_layout()
    for j, i in enumerate(images):
        if j == nmax:
            print('\n\nToo many images to show... \n\n')
            break
        temp_image = imageio.imread(i)
        if len(temp_image.shape) < 3:
            # needs to be converted to rgb
            temp_image = to_rgb(temp_image)
        plt.subplot(n_row, n_col, j+1)
        plt.imshow(temp_image)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)

    if holdon is False:
        plt.show(block=False)
        plt.pause(0.1)


def calc_avg_emb():
    # a function to create a vector of n average embeddings for each
    # tinder profile
    # get the embeddings per profile
    labels = np.load('labels.npy')
    # label_strings = np.load('label_strings.npy')
    embeddings = np.load('embeddings.npy')
    image_list = np.load('image_list.npy')

    # determine the n dimension of embeddings
    n_emb = embeddings.shape[1]

    # find the maximum number of images in a profile
    split_image_list = []
    profile_list = []
    for i in image_list:
        split_image_list.append(i.split('.')[1])
        # split_image_list.append(i.replace('/', '.').split('.'))
        profile_list.append(i.split('.')[0])

    # convert profile list to pandas index
    pl = pd.Index(profile_list)
    pl_unique = pl.value_counts()

    # get the summar statics of pl
    pl_describe = pl_unique.describe()
    print('Summary statistics of profiles with at least one detectable face')
    print(pl_describe)
    number_of_profiles = int(pl_describe[0])
    # number_of_images = int(pl_describe[-1])

    # convert the embeddings to a data frame
    eb = pd.DataFrame(embeddings, index=pl)
    dislike = pd.Series(labels, index=pl)
    # if dislike == 1 it means I disliked the person!

    # create a blank numpy array for embeddings
    new_embeddings = np.zeros((number_of_profiles, n_emb))
    new_labels = np.zeros(number_of_profiles)
    for i, j in enumerate(pl_unique.index):
        temp = eb.loc[j]

        # if a profile has more than one face it will be a DataFrame
        # else the profile will be a Series
        if isinstance(temp, pd.DataFrame):
            # get the average of each column
            temp_embeddings = np.mean(temp.values, axis=0)
        else:
            temp_embeddings = temp.values

        # save the new embeddings
        new_embeddings[i] = temp_embeddings

        # Save the profile label, 1 for dislike, 0 for like
        new_labels[i] = dislike[j].max()

    # save the files
    np.save('embeddings_avg_profile.npy', new_embeddings)
    np.save('labels_avg_profile.npy', new_labels)
    return new_embeddings, new_labels


def calc_avg_emb_temp(embeddings):
    # a function to create a vector of n average embeddings for each
    # in the temp_images_aligned folder
    # embeddings = np.load('temp_embeddings.npy')
    # determine the n dimension of embeddings
    n_emb = embeddings.shape[1]
    # calculate the average embeddings
    new_embeddings = np.zeros((1, n_emb))
    new_embeddings[0] = np.mean(embeddings, axis=0)
    return new_embeddings


def fit_log_reg(X, y):
    # fits a logistic regression model to your data
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)
    print('Train size: ', len(X))
    train_score = model.score(X, y)
    print('Training accuracy', train_score)
    ypredz = model.predict(X)
    cm = confusion_matrix(y, ypredz)
    # tn, fp, fn, tp = cm.ravel()
    tn, _, _, tp = cm.ravel()

    # true positive rate When it's actually yes, how often does it predict yes?
    recall = float(tp) / np.sum(cm, axis=1)[1]
    # Specificity: When it's actually no, how often does it predict no?
    specificity = float(tn) / np.sum(cm, axis=1)[0]

    print('Recall/ Like accuracy', recall)
    print('specificity/ Dislike accuracy', specificity)

    # save the model
    joblib.dump(model, 'log_reg_model.pkl')


def like_or_dislike():
    # define a function to get like or dislike input
    likeOrDislike = '0'
    while likeOrDislike != 'j' and likeOrDislike != 'l' \
            and likeOrDislike != 'f' and likeOrDislike != 's':

        likeOrDislike = input()
        if likeOrDislike == 'j' or likeOrDislike == 'f':
            return 'Dislike'
        elif likeOrDislike == 'l' or likeOrDislike == 's':
            return 'Like'
        else:
            print('you must enter either l or s for like,'
                  ' or j or f for dislike')
            likeOrDislike = input()


class client:
    # a class to manage the pynder api
    def __init__(self, facebook_token, distance, model_dir, likes_left=100,
                 tfsess=None):
        self.session = self.login(facebook_token)
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
            self.database = list(np.load('database.npy'))
            print('You have browsed', len(self.database), 'Tinder profiles.')
        except:
            self.database = []

        # attempt to load an auto liked or disliked database
        try:
            self.al_database = list(np.load('al_database.npy'))
            print('You have automatically liked or disliked ',
                  len(self.al_database), 'Tinder profiles.')
        except:
            self.al_database = []

    def login(self, facebook_token):
        # login to Tinder using pynder
        session = pynder.Session(facebook_token)
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
            image_list = download_url_photos(urls, user.id)
            show_images(image_list)

            didILike = like_or_dislike()
            plt.close('all')

            dbase_names = move_images(image_list, user.id, didILike)

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
            clean_temp_images()
            urls = user.get_photos(width='640')
            image_list = download_url_photos(urls, user.id,
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
                X = calc_avg_emb_temp(emb_array)
                # evaluate on the model
                yhat = self.model.predict(X)

                if yhat[0] == 1:
                    didILike = 'Like'
                    # check to see if this is the same user as before
                    if prev_user == user.id:
                        clean_temp_images_aligned()
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

            dbase_names = move_images_temp(image_list, user.id)

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
            clean_temp_images_aligned()

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


def main(args, facebook_token):
    # There are three function choices: browse, build, like
    # browse: review new tinder profiles and store them in your database
    # train: use machine learning to create a new model that likes and dislikes
    # profiles based on your historical preference
    # like: use your machine leanring model to like new tinder profiles
    if args.function == 'browse':
        my_sess = client(facebook_token, args.distance, args.model_dir,
                         likes_left=args.likes)
        my_sess.browse()

    elif args.function == 'train':
        # align the database
        tindetheus_align.main()
        # export the embeddings from the aligned database
        export_embeddings.main(model_dir=args.model_dir,
                               image_batch=args.image_batch)
        # calculate the n average embedding per profiles
        X, y = calc_avg_emb()
        # fit and save a logistic regression model to the database
        fit_log_reg(X, y)

    elif args.function == 'validate':
        print('\n\nAttempting to validate the dataset...\n\n')
        valdir = 'validation'
        # align the validation dataset
        tindetheus_align.main(input_dir=valdir,
                              output_dir=valdir+'_aligned')
        # export embeddings
        # y is the image list, X is the embedding_array
        image_list, emb_array = export_embeddings.main(model_dir=args.model_dir,  # noqa: E501
                                        data_dir=valdir+'_aligned',
                                        image_batch=args.image_batch,
                                        embeddings_name='val_embeddings.npy',
                                        labels_name='val_labels.npy',
                                        labels_strings_name='val_label_strings.npy',  # noqa: E501
                                        return_image_list=True)
        # print(image_list)
        # convert the image list to a numpy array to take advantage of
        # numpy array slicing
        image_list = np.array(image_list)
        print('\n\nEvaluating trained model\n \n')
        model = joblib.load('log_reg_model.pkl')
        yhat = model.predict(emb_array)
        # print(yhat)
        # 0 should be dislike, and 1 should be like
        # if this is backwards, there is probablly a bug...
        dislikes = yhat == 0
        likes = yhat == 1
        show_images(image_list[dislikes], holdon=True, title='Dislike')
        print('\n\nGenerating plots...\n\n')
        plt.title('Dislike')

        show_images(image_list[likes], holdon=True, title='Like')
        plt.title('Like')

        cols = ['Image name', 'Model prediction (0=Dislike, 1=Like)']
        results = np.array((image_list, yhat)).T
        print('\n\nSaving results to validation.csv\n\n')
        my_results_DF = pd.DataFrame(results, columns=cols)
        my_results_DF.to_csv('validation.csv')

        plt.show()

    elif args.function == 'like':
        print('... Loading the facenet model ...')
        print('... be patient this may take some time ...')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # pass the tf session into client object
                my_sess = client(facebook_token, args.distance, args.model_dir,
                                 likes_left=args.likes, tfsess=sess)
                # Load the facenet model
                facenet.load_model(my_sess.model_dir)
                print('Facenet model loaded successfully!!!')
                # automatically like users
                my_sess.like()

    else:
        text = '''You must specify a function. Your choices are either
tindetheus browse
tindetheus train
tindetheus like
tindetheus validate'''
        print(text)


def parse_arguments(argv, defaults):
    help_text = '''There are four function choices: browse, train, like, or validate.
\n

1) tindetheus browse
-- Let's you browse tinder profiles to add to your database.
-- Browses tinder profiles in your distance until you run out.
-- Asks if you'd like to increase the distance by 5 miles.
-- Use to build a database of the tinder profiles you look at.
\n
2) tindetheus train
-- Trains a model to your Tinder database.
-- Uses facenet implementation for facial detection and classification.
-- Saves logistic regression model to classify which faces you like and
-- dislike.
\n
3) tindetheus like
-- Automatically like and dislike Tinder profiles based on your historical
-- preference. First run browse, then run train, then prosper with like.
-- Uses the trained model to automatically like and dislike profiles.
-- Profiles where a face isn't detected are automatically disliked.
\n
4) tindetheus validate
-- This validate functions applies your personally trained tinder model on
-- an external set of images. Place images you'd like to run tindetheus on
-- withing a folder within the validation directory. See README for more
-- details. The results are saved in validation.csv.
\n
Settings are stored in your config.txt file. A typically config.txt will
contain the following:
facebook_token = XXXXXXX  # your facebook token hash
model_dir = 20170512-110547  # the location of your model directory
image_batch = 1000  # number of images to load in a batch during train
# the larger the image_batch size, the faster the training process, at the
# cost of additional memory. A 4GB machine may struggle with 1000 images.
distance = 5  # Set the starting distance in miles
likes = 100  # set the number of likes you want to use
# note that free Tinder users only get 100 likes in 24 hours
\n
Optional arguments will overide config.txt settings.
'''
    parser = argparse.ArgumentParser(description=help_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # noqa: E501
    parser.add_argument('function', type=str, help='browse, train, or like')
    parser.add_argument('--distance', type=int,
                        help='Set the starting distance in miles.'
                        'Tindetheus will crawl in 5 mile increments from here'
                        '. Default=5.', default=defaults['distance'])
    parser.add_argument('--image_batch', type=int,
                        help='The number of images to load in the facenet'
                        ' model at one time. This only affects the train '
                        'functionality. A larger number will be faster at the'
                        ' cost for larger memory. Default=1000.',
                        default=defaults['image_batch'])
    parser.add_argument('--model_dir', type=str, help='Location of your '
                        'pretrained facenet model. Default="20170512-110547"',
                        default=defaults['model_dir'])
    parser.add_argument('--likes', type=int, help='Set the number of likes to '
                        'use. Note that free Tinder users only get 100 likes '
                        'in 24 hour period', default=defaults['likes'])
    parser.add_argument('--version', action='version', version=__version__)
    return parser.parse_args(argv)


def command_line_run():
    # settings to look for
    defaults = {'facebook_token': None,
                'model_dir': '20170512-110547',
                'image_batch': 1000,
                'distance': 5,
                'likes': 100}
    # check for a config file first
    try:
        with open('config.txt') as f:
            lines = f.readlines()
            for line in lines:
                my_line_list = line.split(' ')
                if len(my_line_list) > 1:
                    if my_line_list[0] == 'image_batch':
                        defaults['image_batch'] = int(my_line_list[2].strip('\n'))
                    elif my_line_list[0] == 'distance':
                        defaults['distance'] = int(my_line_list[2].strip('\n'))
                    elif my_line_list[0] == 'likes':
                        defaults['likes'] = int(my_line_list[2].strip('\n'))
                    else:
                        defaults[my_line_list[0]] = my_line_list[2].strip('\n')

    except FileNotFoundError:
        print('No config.txt found')
        print('You must create a config.txt file as specified in the README')
        # create_new_config = input('Would you like us to create a
        # new config.txt file? (y,n) : ')
        # if create_new_config == 'y' or create_new_config == 'Y':
        #     print('Creating a new config...')

    # parse the supplied arguments
    args = parse_arguments(sys.argv[1:], defaults)

    if defaults['facebook_token'] is None:
        raise('ERROR: No facebook token in config.txt. You must supply a '
              'facebook token in order to use tindetheus!')

    # run the main function with parsed arguments
    main(args, defaults['facebook_token'])


if __name__ == '__main__':
    command_line_run()
