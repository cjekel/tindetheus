# MIT License
#
# Copyright (c) 2017-2019 Charles Jekel
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

import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib

from tindetheus import export_embeddings
from tindetheus import tindetheus_align
from tindetheus.tinder_client import client
import tindetheus.facenet_clone.facenet as facenet
import tindetheus.image_processing as imgproc
import tindetheus.machine_learning as ml

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# add version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()


def main(args, facebook_token, x_auth_token=None):
    # There are three function choices: browse, build, like
    # browse: review new tinder profiles and store them in your database
    # train: use machine learning to create a new model that likes and dislikes
    # profiles based on your historical preference
    # like: use your machine leanring model to like new tinder profiles
    if args.function == 'browse':
        my_sess = client(facebook_token, args.distance, args.model_dir,
                         likes_left=args.likes, x_auth_token=x_auth_token)
        my_sess.browse()

    elif args.function == 'train':
        # align the database
        tindetheus_align.main()
        # export the embeddings from the aligned database
        export_embeddings.main(model_dir=args.model_dir,
                               image_batch=args.image_batch)
        # calculate the n average embedding per profiles
        X, y = ml.calc_avg_emb()
        # fit and save a logistic regression model to the database
        ml.fit_log_reg(X, y)

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
        # if this is backwards, there is probably a bug...
        dislikes = yhat == 0
        likes = yhat == 1
        imgproc.show_images(image_list[dislikes], holdon=True, title='Dislike')
        print('\n\nGenerating plots...\n\n')
        plt.title('Dislike')

        imgproc.show_images(image_list[likes], holdon=True, title='Like')
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
                                 likes_left=args.likes, tfsess=sess,
                                 x_auth_token=x_auth_token)
                # Load the facenet model
                facenet.load_model(my_sess.model_dir)
                print('Facenet model loaded successfully!!!')
                # automatically like users
                my_sess.like()
    elif args.function == 'like_folder':
        print('Copying al_database profiles into either al/like or al/dislike')
        # make folders
        if not os.path.exists('al'):
            os.makedirs('al')
        if not os.path.exists('al/like'):
            os.makedirs('al/like')
        if not os.path.exists('al/dislike'):
            os.makedirs('al/dislike')

        # load the auto like database
        al_data = np.load('al_database.npy', allow_pickle=True)

        # copy profile images to either al/like or al/dislike
        for user in al_data:
            imgproc.al_copy_images(user[8], user[0], user[-1])

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
5) tindetheus like_folder
-- Creates al/like and al/dislike folders based on the profiles you have
-- automatically liked. This copies the profile images from al_database
-- into al/like or al/disliked based on whether the model liked or
-- disliked the profile.
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
                'XAuthToken': None,
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
                        defaults['image_batch'] = int(my_line_list[2].strip('\n'))  # noqa E501
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

    if defaults['facebook_token'] is None and defaults['XAuthToken'] is None:
        raise('ERROR: No facebook token nor XAuth token in config.txt. '
              'You must supply a facebook token in order to use tindetheus!')

    # run the main function with parsed arguments
    main(args, defaults['facebook_token'], defaults['XAuthToken'])


if __name__ == '__main__':
    command_line_run()
