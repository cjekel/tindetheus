# MIT License
#
# Copyright (c) 2017 Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The facenet implmentation has been hard coded into tindetheus. This has been
# hardcoded into tindetheus for the following reasons: 1) there is no setup.py
# for facenet yet. 2) to prevent changes to facenet from breaking tindetheus.
#
# facenet is used to align the database, crop the faces in database, and
# to calculate the embeddings for the database. I've included the copyright
# from facenet below. The specific code that is in this file from facenet
# is within the like_or_dislike_users(self, users) function.

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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
import os, shutil
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

# def to_rgb1(im):
#     # convert from grayscale to rgb
#     w, h = img.shape
#     ret = np.empty((w, h, 3), dtype=np.uint8)
#     ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
#     return ret

def clean_temp_images():
    # delete the temp_images dir
    shutil.rmtree('temp_images')
    os.makedirs('temp_images')

def clean_temp_images_aligned():
    # delete the temp images aligned dir
    shutil.rmtree('temp_images_aligned')

def download_url_photos(urls,userID,is_temp=False):
#   define a function which downloads the pictures of urls
    count = 0
    image_list = []
    if is_temp == True:
        os.makedirs('temp_images/temp')
    for url in urls:
        if is_temp ==True:
            image_list.append('temp_images/temp/'+userID+'.'+str(count)+'.jpg')
        else:
            image_list.append('temp_images/'+userID+'.'+str(count)+'.jpg')
        urlretrieve(url, image_list[-1])
        count+=1
    return image_list
def move_images_temp(image_list,userID):
    # move images from temp folder to al_database
    count = 0
    database_loc = []
    for i,j in enumerate(image_list):
        new_fname = 'al_database/'+userID+'.'+str(count)+'.jpg'
        os.rename(j,new_fname)
        database_loc.append(new_fname)
        count+=1
    return database_loc

def move_images(image_list,userID, didILike):
    # move images from temp folder to database
    if didILike == 'Like':
        fname = 'like/'
    else:
        fname = 'dislike/'
    count = 0
    database_loc = []
    for i,j in enumerate(image_list):
        new_fname = 'database/'+fname+userID+'.'+str(count)+'.jpg'
        os.rename(j,new_fname)
        database_loc.append(new_fname)
        count+=1
    return database_loc

def show_images(images):
    # use matplotlib to display profile images
    n = len(images)
    n_col = 3
    if n % n_col == 0:
        n_row =   n // n_col
    else:
        n_row = n // 3  + 1
    plt.figure()
    plt.tight_layout()
    for j,i in enumerate(images):
        temp_image = imageio.imread(i)
        if len(temp_image.shape) < 3:
            # needs to be converted to rgb
            temp_image = to_rgb(temp_image)
        plt.subplot(n_row, n_col, j+1)
        plt.imshow(temp_image)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show(block=False)
    plt.pause(0.1)


def calc_avg_emb():
    # a function to create a vector of 128 average embeddings for each
    # tinder profile
    # get the embeddings per profile
    labels = np.load('labels.npy')
    label_strings = np.load('label_strings.npy')
    embeddings = np.load('embeddings.npy')
    image_list = np.load('image_list.npy')

    # find the maximum number of images in a profile


    split_image_list = []
    profile_list = []
    for i in image_list:
        split_image_list.append(i.replace('/','.').split('.'))
        profile_list.append(split_image_list[-1][2])

    # conver profile list to pandas index
    pl = pd.Index(profile_list)
    pl_unique = pl.value_counts()

    # get the summar statics of pl
    pl_describe = pl_unique.describe()
    print('Summary statiscs of profiles with at least one detectable face')
    print(pl_describe)
    number_of_profiles = int(pl_describe[0])
    number_of_images = int(pl_describe[-1])

    # convert the emebeddigns to a data frame
    eb = pd.DataFrame(embeddings, index=pl)
    dislike = pd.Series(labels, index=pl)
    # if dislike == 1 it means I disliked the person!

    # create a blank numpy array for embeddings
    new_embeddings = np.zeros((number_of_profiles,128))
    new_labels = np.zeros(number_of_profiles)
    for i,j in enumerate(pl_unique.index):
        temp = eb.loc[j]

        # if a profile has more than one face it will be a DataFrame
        # else the profile will be a Series
        if isinstance(temp,pd.DataFrame):
            # get the average of each column
            temp_embedings = np.mean(temp.values, axis=0)
        else:
            temp_embedings = temp.values

        # save the new embeddings
        new_embeddings[i] = temp_embedings

        # Save the profile label, 1 for dislike, 0 for like
        new_labels[i] = dislike[j].max()

    # save the files
    np.save('embeddings_avg_profile.npy',new_embeddings)
    np.save('labels_avg_profile.npy',new_labels)
    return new_embeddings, new_labels

def calc_avg_emb_temp(embeddings):
    # a function to create a vector of 128 average embeddings for each
    # in the temp_images_aligned folder
    # embeddings = np.load('temp_embeddings.npy')

    # caluclate the average embeddings
    new_emeeddings = np.zeros((1,128))
    new_emeeddings[0] = np.mean(embeddings,axis=0)
    return new_emeeddings


def fit_log_reg(X,y):
    # fits a logistic regression model to your data
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)
    print('Train size: ', len(X))
    train_score = model.score(X,y)
    print('Training accuracy', train_score)
    ypredz = model.predict(X)
    cm = confusion_matrix(y, ypredz)
    tn, fp, fn, tp = cm.ravel()
    # true positive rate When it's actually yes, how often does it predict yes?
    recall = float(tp) / np.sum(cm,axis=1)[1]
    # Specificity: When it's actually no, how often does it predict no?
    specificity = float(tn) /np.sum(cm,axis=1)[0]

    print('Recall/ Like accuracy', recall)
    print('specificity/ Dislike accuracy', specificity)

    # save the model
    joblib.dump(model, 'log_reg_model.pkl')

#   define a function to get like or dislike input
def like_or_dislike():
    likeOrDislike = '0'
    while likeOrDislike != 'j' and likeOrDislike != 'l' \
            and likeOrDislike != 'f' and likeOrDislike != 's':

        likeOrDislike = input()
        if likeOrDislike == 'j' or likeOrDislike == 'f':
            return 'Dislike'
        elif likeOrDislike == 'l' or likeOrDislike == 's':
            return 'Like'
        else:
            print('you must enter either l or s for like, or j or f for dislike')
            likeOrDislike = input()

class client:
    # a class to manage the pynder api
    def __init__(self, facebook_token, distance, likes_left=100):
        self.session = self.login(facebook_token)
        self.likes_left = likes_left
        #   set your search distance in miles
        self.search_distance = distance
        self.session.profile.distance_filter = self.search_distance

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
            print('You have automatically liked or disliked ', len(self.al_database), 'Tinder profiles.')
        except:
            self.al_database = []


    def login(self, facebook_token):
        # login to Tinder using pynder
        session = pynder.Session(facebook_token)
        print('Hello ', session.profile)
        return session

    def look_at_users(self, users):
        # browse user profiles one at a time
        # you will be presented with the oppurtunity to like or dislike profiles
        # your history will be stored in a database that you can use for training
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
            image_list = download_url_photos(urls,user.id)
            show_images(image_list)

            didILike = like_or_dislike()
            plt.close('all')

            dbase_names = move_images(image_list,user.id, didILike)

            if didILike == 'Like':
                print(user.like())
                self.likes_left-=1
            else:
                print(user.dislike())
            userList = [user.id, user.name, user.age, user.bio, user.distance_km, user.jobs, user.schools, user.get_photos(width='640'), dbase_names, didILike]
            self.database.append(userList)
            np.save('database.npy',self.database)

    def like_or_dislike_users(self, users):
        # automatically like or dislike users based on your previously trained
        # model on your historical preference.

        # facenet settings from export_embeddings....
        model_dir='20170512-110547'
        data_dir='temp_images_aligned'
        embeddings_name='temp_embeddings.npy'
        labels_name='temp_labels.npy'
        labels_strings_name='temp_label_strings.npy'
        is_aligned=True
        image_size=160
        margin=44
        gpu_memory_fraction=1.0
        image_batch=1000
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load the facenet model
                facenet.load_model(model_dir)
                for user in users:
                    clean_temp_images()
                    urls = user.get_photos(width='640')
                    image_list = download_url_photos(urls,user.id,is_temp=True)
                    # align the database
                    tindetheus_align.main(input_dir='temp_images',
                                        output_dir='temp_images_aligned')
                    # export the embeddinggs from the aligned database

                    train_set = facenet.get_dataset(data_dir)
                    image_list_temp, label_list = facenet.get_image_paths_and_labels(train_set)
                    # fetch the classes (labels as strings) exactly as it's done in get_dataset
                    path_exp = os.path.expanduser(data_dir)
                    classes = [path for path in os.listdir(path_exp) \
                        if os.path.isdir(os.path.join(path_exp, path))]
                    classes.sort()
                    # get the label strings
                    label_strings = [name for name in classes if \
                        os.path.isdir(os.path.join(path_exp, name))]

                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

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
                        if i == nrof_batches -1:
                            n = nrof_images
                        else:
                            n = i*batch_size + batch_size
                        # Get images for the batch
                        if is_aligned is True:
                            images = facenet.load_data(image_list_temp[i*batch_size:n], False, False, image_size)
                        else:
                            images = load_and_align_data(image_list_temp[i*batch_size:n], image_size, margin, gpu_memory_fraction)
                        feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                        # Use the facenet model to calcualte embeddings
                        embed = sess.run(embeddings, feed_dict=feed_dict)
                        emb_array[i*batch_size:n, :] = embed
                        print('Completed batch', i+1, 'of', nrof_batches)

                    run_time = time.time() - start_time
                    print('Run time: ', run_time)

                    #   export emedings and labels
                    label_list  = np.array(label_list)

                    np.save(embeddings_name, emb_array)

                    if emb_array.size > 0:
                        # calculate the 128 average embedding per profiles
                        X = calc_avg_emb_temp(emb_array)
                        # ealuate on the model
                        yhat = self.model.predict(X)

                        if yhat[0] == 1:
                            didILike = 'Like'
                        else:
                            didILike = 'Dislike'
                    else:
                        # there were no faces in this profile
                        didILike = 'Dislike'
                    print('********************************************************')
                    print(user.name, user.age, didILike)
                    print('********************************************************')

                    dbase_names = move_images_temp(image_list, user.id)

                    if didILike == 'Like':
                        print(user.like())
                        self.likes_left-=1
                    else:
                        print(user.dislike())
                    userList = [user.id, user.name, user.age, user.bio, user.distance_km, user.jobs, user.schools, user.get_photos(width='640'), dbase_names, didILike]
                    self.al_database.append(userList)
                    np.save('al_database.npy',self.al_database)
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
                stayOrQuit  = input()
                if stayOrQuit == 'l' or stayOrQuit == 's':
                    # if self.search_distance < 100:
                    self.search_distance+=5
                    self.session.profile.distance_filter = self.search_distance
                    self.browse()
                else:
                    break

    def like(self):
        # like and dislike Tinder profiles using your trained logistic
        # model. Note this requires that you frist run tindetheus browse to
        # build a database. Then run tindetheus train to train a model.

        # load the pretrained model
        self.model = joblib.load('log_reg_model.pkl')

        while self.likes_left > 0:
            try:
                users = self.session.nearby_users()
                self.like_or_dislike_users(users)
            except RecsTimeout:
                    self.search_distance+=5
                    self.session.profile.distance_filter = self.search_distance
                    self.like()


def main(args, facebook_token):
# There are three function choices: browse, build, like
# browse: review new tinder profiles and store them in your database
# train: use machine learning to create a new model that likes and dislikes profiles based on your historical preference
# like: use your machine leanring model to like new tinder profiles
    if args.function == 'browse':
        my_sess = client(facebook_token, args.distance)
        my_sess.browse()

    elif args.function == 'train':
        # align the database
        tindetheus_align.main()
        # export the embeddinggs from the aligned database
        export_embeddings.main()
        # calculate the 128 average embedding per profiles
        X, y = calc_avg_emb()
        # fit and save a logistic regression model to the database
        fit_log_reg(X,y)

    elif args.function == 'like':
        my_sess = client(facebook_token, args.distance)
        my_sess.like()

    else:
        text = '''You must specify a function. Your choices are either
tindetheus browse
tindetheus train
tindetheus like'''
        print(text)



def parse_arguments(argv):
    help_text = '''There are three function choices: browse, train, or like.

1) tinetheus browse
-- Let's you browse tinder profiles to add to your database.
-- Browses tinder profiles in your distance until you run out.
-- Asks if you'd like to increase the distance by 5 miles.
-- Use to build a database of the tinder profiles you look at.

2) tindetheus train
-- Trains a model to your Tinder database.
-- Uses facenet implemnation for facial detection and classifcation.
-- Saves logisitc regression model to classify which faces you like and dislike.

3) tindetheus like
-- Automatically like and dislike Tinder profiles based on your historical preference.
-- First run browse, then run train, then prosper with like.
-- Uses the trained model to automaticlaly like and dislike profiles.
-- Profiles where a face isn't detected are automatically disliked.
'''
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help=help_text)
    parser.add_argument('--distance', type=int,
        help='Set the starting distance in miles. Tindetheus will crawl in 5 mile increments from here.', default=5)
    return parser.parse_args(argv)

def command_line_run():
    # check for a config file
    try:
        with open('config.txt') as f:
            lines = f.readlines()
            facebook_token = lines[0].split(' ')[-1].strip()


    except:
        print('No config.txt found')
        print('You must create a config.txt file as specified in the README')
        # create_new_config = input('Would you like us to create a new config.txt file? (y,n) : ')
        # if create_new_config == 'y' or create_new_config == 'Y':
        #     print('Creating a new config...')

    main(parse_arguments(sys.argv[1:]), facebook_token)

if __name__ == '__main__':
    command_line_run
