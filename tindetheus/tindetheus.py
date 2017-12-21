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
# from facenet below.

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
import os
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

import export_embeddings
import tindetheus_align

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

def to_rgb1(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def download_url_photos(urls,userID):
#   define a function which downloads the pictures of urls
    count = 0
    image_list = []
    for url in urls:
        image_list.append('temp_images/'+userID+'.'+str(count)+'.jpg')
        urlretrieve(url, image_list[-1])
        count+=1
    return image_list

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
            temp_image = to_rgb1(temp_image)
        plt.subplot(n_row, n_col, j+1)
        plt.imshow(temp_image)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show(block=False)
    plt.pause(0.1)

    # plt.pause(0.001)
    # plt.draw()

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
    def __init__(self, facebook_id, facebook_token, likes_left=100):
        self.session = self.login(facebook_id, facebook_token)
        self.likes_left = likes_left
        #   set your search distance in miles
        self.search_distance = 5
        self.session.profile.distance_filter = self.search_distance

        # ensure that there is a temp_images dir
        if not os.path.exists('temp_images'):
            os.makedirs('temp_images')
        if not os.path.exists('database/like'):
            os.makedirs('database/like')
        if not os.path.exists('database/dislike'):
            os.makedirs('database/dislike')

        # attempt to load database
        try:
            self.database = list(np.load('database.npy'))
            print('You have browsed', len(self.database), 'Tinder profiles.')
        except:
            self.database = []

    def login(self, facebook_id, facebook_token):
        session = pynder.Session(facebook_token)
        print('Hello ', session.profile)
        return session

    def look_at_users(self, users):
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

        while self.likes_left > 0:
            try:
                users = self.session.nearby_users()
                self.look_at_users(users)
            except RecsTimeout:
                    self.search_distance+=5
                    self.session.profile.distance_filter = self.search_distance
                    self.browse()

# set path for security
sys.path.append(r'C:\Users\cj\Documents\run_tin')

def create_new_config():
    print('test')





def main(args, facebook_id, facebook_token):

    if args.function == 'browse':
        my_sess = client(facebook_id,facebook_token)
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

    # print(args)
    # sleep(random.random())
    # output_dir = os.path.expanduser(args.output_dir)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # # Store some git revision info in a text file in the log directory
    # src_path,_ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    # dataset = facenet.get_dataset(args.input_dir)

'''
There are three function choices: browse, build, like
browse: review new tinder profiles and store them in your database
build: use machine learning to create a new model that likes and dislikes profiles based on your historical preference
like: use your machine leanring model to like new tinder profiles
'''

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='There are three function choices: browse, build, or like')
    #
    # parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    # parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    # parser.add_argument('--image_size', type=int,
    #     help='Image size (height, width) in pixels.', default=182)
    # parser.add_argument('--margin', type=int,
    #     help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    # parser.add_argument('--random_order',
    #     help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    # parser.add_argument('--gpu_memory_fraction', type=float,
    #     help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    # parser.add_argument('--detect_multiple_faces', type=bool,
    #                     help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':

    # check for a config file
    try:
        with open('config.txt') as f:
            lines = f.readlines()
            facebook_token = lines[0].split(' ')[-1].strip()
            facebook_id = lines[1].split(' ')[-1].strip()
            # print('token:', facebook_token)
            # print('id:', facebook_id)

    except:
        print('No config.txt found')
        create_new_config = input('Would you like us to create a new config.txt file? (y,n) : ')
        if create_new_config == 'y' or create_new_config == 'Y':
            print('Creating a new config...')

    main(parse_arguments(sys.argv[1:]), facebook_id, facebook_token)
