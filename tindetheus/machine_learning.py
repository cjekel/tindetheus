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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import input

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib


def calc_avg_emb():
    # a function to create a vector of n average embeddings for each
    # tinder profile
    # get the embeddings per profile
    labels = np.load('labels.npy', allow_pickle=True)
    # label_strings = np.load('label_strings.npy', allow_pickle=True)
    embeddings = np.load('embeddings.npy', allow_pickle=True)
    image_list = np.load('image_list.npy', allow_pickle=True)

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
    # embeddings = np.load('temp_embeddings.npy', allow_pickle=True)
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
