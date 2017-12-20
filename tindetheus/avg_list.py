from __future__ import print_function
import numpy as np
import pandas as pd

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
    profile_list.append(split_image_list[-1][7])

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
