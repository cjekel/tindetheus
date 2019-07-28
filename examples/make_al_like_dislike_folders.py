import shutil
import os
import numpy as np


# This script is what tindetheus like_folder does


def al_copy_images(image_list, userID, didILike, database_str='al/'):
    # move images from temp folder to database
    if didILike == 'Like':
        fname = 'like/'
    else:
        fname = 'dislike/'
    count = 0
    database_loc = []
    for i, j in enumerate(image_list):
        new_fname = database_str+fname+userID+'.'+str(count)+'.jpg'

        shutil.copyfile(j, new_fname)

        count += 1
    return database_loc


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
    al_copy_images(user[8], user[0], user[-1])
