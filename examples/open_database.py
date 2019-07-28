from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tindetheus

# you can copy this script into your my_tinder_data folder

# load the database

# database.npy is the database for the users that you
# have manually liked

# al_database.npy is the database for the users that you
# have automatically liked

database = np.load('database.npy', allow_pickle=True)

n = len(database)

print('You have ', n, ' profiles in your data base')

# each row in database is the information for each user
# the columns contain the following information
# [user.id, user.name, user.age, user.bio, user.distance_km, user.jobs,
#  user.schools, photos_urls, dbase_names, didILike]

# to dislay everything about the first user
print('User ID:', database[0][0])
print('Name:', database[0][1])
print('Age:', database[0][2])
print('Bio:', database[0][3])
print('Distance (km):', database[0][4])
print('Jobs:', database[0][5])
print('School:', database[0][6])
print('Liked or Disliked:', database[0][9])

# to open the first user's profile images run
tindetheus.image_processing.show_images(database[0][8])

# How to loop through each user in the database
# for user in database:
#     # print the names in the database
#     print(user[1])
#     # display their profile images
#     tindetheus.show_images(user[8])

# to grab all the names
names = database[:, 1]

# have fun!
