tindetheus
==========

Build personalized machine learning models for Tinder based on your
historical preference using Python.

There are three parts to this: 1. A function to build a database which
records everything about the profiles you’ve liked and disliked. 2. A
function to train a model to your database. 3. A function to use the
trained model to automatically like and dislike new profiles.

How it works
============

The last layer of a CNN trained for facial classification can be used as
a feature set which describes an individual’s face. It just so happens
that this feature set is related to facial attractiveness.

tindetheus let’s you build a database based on the profiles that you
like and dislike. You can then train a classification model to your
database. The model training first uses a MTCNN to detect and box the
faces in your database. Then a facenet model is run on the faces to
extract the embeddings (last layer of the CNN). A logistic regression
model is then fit to the embeddings. The logistic regression model is
saved, and this processes is repeated in automation to automatically
like and dislike profiles based on your historical preference.

.. figure:: https://raw.githubusercontent.com/cjekel/tindetheus/master/examples/how_does_tindetheus_work.png
   :alt: Visual aid explaining tindetheus

   Visual aid explaining tindetheus

This `blog
post <http://jekel.me/2018/Using-facenet-to-automatically-like-new-tinder-profiles/>`__
has a short description of how tindetheus works.

For a more detailed description of how and why this works see [1]
https://arxiv.org/abs/1803.04347

Example usage
=============

.. code:: bash

   tindetheus browse

build a database by liking and disliking profiles on Tinder. The
database contains all the profile information as a numpy array, while
the profile images are saved in a different folder.

.. code:: bash

   tindetheus browse --distance=20

by default tindetheus starts with a 5 mile radius, but you can specify a
search distance by specifying –distance. The above example is to start
with a 20 mile search radius. It is important to note that when you run
out of nearby users, tindethesus will ask you if you’d like to increase
the search distance by 5 miles.

.. code:: bash

   tindetheus train

Use machine learning to build a personalized model of who you like and
dislike based on your database. The more profiles you’ve browsed, the
better your model will be.

.. code:: bash

   tindetheus like

Use your personalized model to automatically like and dislike profiles.
The profiles which you have automatically liked and disliked are stored
in al_database. By default this will start with a 5 mile search radius,
which increases by 5 miles until you’ve used 100 likes. You can change
the default search radius by using

.. code:: bash

   tindetheus like --distance=20

which would start with a 20 mile search radius.

Installation and Getting started
================================

Instructions `here
<https://github.com/cjekel/tindetheus/blob/master/README.md#installation-and-getting-started/>`__

config.txt
==========

You can now store all default optional parameters in the config.txt!
This means you can set your starting distance, number of likes, and
image_batch size without manually specifying the options each time. This
is an example config.txt file:

::

   facebook_token = XXXXXXX  # your facebook token hash
   # alternatively you can use the XAuthToken
   XAuthToken = xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx
   model_dir = 20170512-110547  # the location of your facenet model directory
   # see https://github.com/davidsandberg/facenet#pre-trained-models for other
   # pretrained facenet models
   image_batch = 1000  # number of images to load in a batch during train
   #  the larger the image_batch size, the faster the training process, at the
   #  cost of additional memory. A 4GB machine may struggle with 1000 images.
   distance = 5  # Set the starting distance in miles
   likes = 100  # set the number of likes you want to use
   #  note that free Tinder users only get 100 likes in 24 hours

Using the validate function on a different dataset
==================================================

As of Version 0.4.0, tindetheus now includes a validate function. This
validate functions applies your personally trained tinder model on an
external set of images. If there is a face in the image, the model will
predict whether you will like or dislike this face. The results are
saved in validation.csv.

First you’ll need to get a validation data set. I’ve created a small
subset of the `hot or not
database <http://vision.cs.utexas.edu/projects/rationales/>`__ for
testing purposes. You can download the validation.zip
`here <https://drive.google.com/file/d/13cNUzP_eXKsq8ABHwXHn4b9UgRbk-5oP/view?usp=sharing>`__
which is a a subset of the female images in [2], and extract it to your
tinder database directory.

Then execute

::

   tindetheus validate

to run the pretrained tindetheus model on your validation image set. You
could run the tindetheus trained model on the entire hot or not database
to give you an idea of how your model reacts in the wild. Note that
validate will attempt to rate each face in your image database, while
tindetheus only considers the images with just one face.

The validate function only looks at images within folders in the
validation folder. All images directly within the validation folder will
be ignored. The following directory structure considers the images in
the validation/females and validation/movie_stars directories.

::

   my_tinder_project
   │   config.txt
   |   validation.csv
   │
   └───validation
   |   |   this_image_ignored.jpg
   │   │
   │   └───females
   │   │   │   image00.jpg
   │   │   │   image01.jpg
   │   │   │   ...
   │   └───movie_stars
   │       │   image00.jpg
   │       │   image01.jpg
   │       │   ...

News
====

-  2019/06/23 Version 0.4.6. Add docker container instructions. Update
   readme.md instructions. Bugfix python 2.7 command line parsing.
-  2019/05/05 Version 0.4.3. Add option to log in using XAuthToken
   thanks to charlesduponpon. Add like_folder command line option to
   create al/like and al/dislike folders based on the historically liked
   and disliked profiles. Allows quick access to asses model quality.
-  2019/04/29 Version 0.4.1. Fix issue where line endings that were
   causing authentication failure. Fix handling of config.txt.
-  2018/12/02 Version 0.4.0. New validate function to apply your
   tindetheus model to a new dataset. See README on how to use this
   function. Fix issues with lossy integer conversions. Some other small
   bug fixes.
-  2018/11/25 Version 0.3.3. Update how facenet TensorFlow model is
   based into object. Fixes session recursion limit.
-  2018/11/04 Version 0.3.1. Fix bug related to Windows and
   calc_avg_emb(), which wouldn’t find the unique classes. Version
   0.3.2, tindetheus will now exit gracefully if you have used all of
   your free likes while running tindetheus like.
-  2018/11/03 Version 0.3.0. Major refresh. Bug fix related to calling a
   tindetheus.export_embeddings function. Added version tracking and
   parser with –version. New optional parameters: likes (set how many
   likes you have remaining default=100), and image_batch (set the
   number of images to load into facenet when training default=1000).
   Now all optional settings can be saved in config.txt. Saving the same
   filename in your database no longer bombs out on Windows. Code should
   now follow pep8.
-  2018/05/11 Added support for latest facenet models. The different
   facenet models don’t appear to really impact the accuracy according
   to `this
   post <https://jekel.me/2018/512_vs_128_facenet_embedding_application_in_Tinder_data/>`__.
   You can now specify which facenet model to use in the config.txt
   file. Updated facenet clone implementation. Now requires minimum
   tensorflow version of 1.7.0. Added
   `example <https://github.com/cjekel/tindetheus/blob/master/examples/open_database.py>`__
   script for inspecting your database manually.

Open source libraries
=====================

tindetheus uses the following open source libraries:

-  `pynder <https://github.com/charliewolf/pynder>`__
-  `facenet <https://github.com/davidsandberg/facenet>`__
-  `numpy <http://www.numpy.org/>`__
-  `matplotlib <https://matplotlib.org/>`__
-  `scikit-learn <http://scikit-learn.org/stable/>`__
-  `tensorflow <https://www.tensorflow.org/>`__
-  `imageio <https://imageio.github.io/>`__
-  `pandas <http://pandas.pydata.org/>`__

About the name
==============

Tindetheus is a combination of Tinder (the popular online dating
application) and the Greek Titans:
`Prometheus <https://en.wikipedia.org/wiki/Prometheus>`__ and
`Epimetheus <https://en.wikipedia.org/wiki/Epimetheus_(mythology)>`__.
Prometheus signifies “forethought,” while his brother Epimetheus denotes
“afterthought”. In synergy they serve to improve your Tinder experience.

Epimetheus creates a database from all of the profiles you review on
Tinder.

Prometheus learns from your historical preferences to automatically like
new Tinder profiles.

References
==========

[1] Jekel, C. F., & Haftka, R. T. (2018). Classifying Online Dating
Profiles on Tinder using FaceNet Facial Embeddings. arXiv preprint
arXiv:1803.04347.

[2] Donahue, J., & Grauman, K. (2011). Annotator rationales for visual
recognition. http://vision.cs.utexas.edu/projects/rationales/
