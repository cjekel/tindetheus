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

For a more detailed description of how and why this works see
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

Installation and getting started guide now stored in
`GETTING_STARTED.md <https://github.com/cjekel/tindetheus/blob/master/GETTING_STARTED.md>`__

config.txt
==========

You can now store all default optional parameters in the config.txt!
This means you can set your starting distance, number of likes, and
image_batch size without manually specifying the options each time. This
is an example config.txt file:

::

   facebook_token = XXXXXXX  # your facebook token hash
   # alternatively you can use the XAuthToken
   # XAuthToken = xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx
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
saved in validation.csv. For more information about the validate
function `read
this <https://github.com/cjekel/tindetheus/blob/master/VALIDATE_GUIDE.md>`__.

Changelog
=========

All changes now stored in
`CHANGELOG.md <https://github.com/cjekel/tindetheus/blob/master/CHANGELOG.md>`__

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
