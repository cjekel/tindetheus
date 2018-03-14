tindetheus
==========

Build personalized machine learning models for Tinder based on your
historical preference using Python.

There are three parts to this: 1. A function to build a database which
records everything about the profiles you've liked and disliked. 2. A
function to train a model to your database. 3. A function to use the
trained model to automatically like and dislike new profiles.

How it works
============

Essentially you can use the last layer a CNN trained for facial
classification as a feature set that describes an individual's face. It
just so happens that this feature set is related to facial
attractiveness.

tindetheus let's you build a database based on the profiles that you
like and dislike. You can then train a classification model to your
database. The model training first uses a MTCNN to detect and box the
faces in your database. Then a facenet model is run on the faces to
extract the embeddings (last layer of the CNN). A logistic regression
model is then fit to the embeddings. The logistic regression model is
saved, and this processes is repeated in automation to automatically
like and dislike profiles based on your historical preference.

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

    tindethus browse --distance=20

by default tindetheus starts with a 5 mile radius, but you can specify a
search distance by specifying --distance. The above example is to start
with a 20 mile search radius. It is important to note that when you run
out of nearby users, tindethesus will ask you if you'd like to increase
the search distance by 5 miles.

.. code:: bash

    tindetheus train

Use machine learning to build a personalized model of who you like and
dislike based on your database. The more profiles you've browsed, the
better your model will be.

.. code:: bash

    tindetheus like

Use your personalized model to automatically like and dislike profiles.
The profiles which you have automatically liked and disliked are stored
in al\_database. By default this will start with a 5 mile search radius,
which increases by 5 miles until you've used 100 likes. You can change
the default search radius by using

.. code:: bash

    tindetheus like --distance=20

which would start with a 20 mile search radius.

Installation
============

If you use Windows you may want to read this guide on `how to install
tindetheus on
Windows <http://jekel.me/2018/How-to-install-tindetheus-on-windows-10-to-automatically-like-users-on-tinder/>`__.

1. Install pynder from source (pynder on pip has not been updated)

   .. code:: bash

       git clone https://github.com/charliewolf/pynder.git
       [sudo] pip install ./pynder

2. Install tindetheus

   .. code:: bash

       [sudo] pip install tindetheus

Getting started
===============

1. After you have installed tindetheus. Create a new folder that will be
   your Tinder database.

   .. code:: bash

       mkdir my_tinder_data
       cd my_tinder_data

2. You need your facebook auth token. There are many discussions on this
   on the internet to find this. You can find your facebook auth toeken
   by using a man in the middle (MIM) attack to sniff out the requests.
   You are looking for *access\_token=*. The MIM attack can be conducted
   by creating a proxy with ssl certiticate. If you are still lost,
   perhaps check out `this <https://gist.github.com/rtt/10403467>`__ or
   `this <http://www.joelotter.com/2015/05/17/dj-khaled-tinder-bot.html>`__.
3. Create a config.txt file that contains the following line exactly

   ::

       facebook_token = YYYY

   where YYYY is replaced with your facebook token in order to login
   using pynder.

4. Download a pretrained facenet model. I recommend using this model
   `20170512-110547 <https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit>`__
   `mirror <https://mega.nz/#!d6gxFL5b!ZLINGZKxdAQ-H7ZguAibd6GmXFXCcr39XxAvIjmTKew>`__.
   You must download 20170512-110547.zip and extract the contents in
   your my\_tinder\_data folder. The contents will be a folder named
   20170512-110547. You can use other `pretrained facenet
   models <https://github.com/davidsandberg/facenet>`__ as long as you
   rename the folder to 20170512-110547.

5. You need to initialize git in your my\_tinder\_data folder which is
   used to track revision history. Run the following commands to
   initialize git.

   .. code:: bash

       git init
       git add .
       git commit -m "first commit"

6. Start building your database. Manually reviewing 20-40 profiles will
   be a good starting point, but you can do it with less. Before you
   start training a model you have to be sure that you've liked and
   disliked at leach one profile.

   .. code:: bash

       tindetheus browse

7. After browsing profiles you can train your personalized classifcation
   model at any time. Just run

   .. code:: bash

       tindetheus train

   to build your personalized model. With more profiles you can build a
   more accurate model, so feel free to browse more profiles at any time
   and build to your database. Newly browsed profiles aren't
   automatically added to the model, so you must manually run tinetheus
   train to update your model.

8. You can automatically like and dislike profiles based on your trained
   model. To do this simply run

   .. code:: bash

       tindetheus like

   which will use your latest trained model to automatically like and
   dislike profiles. The application will start with a 5 mile search
   radius, and automatically like and dislike the people in this radius.
   After running out of people, the search radius is increased by 5
   miles and the processes repeats. This goes on until you've used 100
   likes, at which point the application stops.

9. This is all in the early stages, so after each session I highly
   recommend you backup your my\_tinder\_data folder by creating an
   archive of the folder.

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
Prometheus signifies "forethought," while his brother Epimetheus denotes
"afterthought". In synergy they serve to improve your Tinder experience.

Epimetheus creates a database from all of the profiles you review on
Tinder.

Prometheus learns from your historical preferences to automatically like
new Tinder profiles.
