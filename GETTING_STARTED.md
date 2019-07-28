# Installation and Getting started

1. Create a new folder that will be your Tinder database.
```bash
mkdir tinder
cd tinder
```

2. You need your facebook auth token. There are many discussions on this on the internet to find this. You can find your facebook auth token by using a man in the middle (MIM) attack to sniff out the requests. You are looking for *access_token=*. The MIM attack can be conducted by creating a proxy with ssl certificate. If you are still lost, perhaps check out [this](https://gist.github.com/rtt/10403467) or [this](http://www.joelotter.com/2015/05/17/dj-khaled-tinder-bot.html).

3. Create a config.txt file that contains the following line exactly
```
facebook_token = YYYY
```
where YYYY is replaced with your facebook token in order to login using pynder. Alternatively you can use *XAuthToken = xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx* instead of facebook_token.

4. You need to initialize git in your *tinder* folder which is used to track revision history. Run the following commands to initialize git.
```bash
git init
git add .
git commit -m "first commit"
```
5. Choose between a docker container or native setup for tindetheus. I'd highly recommend using the docker container as this is a dependency heavy library, but tindetheus will work either way you choose!

- [docker setup](#docker-setup)
- [native setup](#native-setup)

## docker setup

1. Add the *model_dir* line to the the config.txt file exactly as below.
```
facebook_token = YYYY
model_dir = /models/20170512-110547
```
The docker container includes a pretrained facenet model (for more information read step 3 of [native setup](#native-setup)). You are welcome to experiment with other pretrained facenet models.

2. Get the docker container.
```bash
docker pull cjekel/tindetheus
```

3. Run the docker container while mounting the *tinder* directory to */mnt/tinder*
```bash
docker run -it -v /home/cj/tinder/:/mnt/tinder cjekel/tindetheus
```
In this case */home/cj/tinder/* is the location of my *tinder* folder on my host machine. You should see something like the following when you run the docker container.
```bash
root@c4771abc41i9:/# 
```

4. cd into the mounted tinder folder
```bash
root@c4771abc41i9:/# cd /mnt/tinder
```

5. Start building your database. Manually reviewing 20-40 profiles will be a good starting point, but you can do it with less. Before you start training a model you have to be sure that you've liked and disliked at leach one profile.
```bash
tindetheus browse
```
The profile images will show up in *tinder/temp_images*. To view these images open *tinder/temp_images* in the file explore on your host machine. This works best with large grid icons. Follow the command line instructions to like or dislike the profile.

6. Continue to [further instructions](#further-instructions)

## native setup

If you use Windows you may want to read this guide on [how to install tindetheus on Windows](http://jekel.me/2018/How-to-install-tindetheus-on-windows-10-to-automatically-like-users-on-tinder/).

1. Install my pynder PR from source (pynder on pip has not been updated)
```bash
git clone https://github.com/charliewolf/pynder.git
cd pynder
git fetch origin +refs/pull/211/merge
git checkout -qf FETCH_HEAD
[sudo] python -m pip install .
```

2. Install tindetheus
```bash
[sudo] pip install tindetheus
```

3. Download a pretrained facenet model. I recommend using this model [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) [mirror](https://mega.nz/#!d6gxFL5b!ZLINGZKxdAQ-H7ZguAibd6GmXFXCcr39XxAvIjmTKew). You must download 20170512-110547.zip and extract the contents in your *tinder* folder. The contents will be a folder named 20170512-110547. You should specify the pretrained model that you use in the second line of the config.txt tile. You can use other [pretrained facenet models](https://github.com/davidsandberg/facenet#pre-trained-models) as long as you include the model directory in your folder and change the config.txt accordingly. 

4. Start building your database. Manually reviewing 20-40 profiles will be a good starting point, but you can do it with less. Before you start training a model you have to be sure that you've liked and disliked at leach one profile.
```bash
tindetheus browse
```
The profile images will show up in a window. Follow the command line instructions to like or dislike the profile.

## further instructions

5. After browsing profiles you can train your personalized classification model at any time. (Make sure you have liked and disliked at least one profile each before running!) Just run
```bash
tindetheus train
```
to build your personalized model. With more profiles you can build a more accurate model, so feel free to browse more profiles at any time and build to your database. Newly browsed profiles aren't automatically added to the model, so you must manually run tindetheus train to update your model.

6. You can automatically like and dislike profiles based on your trained model. To do this simply run
```bash
tindetheus like
```
which will use your latest trained model to automatically like and dislike profiles. The application will start with a 5 mile search radius, and automatically like and dislike the people in this radius. After running out of people, the search radius is increased by 5 miles and the processes repeats. This goes on until you've used 100 likes, at which point the application stops.

7. This is all in the early stages, so after each session I highly recommend you backup your *tinder* folder by creating an archive of the folder.

8. If you want to manually browse your database, check out this [example](https://github.com/cjekel/tindetheus/blob/master/examples/open_database.py) file.