# Using the validate function on a different dataset
As of Version 0.4.0, tindetheus now includes a validate function. This validate functions applies your personally trained tinder model on an external set of images. If there is a face in the image, the model will predict whether you will like or dislike this face. The results are saved in validation.csv. For more information about the validate function [read this].

First you'll need to get a validation data set. I've created a small subset of the [hot or not database](http://vision.cs.utexas.edu/projects/rationales/) for testing purposes. You can download the validation.zip [here](https://drive.google.com/file/d/13cNUzP_eXKsq8ABHwXHn4b9UgRbk-5oP/view?usp=sharing) which is a a subset of the female images in [1], and extract it to your tinder database directory. 

Then execute
```
tindetheus validate
```
to run the pretrained tindetheus model on your validation image set. You could run the tindetheus trained model on the entire hot or not database to give you an idea of how your model reacts in the wild. Note that validate will attempt to rate each face in your image database, while tindetheus only considers the images with just one face.

The validate function only looks at images within folders in the validation folder. All images directly within the validation folder will be ignored. The following directory structure considers the images in the validation/females and validation/movie_stars directories.

```
my_tinder_project
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
```

# References

[1] Donahue, J., & Grauman, K. (2011). Annotator rationales for visual recognition. http://vision.cs.utexas.edu/projects/rationales/
