# Changelog
All notable changes to this project will be documented in this file.

## [0.5.4] - 2019-11-27
## Added
- retries option for tindetheus like or browse command, default to 20, and can be modified in config.txt
## Changed
- cleaned up some code for parsing config.txt, seperated interger and string types
- setup.py now lists ```tensorflow < 2.0.0``` as a requirement

## [0.5.3] - 2019-11-17
### Changed
- skimage.transform.resize now uses defaults

## [0.5.2] - 2019-11-13
### Added
- setuptools now shows up in the requirements

## [0.5.1] - 2019-11-07
### Changed
- fixed a bug that would cause tindetheus not to run if a line in config.txt was a list of 2 items when separated by a space
- switched to setuptools, remove README.rst

## [0.5.0] - 2019-07-28
### Added
- Basic travis.ci checking for installation on python 2.7, 3.5, 3.6 and flake8 on files
- Reorganize tindetheus functions into submodules image_processing, machine_learning, tinder_client
### Changed
- Bugfix python 2.7 installation
- Bugfix related to try except that was overriding the database folder on newer numpy distributions
- Moved installation instructions, changelog, and validate function details to separate files
- numpy.load now uses allow_pickle=True
### Removed
- Remove function from tindetheus into submodules image_processing, machine_learning, tinder_client

## [0.4.7] - 2019-07-27
### Changed
- Fix installation on Windows
- Fix loading MTCNN weights on newer version of numpy

## [0.4.6] - 2019-06-23
### Added
- Docker container instructions.
### Changed
- Readme notes 
- Bugfix related to Python 2.7 command line parsing

## [0.4.3] - 2019-05-05
### Added
- Add option to log in using XAuthToken thanks to charlesduponpon
- Add like_folder command line option to create al/like and al/dislike folders based on the historically liked and disliked profiles (this is a quick way to assess model quality)

## [0.4.1] - 2019-04-29
### Changed
- Fix issue where line endings that were causing authentication failure
- Fix handling of config.txt

## [0.4.0] - 2018-12-02
### Added
- New validate function to apply your tindetheus model to a new dataset. See README on how to use this function.
### Changed
- Fix issues with lossy integer conversions
- Some other small bug fixes

## [0.3.3] - 2018-11-25
### Changed
- Update how facenet TensorFlow model is based into object
- Fixes session recursion limit

## [0.3.2] - 2018-11-04
### Changed
- tindetheus will now exit gracefully if you have used all of your free likes while running tindetheus like

## [0.3.1] - 2018-11-04
### Changed
- Fix bug related to Windows and calc_avg_emb(), which wouldn't find the unique classes

## [0.3.0] - 2018-11-03
### Added
- Added version tracking and parser with --version
- New optional parameters: likes (set how many likes you have remaining default=100), and image_batch (set the number of images to load into facenet when training default=1000)
- Now all optional settings can be saved in config.txt
### Changes
- Bug fix related to calling a tindetheus.export_embeddings function
- Saving the same filename in your database no longer bombs out on Windows
- Code should now follow pep8

## [0.2.11] - 2018-05-11
### Added
- Added support for latest facenet models. The different facenet models don't appear to really impact the accuracy according to [this post](https://jekel.me/2018/512_vs_128_facenet_embedding_application_in_Tinder_data/).
- You can now specify which facenet model to use in the config.txt file.
- Added [example](https://github.com/cjekel/tindetheus/blob/master/examples/open_database.py) script for inspecting your database manually
### Changed
- Updated facenet clone implementation
- Now requires minimum tensorflow version of 1.7.0
