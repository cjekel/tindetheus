from .tindetheus import *  # noqa F401
import .machine_learning as machine_learning # noqa F401
import .image_processing as image_processing # noqa F401
import .tinder_client as tinder_client # noqa F401
import os

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
