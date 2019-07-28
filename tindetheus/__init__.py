from .tindetheus import *  # noqa F401
from . import tindetheus_align  # noqa F401
from . import export_embeddings  # noqa F401
from . import machine_learning  # noqa F401
from . import image_processing  # noqa F401
from . import tinder_client  # noqa F401
import os

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
