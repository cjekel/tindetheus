#!/usr/bin/env python3

import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

TINDETHEUS_MODEL_DIR = os.getenv('TINDETHEUS_MODEL_DIR', '20170512-110547')
TINDETHEUS_IMAGE_BATCH = os.getenv('TIDETHEUS_IMAGE_BATCH', 1000)
TINDETHEUS_DISTANCE = os.getenv('TINDETHEUS_DISTANCE', 5)
TINDETHEUS_LIKES = os.getenv('TINDETHEUS_LIKES', 100)
TINDETHEUS_RETRIES = os.getenv('TINDETHEUS_LIKES', 20)

TINDER_AUTH_TOKEN = os.getenv('TINDER_AUTH_TOKEN', None)
FACEBOOK_AUTH_TOKEN = os.getenv('FACEBOOK_AUTH_TOKEN', None)
