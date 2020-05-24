#!/usr/bin/env python3
#
# MIT License
#
# Copyright (c) 2020 Vikash Kothary, Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

from dotenv import load_dotenv
cwd = os.getcwd()
load_dotenv(dotenv_path=os.path.join(cwd, '.env'))


def strip_strings(var):
    if var is not None:
        return var.rstrip()
    else:
        return var


TINDETHEUS_MODEL_DIR = os.getenv('TINDETHEUS_MODEL_DIR', '20170512-110547')
TINDETHEUS_IMAGE_BATCH = os.getenv('TIDETHEUS_IMAGE_BATCH', 1000)
TINDETHEUS_DISTANCE = os.getenv('TINDETHEUS_DISTANCE', 5)
TINDETHEUS_LIKES = os.getenv('TINDETHEUS_LIKES', 100)
TINDETHEUS_RETRIES = os.getenv('TINDETHEUS_RETRIES', 20)

TINDER_AUTH_TOKEN = os.getenv('TINDER_AUTH_TOKEN', None)
FACEBOOK_AUTH_TOKEN = os.getenv('FACEBOOK_AUTH_TOKEN', None)

# remove line endings if needed
TINDETHEUS_MODEL_DIR = strip_strings(TINDETHEUS_MODEL_DIR)
TINDER_AUTH_TOKEN = strip_strings(TINDER_AUTH_TOKEN)
FACEBOOK_AUTH_TOKEN = strip_strings(FACEBOOK_AUTH_TOKEN)
