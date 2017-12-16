# MIT License
#
# Copyright (c) 2017 Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import pynder

# set path for security
sys.path.append(r'C:\Users\cj\Documents\run_tin')

def create_new_config():
    print('test')

def login_test():
    'test'
def main(args):


    print(args)
    # sleep(random.random())
    # output_dir = os.path.expanduser(args.output_dir)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # # Store some git revision info in a text file in the log directory
    # src_path,_ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    # dataset = facenet.get_dataset(args.input_dir)

'''
There are three function choices: browse, build, like
browse: review new tinder profiles and store them in your database
build: use machine learning to create a new model that likes and dislikes profiles based on your historical preference
like: use your machine leanring model to like new tinder profiles
'''

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='There are three function choices: browse, build, or like')
    #
    # parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    # parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    # parser.add_argument('--image_size', type=int,
    #     help='Image size (height, width) in pixels.', default=182)
    # parser.add_argument('--margin', type=int,
    #     help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    # parser.add_argument('--random_order',
    #     help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    # parser.add_argument('--gpu_memory_fraction', type=float,
    #     help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    # parser.add_argument('--detect_multiple_faces', type=bool,
    #                     help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':

    # check for a config file
    try:
        with open('config.txt') as f:
            lines = f.readlines()
            facebook_token = lines[0].split(' ')[-1].strip()
            facebook_id = lines[1].split(' ')[-1].strip()
            print('token:', facebook_token)
            print('id:', facebook_id)

    except:
        print('No config.txt found')
        create_new_config = input('Would you like us to create a new config.txt file? (y,n) : ')
        if create_new_config == 'y' or create_new_config == 'Y':
            print('Creating a new config...')

    main(parse_arguments(sys.argv[1:]))
