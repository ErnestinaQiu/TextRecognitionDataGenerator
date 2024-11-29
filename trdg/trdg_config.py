import argparse
import errno
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random as rnd
import string
import sys
from multiprocessing import Pool

from tqdm import tqdm

from trdg.data_generator import FakeTextDataGenerator
from trdg.string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly,
)
from trdg.utils import load_dict, load_fonts


def margins(margin):
    margins = margin.split(",")
    if len(margins) == 1:
        return [int(margins[0])] * 4
    return [int(m) for m in margins]


class TRDG_config(object):
    def __init__(self) -> None:
        self.language = 'en'
        self.count = 1                    # the number of samples are generated in each turn
        self.random_sequences = True      # Use random sequences as the source text for the generation
        self.include_letters = True       # Define if random sequences should contain letters. Only works with random sequences True
        self.include_numbers = True       # Define if random sequences should contain numbers. Only works with random sequences True
        self.include_symbols = True       # Define if random sequences should contain symbols. Only works with random sequences True
        self.length = 1                   # Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length
        self.random = True               # Define if the produced string will have variable word count (with --length being the maximum)
        self.format = 48                  # Define the height of the produced images if horizontal, else the width 
        self.thread_count = 1             # Define the number of thread to use for image generation
        self.skew_angle = 0               # Define skewing angle of the generated text. In positive degrees
        self.random_skew = False          # When set, the skew angle will be randomized between the value set with -k and it's opposite
        self.blur = 0                     # Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius
        self.random_blur = False          # When set, the blur radius will be randomized between 0 and -bl
        self.background = 1               # Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image
        self.name_format = 4              # Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings, 3: [ID]_[TEXT].[EXT] + one file labels.txt containing id-to-label mappings, 4: only output image and text, not save
        self.output_mask = 0              # Define if the generator will return masks for the text
        self.output_bboxes = 0            # Define if the generator will return bounding boxes for the text, 1: Bounding box file, 2: Tesseract format
        self.distorsion = 0               # Define a distortion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random
        self.distorsion_orientation = 0   # Define the distortion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both
        self.width = -1                   # Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used
        self.alignment = 1                # Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right
        self.orientation = 0              # Define the orientation of the text. 0: Horizontal, 1: Vertical
        self.text_color = "#282828"       # Define the text's color, should be either a single hex color or a range in the ?,? format.
        self.space_width = 1.0            # Define the width of the spaces between words. 2.0 means twice the normal space width
        self.character_spacing = 0        # Define the width of the spaces between characters. 2 means two pixels
        self.margins = (5, 5, 5, 5)       # Define the margins around the text when rendered. In pixels
        self.fit = False                  # Apply a tight crop around the rendered text
        self.font = ''                    # Define font to be used
        self.font_dir = ''                # Define a font directory to be used
        self.image_dir = ''               # Define an image directory to use when background is set to image
        self.case = ''                    # Generate upper or lowercase only. arguments: upper or lower. Example: --case upper
        self.dict = ''                    # Define the dictionary to be used
        self.word_split = False           # Split on words instead of on characters (preserves ligatures, no character spacing)
        self.stroke_width = 0             # Define the width of the strokes
        self.stroke_fill = "#282828"      # Define the color of the contour of the strokes, if stroke_width is bigger than 0
        self.image_mode = 'RGB'           # Define the image mode to be used. RGB is default, L means 8-bit grayscale images, 1 means 1-bit binary images stored with one pixel per byte, etc.
        self.color_inverse = False        # Use 255 - img to inverse color value of image
        self.output_dir = ''
        self.extension = ''
        self.handwritten = False
        self.input_file = ''