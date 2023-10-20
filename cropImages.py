# This is a utility to crop images to a region of their highest information density.
# The purpose is to sample images for use in a CNN, reducing the input size to only an "important" region
# It pools each region to find an avg then takes lowest average (1 being fully white, 0 fully black)
# This works because we're looking at standardized grey-on-white text
# This is more simple that sampling each possible 32x32 region, but misses out on the actual "best" region

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.python.ops.numpy_ops import np_config
from numpy import unravel_index
from PIL import Image

# Set desired region size e.g. 32 gives a 32x32 kernel
# Default 128 seems to capture a good amount of data from images of 19px text
regionSize = 128
folderName = "data/train"
outputFolder = "data/trainCrop128"

for count, filename in enumerate(os.listdir(folderName)):

    # Ignore hidden files
    if not filename.startswith('.'):

        fullPath = folderName+"/"+filename
        # Load in image
        pixels = imread(fullPath)

        # Get image dims
        height = pixels.shape[0]
        width = pixels.shape[1]

        # Only resize images that are at least 1 region large
        # This loses a couple images but not many (~200/40000 total)
        if (height > regionSize) & (width > regionSize):

            # Reshape to tensor for compatibility
            x = tf.reshape(pixels, shape=(1,height,width,1))

            # Tf throws an error if the input is an int - not an issue with smaller imgs but larger ones will fatally error
            x = tf.cast(x, tf.float32)

            # Use average pool to find relative pixel "density" per region of given size
            # Strides = region size so that each pixel is only looked at once. This is nice for reducing complexity but
            # almost certainly misses out on the absolute best possible region.
            avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(regionSize, regionSize), strides=(regionSize, regionSize), padding='valid')
            pooledimg = avg_pool_2d(x)

            pooledRows = pooledimg.shape[1]
            pooledCols = pooledimg.shape[2]

            np_config.enable_numpy_behavior()
            img = pooledimg.reshape((pooledRows,pooledCols))

            # Convert to np array to use argmin()
            img = np.asarray(img)

            # image is greyscale, white px will be 1 and black 0 so we want the lowest value in the array of averages
            minValue = img.argmin()
            cropcoords = unravel_index(img.argmin(), img.shape)

            # Set coordinates for cropping
            # Images start at the top-left corner
            top = cropcoords[0]*regionSize
            left = cropcoords[1]*regionSize
            bottom = top + regionSize
            right = left + regionSize

            # Open, crop, and save image
            im = Image.open(fullPath)
            im1 = im.crop((left, top, right, bottom))
            
            outputPath = outputFolder+"/"+filename
            im1 = im1.save(f"{outputPath}{str(regionSize)}.jpg")