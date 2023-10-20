# Adapted from lab 08 (squares-circles)

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print("tensorflow version", tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from random import sample
import matplotlib.pyplot as plt
import os
import sys

## Define image properties:
imgDir = "data"
targetWidth, targetHeight, channels = 64, 64, 1
imageSize = (targetWidth, targetHeight)
## define other constants, including command line argument defaults
epochs = 1
plot = False 
maxTrainImg = len(os.listdir(os.path.join(imgDir, "trainCrop128")))
## command line arguments
# check if this was run as a separate file (not inside notebook)
import __main__ as main
if hasattr(main, "__file__"):
# run as file
    print("parsing command line arguments")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d",
    help = "directory to read images from",
    default = imgDir)
    parser.add_argument("--epochs", "-e",
    help = "how many epochs",
    default= epochs)
    parser.add_argument("--images", "-i",
    help = "how many training images",
    default= maxTrainImg)
    parser.add_argument("--plot", "-p",
    action = "store_true",
    help = "plot a few wrong/correct results")
    args = parser.parse_args()
    imgDir = args.dir
    epochs = int(args.epochs)
    maxImgs = int(args.images)
    plot = args.plot
else:
# run as notebook
    print("run interactively from", os.getcwd())
    imageDir = os.path.join(os.path.expanduser("~"),
    "data", "images", "text", "language-text-images")

print("Load images from", imgDir)
print("epochs:", epochs)
print("Limit number of training images to:", maxImgs)
## Prepare dataset for training model:
filenames = os.listdir(os.path.join(imgDir, "trainCrop128"))
print(len(filenames), "total training images found")
trainingResults = pd.DataFrame({
    'filename':filenames,
    'category':pd.Series(filenames).str[:2]
})

# Limit filenames to specified number
trainingResults = trainingResults.sample(maxImgs)

print("data files sample:")
print(trainingResults.sample(5))
nCategories = trainingResults.category.nunique()
print("categories:\n", trainingResults.category.value_counts())
## Create model
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,\
    MaxPooling2D, AveragePooling2D,\
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization
# sequential model (one input, one output)
model=Sequential()
model.add(Conv2D(64,
                kernel_size=4,
                strides=1,
                activation='relu',
                kernel_initializer = initializers.HeNormal(),
                input_shape=(targetWidth, targetHeight, channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=4))
model.add(Dropout(0.5))

model.add(Conv2D(32,
                kernel_size=4,
                strides=2,
                kernel_initializer = initializers.HeNormal(),
                activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(16,
                kernel_initializer = initializers.HeNormal(),
                activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(nCategories,
                kernel_initializer = initializers.HeNormal(),
                activation='softmax'))
model.compile(loss='categorical_crossentropy',
metrics=['accuracy'])
model.summary()

## Training and validation data generator:
trainingGenerator = ImageDataGenerator(
    rescale=1./255
).\
flow_from_dataframe(trainingResults,
                    os.path.join(imgDir, "trainCrop128"),
                    x_col='filename', y_col='category',
                    target_size=imageSize,
                    class_mode='categorical',
                    color_mode="grayscale",
                    shuffle=True)
label_map = trainingGenerator.class_indices
## Model Training:
history = model.fit(
    trainingGenerator,
    epochs=epochs
)

## Validation data preparation:
validationDir = os.path.join(imgDir, "validationCrop128")
fNames = os.listdir(validationDir)

print(len(fNames), "validation images")
validationResults = pd.DataFrame({
    'filename': fNames,
    'category': pd.Series(fNames).str[:2]
})


print(validationResults.shape[0], "validation files read from", validationDir)
validationGenerator = ImageDataGenerator(rescale=1./255).\
flow_from_dataframe(validationResults,
                    os.path.join(imgDir, "validationCrop128"),
                    x_col='filename',
                    class_mode = None,
                    target_size = imageSize,
                    shuffle = False,
                    color_mode="grayscale"
)
## Make categorical prediction:
print(" --- Predicting on validation data ---")
phat = model.predict(validationGenerator)
print("Predicted probability array shape:", phat.shape)
print("Example:\n", phat[:5])
## Convert labels to categories:
validationResults['predicted'] = pd.Series(np.argmax(phat, axis=-1),
index=validationResults.index)
print(validationResults.head())
labelMap = {v: k for k, v in label_map.items()}
validationResults["predicted"] = validationResults.predicted.replace(labelMap)
print("--- confusion matrix (validation) ---")
print(pd.crosstab(validationResults.category, validationResults.predicted))
print("-------------------------------------")
print("--- Validation accuracy: ", np.mean(validationResults.category == validationResults.predicted), " ---")

## Print and plot misclassified results
wrongResults = validationResults[validationResults.predicted !=
validationResults.category]
rows = np.random.choice(wrongResults.index, min(4, wrongResults.shape[0]),
replace=False)
print("--- Example wrong results (validation data) ---")
print(wrongResults.sample(min(10, wrongResults.shape[0])))
if plot:
    plt.figure(figsize=(12, 12))
    index = 1
    
    # Add wrong exmaples
    for row in rows:
        filename = wrongResults.loc[row, 'filename']
        predicted = wrongResults.loc[row, 'predicted']
        img = load_img(os.path.join(imgDir, "validationCrop128", filename), target_size=imageSize)
        plt.subplot(4, 2, index)
        plt.imshow(img)
        plt.xlabel(filename + " ({})".format(predicted))
        index += 1
    # now show correct results
    index = 5
    correctResults = validationResults[validationResults.predicted == validationResults.category]
    rows = np.random.choice(correctResults.index, min(4, correctResults.shape[0]), replace=False)
    
    # Add correct examples
    for row in rows:
        filename = correctResults.loc[row, 'filename']
        predicted = correctResults.loc[row, 'predicted']
        img = load_img(os.path.join(imgDir, "validationCrop128", filename),
        target_size=imageSize)
        plt.subplot(4, 2, index)
        plt.imshow(img)
        plt.xlabel(filename + " ({})".format(predicted))
        index += 1
    
    plt.tight_layout()
    plt.savefig("CategorizationExamples.png")
    plt.close()