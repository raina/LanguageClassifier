# LanguageClassifier
A feed-forward neural network that classifies scanned book pages by language (options: EN, DA, RU, TH, ZN).

Originally developed as a final project for INFO 371 at the University of Washington.

---
## Model Overview

This is a feed-forward neural network that categorizes black and white scans of book pages by language. The books are written in English, Danish, Chinese (simplified), Russian, and Thai. 

As the images are all of varying sizes, and the page formats vary significantly (e.g. title pages, dense prose, sparse poems, etc.), I created a helper utility to crop the images to their most "information dense" region. This allows the model to focus on the most relevant details (differences in character forms and order) while also greatly boosting efficiency. 

The final model has a validation accuracy of 96.6% across a dataset of 32,000 images. 


---

## Running the Model

1. Load the dataset from the Google Drive folder.
2. Unzip it and rename folders so that you have a main "data" folder, which contains "train" and "validation" subfolders. 
3. Run `filenames.py` from the command line. This renames each file in a folder by cutting off anything up to the first underscore character. Due to one novel containing an extra underscore, it must be run twice per folder. Once run, each file will start with its language identifier (e.g. "EN").
4. From the command line, run `cropImages.py`. This crops each image in a folder to it's "best" 128x128 section (further information is below). It should be run on both the validation and training image folders.
5. Finally, run the model `language-detector.py` from the command line. Using the optional argument `-p` will create and save a plot of 4 mis-categorized and 4 correctly categorized images for reference.

---

## Cropping Algorithm

The model's performance can largely be attributed to the cropping algorithm. The algorithm is very straightforward, and improves results greatly. It works by checking each region of a given size (default: 128 x 128) in each image. The pixel values of each kernel are pooled to form an average and the highest average kernel is stored. This works because the images are black and white, so the data of the image is fully encoded in black pixels. Therefore, the higher average, the higher density of information in a kernel. After each possible kernel is checked, the image is cropped to the highest-value region and a new copy of the image is stored. 