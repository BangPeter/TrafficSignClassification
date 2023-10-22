Utilizing a Convolutional Neural Network to create an Image Classification model for autonomous vehicles.

## Contents
SRC folder - containing source code for our EDA and model.

DATA folder - containing our dataset.

FIGURES folder - containing figures/visualizations of our data.

LICENSE.md - MIT License.

README.md - The current file.

More details below.

## SRC
DS 4002 Dataset/EDA - Code files containing our initial EDA and visualizations.
Project2Images.ipynb - Jupyter Notebook containing our CNN model and analysis
Project2.ipynb - Jupyter Notebook for EDA with visuals


### Installing/Building the model
Make sure you have the latest version of python installed, with a IDE capable of opening jupyter notebooks. VSCode/Google Colab both work. Make sure all necessary libraries are installed as well. Most should be automatically installed once our code is run. From there, copy the dataset from our drive link to the same location as Project2.ipynb and Project2Images.ipynb. 

### Usage of the code
Each code block can be run in sequential order, and the model should work. Training time may vary depending on dataset size and number of epochs.

## DATA
Data (image_names.csv) is stored in the in the Project2 folder in the [Google Drive link](https://drive.google.com/drive/folders/1t95iHN5Yjhsas3PWvepcH_d1jYshHVLU).

| **File**       | **File Type**     | **Description** | **Size**   |  **Color Space** |
|--------------|-----------|------------|-----------|------------|
| Image | JPG  | Images of Traffic Signs and Surrounding Location      | Width, Height | RGB |
| Annontation| JSON  | Annotations that Provide Labels, Width, Height, Bounding Box Dimensions, etc for the Images     | Width, Height | N/A |

Data was split into a training set and testing set for a total of 1500 rows. The training set contained 25000 rows and was used to train the model. The testing set contained 10000 rows and was used to test the results of the model.

## FIGURES
| **Figure**       | **Description**     | **Takeaways** |
|--------------|-----------|------------|
| Sampling of Traffic Sign Images | Display of some images from the data set | N/A   |
| Histogram of Image Height | Histogram displaying the distribution of image heights in the data set  |   A majority of pictures have a height around 2500px or 3000px, with a significant gap between those to bins. This is due to a lack of picture sizes that fit a height between 2500-3000px. |
| Histogram of Image Width | Histogram displaying the distribution of image widths in the data set   | Image widths are more closely grouped together from the 3000-5000px range. This not only shows that images are typically wider on average, by comaring the distributions to the image hieght graph, but it also reveals that images tend to fit a more normal distribution pattern in terms of width.

## REFERENCES
[1]	“The Importance of Robust Algorithms for Autonomous Vehicle’s,” www.linkedin.com. https://www.linkedin.com/pulse/importance-robust-algorithms-avs-a-ernesto-aguilar/  (accessed Oct. 08, 2023). 

[2]	“Guide on object detection & its use in self-driving cars,” Labellerr, Oct. 11, 2022. https://www.labellerr.com/blog/how-object-detection-works-in-self-driving-cars-using-deep-learning/ (accessed Oct. 08, 2023).

[3]	“Mapillary,” www.mapillary.com. https://www.mapillary.com/dataset/trafficsign

[4]	“How To Build Powerful Keras Image Classification Models | Simplilearn,” Simplilearn.com. https://www.simplilearn.com/tutorials/deep-learning-tutorial/guide-to-building-powerful-keras-image-classification-models

[5]	“Python Project on Traffic Signs Recognition with 95% Accuracy using CNN & Keras,” DataFlair, Dec. 04, 2019. https://data-flair.training/blogs/python-project-traffic-signs-recognition/	

[6]	“How to Evaluate An Image Classification Model | Clarifai Guide,” docs.clarifai.com. https://docs.clarifai.com/tutorials/how-to-evaluate-an-image-classification-model/

[7]	“Image classification | TensorFlow Lite,” TensorFlow. https://www.tensorflow.org/lite/examples/image_classification/overview

[8]	T. Gautam, “Create Your Own Image Classification Model Using Python and Keras,” Analytics Vidhya, Oct. 16, 2020. https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/#h-setting-up-our-image-data (accessed Oct. 15, 2023).

‌[9]	“Standard photo print sizes chart,” PicMonkey. https://www.picmonkey.com/photo-editor/standard-photo-print-sizes (accessed Oct. 15, 2023). 

[10]	“Traffic Signs Recognition using CNN and Keras in Python,” Analytics Vidhya, Dec. 21, 2021. https://www.analyticsvidhya.com/blog/2021/12/traffic-signs-recognition-using-cnn-and-keras-in-python/

‌[11]	A. Gozhulovskyi, “Classification of Traffic Signs with LeNet-5 CNN,” Medium, Apr. 02, 2022. https://towardsdatascience.com/classification-of-traffic-signs-with-lenet-5-cnn-cb861289bd62 (accessed Oct. 15, 2023).
‌
[12]	“python - Loading all images using imread from a given folder,” Stack Overflow. https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder 


### Acknowledgements
Professor Alonzi

Harsh Anand (TA)

Group12

### Previous Works
MI1: https://docs.google.com/document/d/1FZpMf64y3wwJ4Pmt0PROBRixD_NpVWIvQzaojZHrSHw/edit

MI2: https://docs.google.com/document/d/157GVg0jQyZZi0fBuvD9dszffu_FfapXV5NTuHlAjz4Q/edit

This project is licensed under the terms of the MIT license.
