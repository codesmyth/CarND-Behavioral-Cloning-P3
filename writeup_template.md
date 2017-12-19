# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center-lane-driving.png "Center Driving"
[image3]: ./examples/center_2017_12_12_15_40_56_880.jpg "Recovery Image"
[image4]: ./examples/center_2017_12_12_15_40_59_713.jpg "Recovery Image"
[image5]: ./examples/center_2017_12_12_15_41_09_860.jpg "Recovery Image"
[image6]: ./examples/center.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I initially got the data processed using a very simple network, then evolved this network to a deep convolutional network based on LeNet. the results were reasonable but the car did  veer of to the side. I adopted a model based on the Nvidia end to end network and observed some more improvements.

I settled in this architecture and gathered more data.

#### 1. An appropriate model architecture has been employed
The final model is in a function(nvidia_e2e, codes lines XX)
The image data is normalized in the model using a Keras lambda layer
The images are then cropped to remove the sky and trees at the top and the car base of the image.
My model consists of a convolution neural network with 3 5x5 filter sizes and depths between 24 and 48, then two further convolutions
with 3x3 filters. each layer is activated with a RELU operation.

A drop out layer is then included to reduce overfitting. with a keep probability of 50%
Then four fully Connected layers are utilsed, the first three are activated with RELU and the last layer provides a single output.

The model is summarised in the final table below.



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with a learing rate of 0.001

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road driving the track in both clockwise and counter clockwise directions

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to generate sufficient data and apply a couple of different network architectures.

My first step was to use a convolution neural network model similar to the Le Net model I thought this model might be appropriate because it was straighforward to implement and had been effective in classifying images in the previous project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it included a dropout layer but I did include extra circuts of the track.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I provided more recovery data for these specific areas.

I augmented the dataset by making use of the left and right camera angles and by flipping 10% of the total images. for the left and right camera images I included a small correction to steering the car towards the middle when these images were observed.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 33, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two to three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid the edges and stay central, it took a while to generate enough recovery data  These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would remove the steering bais that might occur when going round an approximately circular track. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had sampled 75%(38,962) of the total  datapoints for each camera. After I augmented the images as described above i 58442 number of data points. I then preprocessed this data by converting the colour from BGR to RGB. up until this point the car did continue to drive of the road, when there were shadows or dirt roads etc. Initially I'd attempted to deal with the car leaving the track by generating more data. which is why I ended up using a lot.


I randomly shuffled before the data set when loading so that I could dropout a proportion of the data to and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 3 as the loss had dropped to 0.0090 and the validation loss dropped to 0.0086.

The video that comes along with this write up captures the car driving round the track for just over one lap without ever leaving the road.

