#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./classes.png "Visualization"
[image2]: ./new-images/14.jpg "Traffic Sign 1"
[image3]: ./new-images/15.jpg "Traffic Sign 2"
[image4]: ./new-images/23.jpg "Traffic Sign 3"
[image5]: ./new-images/35.jpg "Traffic Sign 4"
[image6]: ./new-images/40.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! You can find the code for classifying the traffic signs at https://github.com/cdavidr/german-traffic-sign-classification/blob/master/Traffic_Sign_Classifier.ipynb

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the the built-in python library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. I decided to load up an example of each class to get a better idea of what the each class looks like.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to "shuffle" the data because otherwise the ordering of the data can have a huge effect on how our network "learns" the model.

Next, I normalized the image data because it is always a good idea to follow a zero mean, equal variance for the values of the data as it makes it easier for the optimizer to handle the data numerically.



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

The data was pre-split when loading the data and the split consists of setting ~42.5% to train, ~42.5% to validate, and ~15% was set aside for testing.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook.

My model is based upon the LeNet architecture which uses convolutional layers and was implemented using TensorFlow.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Output = 400									|
| Fully Connected		| Output = 120									|
| RELU					|												|
| Fully Connected		| Output = 84									|
| RELU					|												|
| Fully Connected		| Output = 10									|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth cell of the ipython notebook.

To train the model, I used an Adam Optimizer to minimize the loss of the logits.

I trained the model for 10 epochs, mainly to save time by training on my personal laptop. The batch size is set to 128 and the learning rate is 0.001.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.


My final model results were:
* validation set accuracy of 88.8%
* test set accuracy of 88.6%


If a well known architecture was chosen:

* What architecture was chosen?

The architecture chosen was the LeNet architecture used in previous labs.

* Why did you believe it would be relevant to the traffic sign application?

Since LeNet was successfully used to identify text images, I believed it should
adapt well to other images that represent any type of symbol such as those seen
in traffic signs.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The model provides an accuracy ~90% for each set, which means it succesfully determines what an image represents most of the time given that they follow a similar resolution and view angle as in the German Traffic Sign dataset.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6]


Taking account of the images in their resized state: the only difficulty it should have with this image is if the distortion of the stop sign is too big from resizing.

The second image should be straightforward.

The third image may be very difficult as, again, resizing the image squished it together so much that it is difficult for the symbol to completely show up.

The fourth image may be very difficult due to it's postion and angle in the photo. However, the symbol is still pretty clear and should be fine due to translational invariance.

The fifth image may be difficult due to the angle the photo was taken at.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image						|     Prediction	        		|
|:-------------------------:|:---------------------------------:|
| Stop    					| Stop   							|
| No vehicles 				| Stop 								|
| Slippery road 			| Slippery road 					|
| Ahead only 	      		| Ahead only 					 	|
| Roundabout mandatory 		| Roundabout mandatory      		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 88.6%.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is extremely sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Stop    										|
| .00     				| Yield 										|
| .00					| Bicycles crossing								|
| .00      				| Speed limit (50km/h)					 		|
| .00				    | No passing      								|

For the second image, the model is very sure that this is a stop sign (probability of 0.89), but the image does not contain a stop sign. This may be due to how common this shape is within various stop signs, so other stop signs have similar features to this one (the round red border). It was the network's fourth guess but was not sure at all. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .89         			| Stop   									|
| .09     				| Yield 										|
| .01					| No Passing											|
| .01	      			| No Vehicles					 				|
| .00				    | Speed limit (50 km/h)      							|

For the third image, the model is extremely sure that this is a slippery road sign (probability of 0.99), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Slippery road   									|
| .00     				| Dangerous curve to the right 										|
| .00					| Right-of-way at the next intersection											|
| .00	      			| Road narrows on the right					 				|
| .00				    | Road work      							|

For the fourth image, the model is extremely sure that this is a ahead only sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Ahead only   									|
| .01     				| Speed limit (60km/h) 										|
| .00					| Go straight or right											|
| .00	      			| Yield					 				|
| .00				    | Turn left ahead      							|

For the first image, the model is extremely sure that this is a roundabout sign (probability of 0.99), and the image does contain a roundabout sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Roundabout mandatory   									|
| .00     				| Turn right ahead										|
| .00					| Speed limit (50km/h)											|
| .00	      			| Ahead only					 				|
| .00				    | Keep left      							|
