# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imbalanced_data.png "Imbalanced Data"
[image3]: ./30.png "Original Image"
[image4]: ./30s.png "Preprocessed Image Possibilities"
[image6]: ./test_images/11_rigtoffway_atnextintersection_32x32x3.jpg "Traffic Sign 1"
[image7]: ./test_images/12_priority_road_32x32x3.jpg "Traffic Sign 2"
[image8]: ./test_images/17_noentry_32x32x3.jpg "Traffic Sign 3"
[image9]: ./test_images/31_wildanimalscrossing_32x32x3.jpg "Traffic Sign 4"
[image10]: ./test_images/34_turn_left_ahead.jpg "Traffic Sign 5"
[image11]: ./LeNet.png	"LeNet Architecture"
[image12]: ./LeCun_Sermanet.png "LeCun_Sermanet Architecture"
[image13]: ./softmax_probs.png "Softmax Probabilities"


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code]https://github.com/ssimontacchi/Classifying-Traffic-Signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32x32 pixels.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing that the data came extremely imbalanced across all traffic sign types, subsequently to be balanced after the image preprocessing step.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to create more data so the model would train more effectively. I was able to accomplish this by randomly augmenting the brightness, rotating, translating, and adjusting the shear of each image. Below is an initial image, and below are some of the many possible outcomes from this process.

![alt text][image3]
![alt text][image4]


Next, I decided to convert the images to grayscale as this approach was found to be more effective by LeCun in his initial implementation.
Here is an example of a traffic sign image before and after grayscaling.

As a last step, I normalized the image data so the model would be able to converge faster. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model, inspired by LeCun-Sermanet's implementation for this problem, consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	- Layer 1    	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16				|
| RELU					|												|
| Max pooling	- Layer 2    	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 1x1x400				|
| RELU		- Layer 3			|	   	     |
| Flatten and Concatenate Layers 2 and 3		|  		  	|	
| Dropout   |           |
| Softmax				|      									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, the name of which comes from 'adaptive moment optimizer', to update network weights iteratively based in training data. I used a learning rate of 0.001, a batch size of 32, and 12 epochs, which seemed to be the point when the validation accuracy reached its highest point, after which it would be overfitting the training data.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My  model results were:
LeNet Architecture: (10 epochs)
    Training Accuracy: 0.836
    Validation Accuracy: 0.948
    Test Accuracy: 0.913
    
LeCun-Sermanet (15 epochs):
    Training Accuracy: 0.972
    Validation Accuracy: 0.954
    Test Accuracy: 0.936


The first architecture I used was the LeNet model, but since I wasn't achieving high enough accuracy, I decided to switch to a LeCun-Sermanet architecture, which adds an additional convolutional layers and concatenates the results with an earlier layer before sending those weights through a softmax. 
![alt text][image11]
![alt text][image12]
I found that my initial pick of a LeNet architecture just wasn't capturing enough of the information, leading to weak fits on the test data. The LeCun-Sermanet architecture is clearly more powerful, going deeper than the LeNet architecture but avoiding overfitting by using an earlier layers' weights as well.

As my training accuracy is not extremely high, indicating overfitting, I would like to conitinue to add depth to this model with more convolutions to deepen my model's 'experience' and using dropout to protect against overfitting. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way at next intersection     		| Right of way at next intersection   									| 
| Priority Road     			| Priority Road  										|
| No entry					| No entry											|
| Turn Left Ahead      		| Turn Left Ahead 				 				|
| Animal Crossing			| Animal Crossing     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.5%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image13]
