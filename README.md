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

Here is a link to my [project code]https://github.com/ssimontacchi/Classifying-Traffic-Signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32x32 pixels.
* The number of unique classes/labels in the data set is 43.

Here is a visualization of the data set. It is a bar chart showing that the data came extremely imbalanced across all traffic sign types, subsequently to be balanced after the image preprocessing step.

![alt text][image1]

### Design and Test Model Architecture

As a first step, I decided to create more data so the model would train more effectively. I was able to accomplish this by randomly augmenting the brightness, rotating, translating, and adjusting the shear of each image. Below is an initial image, and below are some of the many possible outcomes from this process.

![alt text][image3]
![alt text][image4]


Next, I decided to convert the images to grayscale as this approach was found to be more effective by LeCun in his initial implementation.
Here is an example of a traffic sign image before and after grayscaling.

As a last step, I normalized the image data so the model would be able to converge faster.


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


#### Training

To train the model, I used an AdamOptimizer, the name of which comes from 'adaptive moment optimizer', to update network weights iteratively based in training data. I used a learning rate of 0.001, a batch size of 32, and 12 epochs, which seemed to be the point when the validation accuracy reached its highest point, after which it would be overfitting the training data.


#### Results

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
Original LeNet Architecture:
![alt text][image11]
LeCun - Sermanet Architecture:
![alt text][image12]

I found that my initial pick of a LeNet architecture just wasn't capturing enough of the information, leading to weak fits on the test data. The LeCun-Sermanet architecture is clearly more powerful, going deeper than the LeNet architecture but avoiding overfitting by using an earlier layers' weights as well.

As my training accuracy is not extremely high, indicating overfitting, I would like to conitinue to add depth to this model with more convolutions to deepen my model's 'experience' and using dropout to protect against overfitting.


### Testing Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]


Here are the results of the predictions:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right of way at next intersection     		| Right of way at next intersection   									|
| Priority Road     			| Priority Road  										|
| No entry					| No entry											|
| Turn Left Ahead      		| Turn Left Ahead 				 				|
| Animal Crossing			| Animal Crossing     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.5%

#### 3. Softmax Probabilities for each Prediction

![alt text][image13]
