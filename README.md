#**Traffic Sign Recognition** 

##About

My solution to the "Build a Traffic Sign Recognition Program" project in the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). Here you can find a [link to Udacity's upstream project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project).


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

[testfrequencyhistogram]: ./testdatafrequency.png "Label frequency histogram for test data"
[70sign]: ./70sign32x32.png "70 Sign"
[100sign]: ./100sign32x32.jpeg "100 Sign"
[donotentersign]: ./Do-Not-Enter32x32.jpg "Do Not Enter Sign"
[stopsign]: ./stopsign32x32.png "Stop Sign"
[yieldsign]: ./5155701-German-traffic-sign-No-205-give-way-Stock-Photo32x32.jpg "Yield Sign"
[100signudacity]: ./100fromUdacity.png "100 Sign from Training Set"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mandermo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 RGB
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The data set is not evenly distributed, some type of traffic signs are more common than others. Here is the label frequency distribution for the test data.
![alt text][testfrequencyhistogram]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Because traffic signs are a bit smaller than the full 32x32 image I have decided to crop them. For the test and validation images I centrally crop them to 28x28. After that I normalize them so they have 1.0 standard deviation and zero mean.

For the training images I have generated more data by rotation them and then randomly cropping them to 28x28. To make the images more robust to situations when the sign is not exactly in the middle and when they are not perfectly horizontal. The rotation angle is randomized as a gaussian with 10 degrees standard deviation. The translation for x and y is a integer that is an offset between -2 to 2 from a central cropping. It is weighted to make a smaller translation more likely.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 28x28x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 8x8x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x16  				|
| Fully connected		| 120 to 84        								|
| RELU					|												|
| Fully connected		| 84 to 43        								|
| Softmax				|           									| 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer and a learning rate of 0.001 and a batch size of 128 and 10 epochs.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.952 
* test set accuracy of 0.939

First I used a more or less unmodified LeNet architecture, which was not good enough. Then I did central cropping to 28x28 and got better, but not enough either. After I did some decent rotation and translation, but later changed it till a bit better code. I got 96% on validation set and then I tried dropout with keep probability of 50% for the fully connected layers and also tried dropout for input layer. It didn't improve the accuracy for the validation set, so I disabled the dropout with a flag, but keeping the code of it.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web which I have manually cropped and rescaled with Gimp.

![70 Sign][70sign] ![100 Sign][100sign] ![Do Not Enter Sign][donotentersign] 
![Stop Sign][stopsign] ![Yield Sign][yieldsign]

I didn't think there would be a problem classifying, but it didn't classify the 100 sign correctly and think it is a no passing when I had dropout and now it think it is a 70 sign. More discussion under next heading.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 Sign         		| 70 Sign   									| 
| 100 Sign     			| No passing 									|
| No entry				| No entry										|
| Stop  	      		| Stop	        				 				|
| Yield     			| Yield              							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. It is too few samples to make a conclusion if it is similar accuracy as the validation and test accuracy.

When I look at a 100 sign from the training set I see that the training sign is more blury and therefore look quite different ![100 Sign from Training Set][100signudacity]. Mine is also covering more of the image, but I don't feel for recropping it. Gaussian bluring any sign that should be classified could solve the problem, but I think that could lead to other problems.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located under the heading "Output Top 5 Softmax Probabilities For Each Image Found on the Web".

For the first image, the model is very sure that this is 70 Sign (probability of 1.0), and the image does contain a 70 sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000                 | 70 sign   									| 
| 1e-10                 | 20 sign                                    	|
| 6e-12                 | 30 sign										|
| 2e-24                 | 50 sign     					 				|
| 6e-27                 | 80 sign            							|

Even 1e-10 is zero according to my defitionion, it is comforting to know it only guess speed limit signs for the top 5 preditions for the 70 sign.

---

For the first image, the model is erroneously quite sure that this is 70 Sign (probability of 0.86), and the image contains a 100 sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.861                 | 70 sign   									| 
| 0.139                 | 120 sign                                    	|
| 4e-6                 | 100 sign										|
| 3e-6                 | 80 sign     					 				|
| 7e-7                 | 50 sign            							|

It guesses wrong and the second top probability is also wrong, the third is the sign that appeared, but with very very low estimated probability. Fortnately all of the guesses are speed limit signs.

---
For the first image, the model is very sure that this is No Entry Sign (probability of 1.0), and the image does contain a No Entry Sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000       			| No entry   									| 
| 2e-13                | NO passing                                  	|
| 5e-17		            | Vehicles over 3.5 metric tons prohibited     |
| 3e-17         		| No passing for vehicles over 3.5 metric tons   |
| 1e-21         	    | Yield sign            							|


---

For the first image, the model is very sure that this is Stop sign (probability of 1.0), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000                 | Stop sign   									| 
| 1e-10                 | 50 sign                                    	|
| 5e-11                 | 30 sign										|
| 5e-11                 | 70 sign     					 				|
| 2e-12                 | 120 sign            							|


---
For the first image, the model is very sure that this is Yield sign (probability of 1.0), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000                 | Yield sign   									| 
| 1e-18                 | No passing                                    	|
| 2e-19                 | No vehicles										|
| 5e-21                 | Ahead only     					 				|
| 1e-21                 | 50 sign            							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Not done.
