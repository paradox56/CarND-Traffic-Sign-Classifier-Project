
# **Traffic Sign Recognition**

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

### Data Set Summary & Exploration

Here is a visual summary of the number of training images with respect to each label:
![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/all_labeled_images.png "Training Data")

There are in total of 34799 training images, 4410 validation images and 12630 testing images. Each image has a dimension of $32 \times 32 \times 3$. In total, there are 43 labels that uniquely determine 43 German Traffic Signs.

#### 2. Include an exploratory visualization of the dataset.

Here is a visualization of each unique label and corresponding traffic sign:
![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/histogram.png "Histogram")

### Design and Test a Model Architecture

#### Image Processing

I have tried to train the neural network with different image pre-processing techniques and I notice the nomalization plays a huge important role. Since I have a powerful GPU on my local machine (GTX980Ti), I decide to use image input with three-color channels (RGB) direclty without applying grayscale. First, I tred to apply normalziation with `(pixel - 128)/ 128` and the validation accuracy can bareley reach 85%. Next, I applied the standard statistical nomalization technique for the training data using `pixel - pixel.mean()/pixel.std()` and the result improves the accracuy dramatlcally.

To increase the ramdomness of data argumentation, I decide to perform an on-the-go data argumentation approach, .i.e., randomly making modifications on each image in one Epoch training. I believe the brightness might play a big role during image classification. Therefore, I ramdomly apply brightness adjustment during each Epoch training process using `brightness_adjustment()` function on the normalized image. In addition, I have also tried gaussian blur to further increase the possibility of image argumentation but the result is not phenomenal. It is then removed to increase data processing speed.

Here is an example of an original image and the normalized image:

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/img_original.png "Origin Image")![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/nomalized_img_original.png "Normalized Image")

Here is an example of normalized image and brightness adjusted image (brighter in this case):

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/nomalized_img_original.png "Normalized Image")![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/nomalized_img_brightness_adjusted.png)

#### Neural Network Models
I have experimented with two different Convolutional Neural Network (CNN) models. Here is the the one I end up using (namely nn_model()):

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5    	| 1x1 stride, Valid padding, outputs 28x28x32 	|
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, Valid padding, outputs 24x24x64   |
| RELU                  |                                               |
| Max Pooling	      	| 2x2 stride, outputs 16x16x64 				    |
| Convolution 3x3       | 1x1 stride, Valid padding, outputs 10x10x128  |
| RELU                  |                                               |
| Max Pooling           | 1x1 stride, outputs 5x5x128                   |
| Convolution 2x2       | 1x1 stride, Valid padding, outputs 4x4x256    |
| RELU                  |                                               |
| Max Pooling           | 2x2 stride, outputs 2x2x256                   |
| Flatten               | outputs 1024                                  |
| Fully connected       | outputs 512                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 256                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 43                                    |
| Softmax				| outputs 43     								|

This particular CNN structure is inspired from VGG16. The idea is to multiple convolutional layers with decreasing kernel size as the layer gets deeper. However, the number of CNN filters doubles consecutively (32, 64, 128 ...). The itution is that the first few CNN filter learns the general features, such as edges etc. As the layer gets deeper, the smaller filters are used to learn the more detailed features. To reduce the possibility of overfitting, I added two dropout layers that remove 50% of the input from previous layers.

Just for reference, the second CNN model`nn_model_2()` is simlar but with much less layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5    	| 1x1 stride, Valid padding, outputs 28x28x32 	|                       |
| Max Pooling	      	| 2x2 stride, outputs 16x16x64 				    |
| RELU                  |                                               |
| Convolution 2x2       | 1x1 stride, Valid padding, outputs 12x12x64   |
| Max Pooling           | 2x2 stride, outputs 6x6x64                    |
| RELU                  |                                               |
| Flatten               | outputs 2304                                  |
| Fully connected       | outputs 1152                                  |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 576                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 43                                    |
| Softmax				| outputs 43     								|


To train the model, I initialize the weights of the CNN with a mean of 0 and variance of 0.01. I break the entire training data set into 256 images for each batch and perform training with 20 epochs.

The loss function I used here is multi-class cross-entropy function:

$$-\sum_{c=1}^m y_c log(p_c)$$
,where $y$ is the label and $p_c$ is the probability of the corresponding label.
Next, I use Adam optimization algorithm with learning rate of 0.0009 to minimize the loss. The training process is relatively fast with a dedicated GPU.

#### CNN Design Process
The design process is performed iteratively. Starting with Lenet from Yann LeCun, the validation accuracy can only reach roughly 85% even with image argumentation. From my experience, we need more complex neural network structure i.e., more CNN filters and deeper NN layers, to train the model better. But there is no free lunch, as the more complciated NN requries more data to train. It would be an overkill to use something like Resnet-50 for this paticular classifier. Therefore, the VGG-16 draws my attension during the design process. The idea is to design a CNN model that can capture enough features, while minimizing the number of layers and parameters. In the end, we do not want a overcomplicated model that is difficult to train. After a few trial and errors, I ended up with a CNN that has 10 layers.

My final model has a validation accuracy around 96%.


### Test a Model on New Images

To test out if my CNN can correctly classify new images that it has never seen before. Here are eight German traffic signs that I found on the web:

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/test_image_set.png "test images")


Before feeding testing images into the pipeline, I perform a quick resizing so it matches the input requirement of the CNN (In this case, 32x32x3). Next, we must also normalize the input using the previously introduced technique. The code is shown as the following:

```
from PIL import Image

# Scale Image to Appropriate Size 32x32x3
test_images_reshaped = np.empty(shape=(num_test_images,32,32,3),dtype =np.uint16)
for index in range(0,num_test_images):
    im = Image.fromarray(new_test_images_array[index])
    im_reshaped = im.resize((32,32),Image.LANCZOS)
    test_images_reshaped[index] = np.asarray(im_reshaped)

# Generate test images label
y_new_test = np.array([17,28,8,25,33,14,4,11])

# Generate test images feature matrix
x_new_test = test_images_reshaped

# Normalization
x_new_test_normalized = (x_new_test - x_new_test.mean())/x_new_test.std()

```

My model has sucessfully identified all testing images from the web.Here are the results of the prediction:

| Testing Image	        |     Predicted Class ID	        					|
|:---------------------:|:---------------------------------------------:|
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/17.png "no entry")     		| 17 (No Entry)   					            |
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/28.png "Children crossing")    			| 28 (Children crossing)		                |
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/8.png "Speed limit 120")				| 8  (Speed limit 120km/h)			            |
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/25.png "Road Work")	      		| 25 (Road Work) 				                |
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/33.png "Turn right ahead")			| 33  (Turn right ahead)			            |
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/14.png "Stop")			| 14  (Stop)				                    |
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/4.png "Speed limit 70")			| 4   (Speed limit 70km/h)  					|
| ![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/11.png "Right-of-way")		| 11  (Right-of-way at the next intersection)	|

The code for making predictions on my final model is the following:

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    server.restore(sess, "./model")
    test_accuracy = evaluate(x_new_test_normalized, y_new_test)
    print('The test accuracy is ',test_accuracy*100,'%')
```

It reads the previoulsy saved model and then use `evaluate()` function to check the accuracy of each prediction.

For the testing image `No Entry`, the model believes it has a probability 1.0 to be `No Entry`.

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/17.png "no entry")

The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| 17  (No entry)									|
| 9.612020e-10		    | 42 	(End of no passing by vehicles over 3.5 metric tons)									|
| 1.866769e-10 		    | 6	(End of speed limit 80km/h)										|
| 1.916227e-12	        | 14	(Stop)			 				|
| 5.450136e-15           | 32  (End of all speed and passing limits)    							|

Note, mathematically the sum of all probabilities has to equal to 1.0. If the first entry is 1.0, that automatically imples all the other classes has a probability of 0. Therefore, any number such as 4.889838e-13 i simply munerical error from computation, and can be treated as 0. Here are the probability for the other 7 images:

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/28.png "Children crossing")

The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| 28  (Children crossing)								|
| 3.297621e-26		    | 29 	(Bicycles crossing)									|
| 4.582971e-28 		    | 24	(Road narrows on the right)										|
| 2.339243e-29 	        | 20	(Dangerous curve to the right)			 				|
| 3.876115e-34           | 23  (Slippery road)    							|

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/8.png "Speed limit 120")

The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| 8  (	Speed limit 120km/h)								|
| 1.147554e-18		    | 7 (Speed limit 100km/h)										|
| 3.509652e-21 		    | 0	(Speed limit 20km/h)										|
| 3.617397e-31	        | 5		(Speed limit 80km/h)			 				|
| 2.273280e-31          | 4    (Speed limit 70km/h)  							|

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/25.png "Road Work")

The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| 25  (Road work)									|
| 4.924969e-12 		    | 22 	(Bumpy road)									|
| 5.002733e-13		    | 24	(Road narrows on the right)										|
| 1.617172e-14	        | 29 (Bicycles crossing)					 				|
| 1.671352e-17           | 30   (Beware of ice/snow)   							|

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/33.png "Turn right ahead")

The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 0.999999         			| 33  (Turn right ahead)									|
| 3.729876e-07 		    | 40 	(Roundabout mandatory)								|
| 2.332406e-07		    | 39	(Keep left)										|
| 2.210692e-07 	        | 35	(Ahead only)				 				|
| 4.265740e-08           | 37  (Go straight or left)   							|

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/14.png "Stop")

The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 1.0        			| 14  (Stop)									|
| 8.224815e-13  		    | 1 (Speed limit 30km/h)										|
| 3.939823e-15 		    | 29	(Bicycles crossing)										|
| 6.252956e-21	        | 0	(Speed limit 20km/h)				 				|
| 2.293709e-21            | 5  (Speed limit 80km/h)   							|


![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/4.png "Speed limit 70")
The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| 4 	(Speed limit 70km/h)								|
|  8.224815e-13		    | 1 	(Speed limit 30km/h)										|
| 3.939823e-15  		    | 0	(Speed limit 20km/h)										|
| 6.252956e-21 	        | 39	(Keep left)				 				|
| 2.293709e-21             | 35 (Ahead only)     							|

![alt text](https://raw.githubusercontent.com/paradox56/CarND-Traffic-Sign-Classifier-Project/master/example_images/11.png "Right-of-way")

The top-5 softmax probability is the following:

| Probability         	|     Predicted Class ID		       					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| 11 	(Right-of-way at the next intersection)								|
|  1.279899e-28		    | 30 	(Beware of ice/snow)										|
| 2.043567e-34 		    | 6	(End of speed limit 80km/h)										|
| 1.646665e-38 	        | 27	(Pedestrians)				 				|
| 1.418140e-38            | 18 (General caution)     							|



```

```
