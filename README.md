# Behavioral Cloning Project
## Steps 
• Use the simulator to collect data of good driving behavior  
• Build, a convolution neural network in Keras that predicts steering angles from
images  
• Train and validate the model with a training and validation set  
• Test that the model successfully drives around track one without leaving the road  
• Summarize the results with a written report  
## File Description:  
1. Files can be used to run the simulator in autonomous mode   
• model.py containing the script to create and train the model  
• drive.py for driving the car in autonomous mode  
• model.h5 containing a trained convolution neural network  
2. The car can be driven autonomously around the track by executing ```python drive.py model.h5```
3. The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains
comments to explain how the code works.
## Data recording
The data necessary for the task of behavioral cloning is recorded in training mode. 3 cameras
are located in the center, left, and right of the car facing forward (see picture below). They
collect images and also record the current steering angle which result of me navigating the
car with the arrow keys of my keypad. All images are used while the side images are altered
by adding/subtracting a correction vector of 0.35. This allows for more data collection and
teaches the car to steer back to the center when drifting off. In order to augment the data
even more and counteract the left-facing drift inherent due to the counter-clockwise facing
track, pictures are flipped vertically.  
Furthermore, the images are cropped so that distracting parts that carry no information like
pixels of the bottom (front lid of the car) or the top (sky and trees) of the pictures are
neglected. Specifically, the upper 70 pixel rows are cropped off and the bottom 25 pixel rows
as well.
## Model Architecture and Training Strategy
Training data was chosen to keep the vehicle driving on the road. I used a combination of
center lane driving, recovering from the left and right sides of the road. I evaluated different
training sets with differing sizes. So I began with a set, driving the track once in one direction
(dataset 1). Then I decided to record training data consisting of driving the track twice in
each direction (dataset 2). The third training set I recorded consisted of driving the track
twice in each direction plus more data recorded driving the difficult parts like shape curves
and the bridge (dataset 3). Interestingly, the success of driving autonomously the whole
track was given by using the second data set, which was gathered by driving the track twice
in each direction. The more comprehensive data collection of dataset 3 led to less satisfying
results making the car get off the track.  
The model I constructed consists of a convolution neural network with 5x5 and 3x3 filter
sizes and depths between 24 and 64. The model takes advantage of RELU-layers after each
convolution to introduce nonlinearity. Furthermore, the data is normalized in the model
using a Keras lambda layer. The model used an Adam optimizer, so the learning rate was not
tuned manually.  
The overall strategy for deriving a model architecture was to make the car stay on track in
autonomous mode. My approach was to use a convolution neural network model similar to
the NVIDIA architecture. This network is proven to solve the task of autonomous driving very
well without taking too much time for computation, which is crucial for real world
application. I modified the architecture in a way so that the performance turned out to be
better for the specific task. I finally chose an architecture of 5 layers consisting of  
• 24 5x5 filters  
• 32 5x5 filters  
• 64 5x5 filters  
• 64 3x3 filters  
• 64 3x3 filters  
followed by RELU-layers as activation, a “Flatten” layer, and 4 “Dense”-layers to reduce the
dimensionality of the fully connected layers finally to one dimension. This output reflects the
regression task of behavioral cloning. For learning the driving behavior successfully and
staying on track, there was just one training epoch necessary. Training on more epochs
made the validation accuracy go up. Attempts to introduce dropout layers did not make the
model better even when training on multiple epochs. In those cases the vehicle fell off the
track at some point, usually in sharp curves.  
I finally randomly shuffled the data set and put 20% of the data into a validation set. The
validation set helped determine if the model was over- or under-fitting. Training loss after 1
epoch turned out to be 0.086 and validation loss 0.089. The ideal number of epochs was 1
because validation accuracy increased with a higher number of epochs.
