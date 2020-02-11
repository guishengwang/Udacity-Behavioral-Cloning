#  **Behavioral Cloning** 

## Udacity Self-Driving Car Nanodegree Project 4



**The goals / steps of this project are the following:**

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image10]: ./examples/csv_header.png "csv header Image"
[image11]: ./examples/Nvidia_CNN.png "CNN"
[image12]: ./examples/origin.jpg "origin"
[image13]: ./examples/cropping.jpg "cropping"
[image14]: ./examples/loss_chart.png "loss chart"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files as below:

File Name | Description
----------|-----------
model.py |all folder path varialbes applied for udacity workspace
model.h5 |              generated from workspace and works for track one in simulator
writeup_report.md |      summarizing the results and the same contents as README.MD
drive.py        |       for driving the car in autonomous mode, original file provided by Udacity
video.mp4 |  video under autonomous mode on track one

Additional files

File Name | Description
----------|-----------
model.ipynb | all folder path varialbles applied for my local drive
model.html |  file exported from Jupytor Notebook with result from model.ipynb 
model_local_track1.h5 | generated from my laptop and was not compatible with simulator on workspace

At begining I ran the code successfuly at my laptop to save the data to h5 file which unfornately was not compatible with the simulator on workplace. I thought it was due to different versions of keras and I tried to downgrade keras from 2.3.1 to version 2.2.4 on my laptop and generated the model.h5 again, but it still didnot work. 

Later, I found out it seems like other students also encountered such problem and someone suggested to run the code on workspace and it did work for me too.

#### 2. Submission includes functional code

The car can be driven autonomously around the track One by executing 
```sh
python drive.py model.h5
```

#### 3. Remove header of csv file

There is a line of header in the file of "driving_log.csv" which should be removed from the list created from csv file.
![alt text][image10]

After appending all lines into the list, the first line (header in csv file) was pop out of the list, otherwise it will cause problem during reading images.

```sh
lines=[]
with open("../p4_data/data/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)
```

#### 4. RGB format of image

The class material mentioned about the OpenCV read images file into BGR formet, instread of using OpenCV and convert to RGB format, I used the mpimg funciton from matplotlib. We have to deal with similar problem on the first two projects of lane finding. 

```sh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
...
...
                    name = '../p4_data/data/IMG/'+batch_sample[i].split('/')[-1]
                    image = mpimg.imread(name)
                    angle = float(batch_sample[3])
```




**_I was spending lots of time debugging and later I realized lot of time was spent on these 3 issues below_**

* Run the model.py under workspace to generate model.h5 to make sure it work with simulator
* Remove the header from csv file 
* Make sure the images read into RGB format


### Model Architecture and Training Strategy

#### 1. Model architecture 

Nvidia model is being used for this project since this has been proven to be effective according many other students. The architecture of the model is illustrated as below but the dimensional of each layer are different. 


**Same Architecture but dimension for Reference only**

![alt text][image11]

The images has been cropped to the size of (90,320) as suggested by the instrutor to keep the middle part of the image only. To remove the top section with sky, tree and lower section with engine hood. 

Original image | Cropped image
----------|-----------
![alt text][image12] | ![alt text][image13]



#### 2. Loading Data 

The dataset provided by Udacity was used as training data. The whole dataset was splited to training set (80%) and validation set(20%) by using sklearn

```sh
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
```
Below are the number of training and validation samples
train samples : 6428
validation samples:  1608

#### 3. Add left and right images to combat the overfitting

A correction factor of 0.2 was used for left and right images, below is the code to read left and right image and calcualtion of the angle with correction factor.

```sh
                for i in range(3):
                    name = '../p4_data/data/IMG/'+batch_sample[i].split('/')[-1]
                    image = mpimg.imread(name)
                    angle = float(batch_sample[3])
                    images.append(image)
                         
                    if(i==0):
                        angles.append(angle)
                    elif(i==1):
                        angles.append(angle+correction)
                    elif(i==2):
                        angles.append(angle-correction)

```

#### 4. Flipping images to help left turn bias

According the instruction on class, to help the left turn bias, the images (middle, left and right) was flipped and added into training set. 

```sh
                    images.append(cv2.flip(image,1))
                    if(i==0):
                        angles.append(angle*-1)
                    elif(i==1):
                        angles.append((angle+correction)*-1)
                    elif(i==2):
                        angles.append((angle-correction)*-1)

```


#### 5. Visualizing loss

Below is the chart to visualize the training and validation loss for each epoch. After 5 epoch, the training and validation loss has been dropped to a similar level.

![alt text][image14]

#### 6. Use Generators

Generator is new concept/method for me and I tried to use different batch size and find out for my laptop, 32 is a good number to use to prevent memory crash and I used the same batch size on workspace.


### Discussion

Compared to the previous project of traffic sign, this one is less complicated but more interesing. The challenge is the track two, unfortunately I didnot have time to finish it, already overdue for nearlly a month. By using the dataset provided by Udacity and all the tricks from the class, the model.h5 generated by Nvidia CNN could drvie the car through the track one successfully. 

I started to use github and markdown file to write the report instead of pdf. It aslo took me some time to learn the git,github and version control. Appreciate the help from my mentor, guide me through the project and the learning process. 
