# Write-Up / README

In this project, we trained a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.


[image_1]: ./docs/misc/screen_1.png
[image_2]: ./docs/misc/screen_2.png
[image_3]: ./docs/misc/screen_3.png
[image_4]: ./docs/misc/screen_4.png
[image_5]: ./docs/misc/screen_5.png


## Data collection
As with all machine learning projects, the more data you have the better your model will perform. So, the first step, in addition to the data provided, we had to add more data. In this manner, I added four more runs, each with more or less 500 pictures. As was sugested in the class, we had to collect the dat through the simulator. The 3rd run is shown in the pictures below, where the hero walks in a fully-conneted octagonal shape, and the crowd would spawn to distract, while the drone goes above taking pictures in follow mode and patrol mode.

Patrol points, hero walking points and crowd spaws:
![alt text][image_1]

Patrol mode: 
![alt text][image_5]

Follow mode:
![alt text][image_4]


## Network architecture

Semantic Segmentation of an image is to assign each pixel in the input image a semantic class in order to get a pixel-wise dense classification. While semantic segmentation / scene parsing has been a part of the computer vision community since 2007, but much like other areas in computer vision, major breakthrough came when fully convolutional neural networks were first used by 2014 Long et. al. to perform end-to-end segmentation of natural images.

In this project we had to implement a Fully Convolutional Network. FCNs are being used for semantic segmentation of natural images, for multi-modal medical image analysis and multispectral satellite image segmentation, self-driving cars, robotics and pretty much everywhere else where computer vision comes to play.

## Training

Lorem Ipsum

```py
learning_rate = 0.025
batch_size = 16
num_epochs = 50
steps_per_epoch = 250
validation_steps = 100
workers = 8
```

## Results

Lorem Ipsum

## Parameter tunning

Lorem Ipsum

## Final Results

Lorem Ipsum

## Future improvement

Lorem Ipsum