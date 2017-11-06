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

FCNs are vey useful when in addition to clasification (what is the image showing), we want the localisation too (where in the picture is something of interest).

During my initial test, i want with 2 encoder and 2 decoder layers connected via 1X1 convolution layer. Later on, to improve the accuaracy i decided to go with 3 encoder and 3 decoder layers. This seemts that it increased accuracy, but, still i would think for the depth versus performance when doing thise kind of architectures.

Here is the network I used:
```python
def fcn_model(inputs, num_classes):
    encode_layer_1 = encoder_block(inputs, 32, 2)
    encode_layer_2 = encoder_block(encode_layer_1, 64, 2)
    encode_layer_3 = encoder_block(encode_layer_2, 128, 2)
    convol_layer_1 = conv2d_batchnorm(encode_layer_3, 256, kernel_size=1, strides=1)
    decode_layer_1 = decoder_block(convol_layer_1, encode_layer_2, 128)
    decode_layer_2 = decoder_block(decode_layer_1, encode_layer_1, 64)
    decode_layer_3 = decoder_block(decode_layer_2, inputs, 32)
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decode_layer_3)
```

## Training

Training is the core of any machine learning algorithm. And its very compute intensisive. And because it needa a lot of concurrency and paralelism to compute different and huge matrices, it is suitable to use GPU rather than CPU. 

We are given a 50$ cuppon for AWS instances, but i decided to save that so i can use for more intensive task down the rod, and use my own computers GPU.

I own a Dell XPS 9550, with Nvidia GTX 960M with compute capability 4.0. I already have GPU versions of many deep learning libraries including: TensorFlow, PyTorch (my favourite), Caffe2 and Cognetive toolkit.

Since my GPU has only 2GB of memory, i had to decrease the batch.size to 16 so, during the training does not run out of memory. And with the parameters that were given (i did not change in the begning), it took less than 1 hour to finish.
## Results

First results were not so good, reaching an accuraacy of 35%. This because, in first run, i bearly changed anything, except the model layers and set learning rate very low.

## Parameter tunning

In order to increase the accuracy, i added more than 2000 pictures, and i changed other parameters as seen below:
```python
learning_rate = 0.025
batch_size = 16
num_epochs = 50
steps_per_epoch = 250
validation_steps = 100
workers = 8
```
## Final Results

After an intensive ~5hr of training, i reached an acceptable accuracy of more than 41%

## Future improvement

The whole model is made with Keras and Tensorflow as backend.

As a huge fan of PyTorch, while learning PyTorch, i want to use this model with it. When the model is in PyTorch, i can use ONNX to easily translate it in Caffe2 and/or Cognetive toolkit too :)