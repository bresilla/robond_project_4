# Write-Up / README

In this project, we trained a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.


[image_1]: ./docs/misc/screen_1.png
[image_2]: ./docs/misc/screen_2.png
[image_3]: ./docs/misc/screen_3.png
[image_4]: ./docs/misc/screen_4.png
[image_5]: ./docs/misc/screen_5.png
[image_6]: ./docs/misc/drawing.png
[image_7]: ./docs/misc/train.png



## Data collection
As with all machine learning projects, the more data you have the better your model will perform. So, the first step, in addition to the data provided, we had to add more data. In this manner, I added four more runs, each with more or less 500 pictures. As was sugested in the class, we had to collect the dat through the simulator. The 3rd run is shown in the pictures below, where the hero walks in a fully-conneted octagonal shape, and the crowd would spawn to distract, while the drone goes above taking pictures in follow mode and patrol mode.

Patrol points, hero walking points and crowd spaws:
![alt text][image_1]

Patrol mode: 
![alt text][image_5]

Follow mode:
![alt text][image_4]


## CNNs and FCNs

CNNs are a specific type of neural networks that take into the consideration the spatial data. Mostly used in computer vision, but not neccesary just there, as CNNs are found in spech recognition, sentence generation, translation and so on. There re four main operations in CNNs:

#### 1. Convolution
Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data

#### 2. Non Linearity (ReLU)
ReLU is an element wise operation and replaces all negative pixel values in the feature map by zero. The aim of ReLU is to introduce non-linearity in our CNN.
#### 3. Pooling or Sampling
Pooling reduces the dimensiaolity of each feature map, but still retaining most important information.
#### 4. Fully Connected Layer (Classification)
Fully conneted layer is a simple NN layer. It flatens all the inputs of last layer and


FCNs are vey useful when in addition to clasification (what is the image showing), we want the localisation too (where in the picture is something of interest). Semantic Segmentation of an image is to assign each pixel in the input image a semantic class in order to get a pixel-wise dense classification. While semantic segmentation / scene parsing has been a part of the computer vision community since 2007, but much like other areas in computer vision, major breakthrough came when fully convolutional neural networks were first used to perform end-to-end segmentation of natural images. FCNs are being used everywhere, for segmentation of natural images, for multi-modal medical image analysis and multispectral satellite image segmentation, self-driving cars, robotics and pretty much everywhere else where computer vision comes to play.

The difference form CNNs is that FCNs is that insted of image-level clasification, it does a pixel-level classification by firstly doing into network by different Convolution/Linearty/Pooling layers (the encoder block), to reach the 1x1 convolution then again going in the oposite again - transpose convolution - by  Convolution/Linearty/Pooling layers (the decoder block). Those block esencially use Depthwise Separable convolutions. In a depthwise convolution, the kernels of each filter are applied separately to each channel and the outputs are concatenated. Then, the pointwise convolution is applied. This greatly reduces the number of parameters that are required while still keeping efficiency and not destroying cross-channel features. So main operations in FCNs are:

#### 1. Encoder block
Goal of encoder block is to extract features from the image
#### 2. 1x1 convolution
This is the most important and distinguished feature of FCNs compared to CNNs, as in 1x1 convolution this substitutes the fully conected layer, but here instead of beinf a 2D Tensor, is 4D, thus saving spatial information.
#### 3. Decoder block
While the decoder upscales the input of encoder (technically 1x1 convolution layer) to reach same size as original image by using upsampling. Thus segmentig pixelwise elements on the image.


## Network architecture

During my initial test, i went with 2 encoder and 2 decoder layers connected via 1X1 convolution layer. Later on, to improve the accuaracy i decided to go with 3 encoder and 3 decoder layers. This seemts that it increased accuracy, but, still i would think for the depth versus performance when doing thise kind of architectures.

Here is the network I used:

![alt text][image_6]
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

We are given a 50$ cuppon for AWS instances, but i decided to save that so i can use for more intensive task down the road, and use my own computer's GPU.

I own a Dell XPS 9550, with Nvidia GTX 960M with compute capability 4.0. I already have GPU versions of many deep learning libraries including: TensorFlow, PyTorch (my favourite), Caffe2 and Cognetive toolkit.

Since my GPU has only 2GB of memory, i had to decrease the batch.size to 16 so, during the training does not run out of memory. And with the parameters that were given (i did not change in the begning), it took less than 1 hour to finish.

![alt text][image_7]

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
- learning rate: is the tweaking value on how much weights can change after one epoch
- batch size: is the number of simultaneus images shown to model before weights are updated (highly dependant on memory of procesing hardware, be that GPU, CPU, TPU, APU or whatever is there nowadays for deep-learning)
- epochs: is the number of hoa many cycles the model goes in the same data
- workers: is number of threds (again depends on processing hardware)

## Final Results

After an intensive ~5hr of training, i reached an acceptable accuracy of more than 41%
Its important to note again that, the model is trained to follow just a specific hero, this model would not work good to follow another hero, or object, or other animal, or other thing. For that it needs to be trained, extract feature, and then reconise it and follow.

## Improvement

Again, as all machine-learning agorithms, for the network to have better accuracy, would be better to use more data. Them more date the better the model.

Another would be, using different regularisation techniques, momentum, L2 regularisaition (weight decay) and/or even drop out - inverted dropout to prevent overfiting 

We can certainly go deeper with layers, but we will face vanishing weights problem at some point.

Another improvement could be, using differnt hero model, so the model is not bound to just a specific person. This might be tricky, as the drone has to follow just a specific person, and when the distractor spawn interfere, the model might get confused. But with enough data, and a model that checks not just a specific picture but considers previous frames (in a continious manner), would certainly make this possible.

More epochs prpbpbly would have been better, as it looks from the graph of last epoch, the model was still to stabilise.

## Future work

The whole model is made with Keras and Tensorflow as backend. As a huge fan of PyTorch, while learning PyTorch, i want to use this model with it.

And when the model is in PyTorch, i can use ONNX to easily translate it in Caffe2 and/or Cognetive toolkit too.
I like the idea of ONNX, as, sometimes some models just work better in another framework. AS of now, ONNX, i think supports only PyTorch, Caffe2 and Cognetive, but i hope a tool like that will be soon supported from tensorflow.
