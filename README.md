# conv-batchnorm
Convolutional batch normalization for computing local gain normalization with translation invariant parameters.

This package implements a convolutional batch normalization layer for neural network models in Keras.

Most batch normalization implementations compute statistics for mean and variance normalization across axes 
complementary to the axes that have learned scale parameters. For instance, the most common batch normalization
implements a parameter per channel and estimates the mean and variance of each channel by marginalizing across
the batch and spatial dimensions.

Suppose however that your neural network has an input that is not completely translation invariant across the batch.
For instance, imagine a video taken from the front of a car at sunrise. If the batch contains sequential frames,
all of the bottom portion of each frameo will have long shadows and be considerably darker than the top half of the
frame which contains bright sunlight. If we normalize across the entire layer, we may shove all pixels below the
horizon below threshold, and only process the well-illuminated parts of the scene.

In the biological visual system, what happens instead is a form of local gain normalization, where neurons sensitive
to different locations in space adapt somewhat independently.

An easy way to implement this is to just use normal batch norm layers and collapse the spatial dimensions together
with the channel dimensions. This will compute statistics for each channel and each x, y location independently.
However, this will also learn a scale parameter for each x and y position. In addition to learning possibly
thousands of additional parameters that may be poorly constrained, this also renders the network no longer
fully convolutional, and the receptive fields of the network will vary with space.

To solve this, here is a simple modification of batch normalization that performs local normalization but with
the same number of parameters obtained with simple channel-wise batch normalization.
