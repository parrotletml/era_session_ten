# This article talks about training of custom ResNet architecture for CIFAR 10

**Highlights**
* Trained a custom form of ResNet to reach **94.12%** accuracy
* Uses Mixed Precision Training to reduce training time from `54s` per epoch to `6s` per epoch
* Used Lable Smoothing to improve model convergence
* One Cycle LR to get faster convergence
* [Notebook](./CIFAR_DAVIDNET_94.12.ipynb) with experiment attached
* [Weights](./weights/cifar_davidnet_94.12.onnx) for trained model attached

---

### Table of Contents
- [Model Architechture](#model-architechture)
  - [Model Summary](#model-summary)
  - [Training Summary](#training-summary)
- [Techniques Used](#techniques-used)
  - [GradCam](#gradcam)
  - [Label Smoothing](#label-smoothing)
  - [Grad Scaler](#grad-scaler)
  - [How do you decide on a learning rate](#how-do-you-decide-on-a-learning-rate)
- [Augumented Images](#augumented-images)
- [Graphs](#graphs)
- [Confusion Matrix](#confusion-matrix)
- [Incorrect Predictions](#incorrect-predictions)
- [References](#references)

___

## Model Architechture

![architecture](./assets/cifar_davidnet_94.12.onnx.svg)
	

  * ### Model Summary
    ```
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 32, 32]           1,728
              WyConv2d-2           [-1, 64, 32, 32]               0
           BatchNorm2d-3           [-1, 64, 32, 32]             128
                  ReLU-4           [-1, 64, 32, 32]               0
                Conv2d-5          [-1, 128, 32, 32]          73,728
              WyConv2d-6          [-1, 128, 32, 32]               0
             MaxPool2d-7          [-1, 128, 16, 16]               0
           BatchNorm2d-8          [-1, 128, 16, 16]             256
                Conv2d-9          [-1, 128, 16, 16]         147,456
             WyConv2d-10          [-1, 128, 16, 16]               0
          BatchNorm2d-11          [-1, 128, 16, 16]             256
               Conv2d-12          [-1, 128, 16, 16]         147,456
             WyConv2d-13          [-1, 128, 16, 16]               0
          BatchNorm2d-14          [-1, 128, 16, 16]             256
              WyBlock-15          [-1, 128, 16, 16]               0
               Conv2d-16          [-1, 256, 16, 16]         294,912
             WyConv2d-17          [-1, 256, 16, 16]               0
            MaxPool2d-18            [-1, 256, 8, 8]               0
          BatchNorm2d-19            [-1, 256, 8, 8]             512
                 ReLU-20            [-1, 256, 8, 8]               0
               Conv2d-21            [-1, 512, 8, 8]       1,179,648
             WyConv2d-22            [-1, 512, 8, 8]               0
            MaxPool2d-23            [-1, 512, 4, 4]               0
          BatchNorm2d-24            [-1, 512, 4, 4]           1,024
               Conv2d-25            [-1, 512, 4, 4]       2,359,296
             WyConv2d-26            [-1, 512, 4, 4]               0
          BatchNorm2d-27            [-1, 512, 4, 4]           1,024
               Conv2d-28            [-1, 512, 4, 4]       2,359,296
             WyConv2d-29            [-1, 512, 4, 4]               0
          BatchNorm2d-30            [-1, 512, 4, 4]           1,024
              WyBlock-31            [-1, 512, 4, 4]               0
            MaxPool2d-32            [-1, 512, 1, 1]               0
                 View-33                  [-1, 512]               0
               Linear-34                   [-1, 10]           5,130
    ================================================================
    Total params: 6,573,130
    Trainable params: 6,573,130
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 8.70
    Params size (MB): 25.07
    Estimated Total Size (MB): 33.78
    ----------------------------------------------------------------
    ```

  * ### Training Summary
    ```
    Using Device: cuda
    Epochs: 24
    Lr: 0.045
    Max Lr: 0.45
    Batch Size: 512
    Dropout: False
    Dropout Ratio: 0.1
    Momentum: 0.9
    Weight Decay: 0.0005
    Use L1: False
    L1 Lambda: 1e-07
    -------------------------------------------------------------------------
    | Epoch | LR       | Time    | TrainLoss | TrainAcc | ValLoss  | ValAcc |
    -------------------------------------------------------------------------
    |     1 | 0.045000 | 00m 06s | 1.815593  | 47.66  % | 1.584001 | 64.03% |
    |     2 | 0.102857 | 00m 06s | 1.553491  | 65.82  % | 1.738120 | 58.42% |
    |     3 | 0.160714 | 00m 06s | 1.465533  | 72.46  % | 1.424718 | 74.25% |
    |     4 | 0.218571 | 00m 06s | 1.393782  | 76.55  % | 1.419394 | 75.95% |
    |     5 | 0.276429 | 00m 06s | 1.332046  | 79.9   % | 1.430989 | 74.53% |
    |     6 | 0.334286 | 00m 06s | 1.290172  | 82.33  % | 1.310123 | 80.82% |
    |     7 | 0.392143 | 00m 06s | 1.277501  | 82.91  % | 1.284005 | 81.85% |
    |     8 | 0.450000 | 00m 06s | 1.254159  | 84.08  % | 1.239892 | 85.28% |
    |     9 | 0.423596 | 00m 06s | 1.227912  | 85.41  % | 1.285054 | 81.59% |
    |    10 | 0.397191 | 00m 06s | 1.208307  | 86.32  % | 1.275494 | 81.61% |
    |    11 | 0.370787 | 00m 06s | 1.19589   | 87.02  % | 1.357005 | 77.8 % |
    |    12 | 0.344382 | 00m 06s | 1.183011  | 87.54  % | 1.279942 | 81.88% |
    |    13 | 0.317978 | 00m 06s | 1.169725  | 88.33  % | 1.239580 | 83.98% |
    |    14 | 0.291574 | 00m 06s | 1.159143  | 88.78  % | 1.187427 | 86.8 % |
    |    15 | 0.265169 | 00m 06s | 1.147206  | 89.44  % | 1.175734 | 87.78% |
    |    16 | 0.238765 | 00m 06s | 1.133442  | 90.31  % | 1.183655 | 87.03% |
    |    17 | 0.212360 | 00m 06s | 1.123376  | 90.84  % | 1.179832 | 87.52% |
    |    18 | 0.185956 | 00m 06s | 1.109443  | 91.63  % | 1.199403 | 86.09% |
    |    19 | 0.159551 | 00m 06s | 1.095237  | 92.41  % | 1.125787 | 90.08% |
    |    20 | 0.133147 | 00m 06s | 1.080023  | 93.36  % | 1.114314 | 91.1 % |
    |    21 | 0.106743 | 00m 06s | 1.065799  | 94.04  % | 1.094508 | 92.13% |
    |    22 | 0.080338 | 00m 06s | 1.049066  | 94.99  % | 1.078320 | 92.92% |
    |    23 | 0.053934 | 00m 06s | 1.031054  | 96.15  % | 1.064695 | 93.52% |
    |    24 | 0.027529 | 00m 06s | 1.017438  | 96.84  % | 1.054367 | 94.12% |
    -------------------------------------------------------------------------
    ```
---
## Techniques Used


### GradCam

  * ### Output for current model

    ![gradcam](./assets/gradcam.png)

  * ### Introduction 
    CAM known as Grad-Cam. Grad-Cam published in 2017, aims to improve the shortcomings of CAM and claims to be compatible with any kind of architecture. This technique does not require any modifications to the existing model architecture and this allows it to apply to any CNN based architecture, including those for image captioning and visual question answering. For fully-convolutional architecture, the Grad-Cam reduces to CAM.
  * ### How do we solve this ?
    Grad-Cam, unlike CAM, uses the gradient information flowing into the last convolutional layer of the CNN to understand each neuron for a decision of interest. To obtain the class discriminative localization map of width u and height v for any class c, we first compute the gradient of the score for the class c, yc (before the softmax) with respect to feature maps Ak of a convolutional layer. These gradients flowing back are global average-pooled to obtain the neuron importance weights ak for the target class.
	
![weights](./assets/Weights_1.png)

After calculating ak for the target class c, we perform a weighted combination of activation maps and follow it by ReLU.
	    
![Linear_Combination](./assets/Linear_Combination.png)
	
This results in a coarse heatmap of the same size as that of the convolutional feature maps. We apply ReLU to the linear combination because we are only interested in the features that have a positive influence on the class of interest. Without ReLU, the class activation map highlights more than that is required and hence achieve low localization performance.


### Label Smoothing 

  * ### What is label smoothing
	  When using deep learning models for classification tasks, we usually encounter the following problems: overfitting, and overconfidence. Overfitting is well studied and can be tackled with early stopping, dropout, weight regularization etc. On the other hand, we have less tools to tackle overconfidence. Label smoothing is a regularization technique that addresses both problems.

  * ### Overconfidence and Calibration
	A classification model is calibrated if its predicted probabilities of outcomes reflect their accuracy. For example, consider 100 examples within our dataset, each with predicted probability 0.9 by our model. If our model is calibrated, then 90 examples should be classified correctly. Similarly, among another 100 examples with predicted probabilities 0.6, we would expect only 60 examples being correctly classified.
	 ### Model calibration is important for
	* model interpretability and reliability
	* deciding decision thresholds for downstream applications
	* integrating our model into an ensemble or a machine learning pipeline
	
	An overconfident model is not calibrated and its predicted probabilities are consistently higher than the accuracy. For example, it may predict 0.9 for inputs where the accuracy is only 0.6. Notice that models with small test errors can still be overconfident, and therefore can benefit from label smoothing.
	
  * ### Formula of Label Smoothing
	Label smoothing replaces one-hot encoded label vector y_hot with a mixture of y_hot and the uniform distribution: 
      ```bash
    y_ls = (1 - α) * y_hot + α / K
      ```
	here K is the number of label classes, and α is a hyperparameter that determines the amount of smoothing. If α = 0, we obtain the original one-hot encoded y_hot. If α = 1, we get the uniform distribution.
	
  * ### Motivation of Label Smoothing
	Label smoothing is used when the loss function is cross entropy, and the model applies the softmax function to the penultimate layer’s logit vectors z to compute its output probabilities p. In this setting, the gradient of the cross entropy loss function with respect to the logits is simply


### Grad Scaler
 * ### Introduction 
	16-bit precision may not always be enough for some computations. One particular case of interest is representing gradient values, a great portion of which are usually small values. Representing them with 16-bit floats often leads to buffer underflows (i.e. they’d be represented as zeros). This makes training neural networks very unstable. GradScalar is designed to resolve this issue. It takes as input your loss value and multiplies it by a large scalar, inflating gradient values, and therefore making them represnetable in 16-bit precision. It then scales them down during gradient update to ensure parameters are updated correctly. This is generally what GradScalar does. But under the hood GradScalar is a bit smarter than that. Inflating the gradients may actually result in overflows which is equally bad. So GradScalar actually monitors the gradient values and if it detects overflows it skips updates, scaling down the scalar factor according to a configurable schedule. (The default schedule usually works but we may need to adjust that for our use case.)

	Using GradScalar is very easy in practice:
	    
![gradscaler](./assets/gradscaler.png)
	
Note that we first create an instance of GradScalar. In training loop we call GradScalar.scale to scale the loss before calling backward to produce inflated gradients, we then use GradScalar.step which (may) update the model parameters. We then call GradScalar.update which performs the scalar update if needed. 

## How do you decide on a learning rate?

Let's start, now the question is how do you decide the learning rate right ? the answer would be if it's too slow, your neural net is going to take forever to learn. But if it's too high, each step you take will go over the minimum and you'll never get to an acceptable loss. Worse case is, a high learning rate could lead you to an increasing loss until it reaches null value. Now you might think why is this happening ? the theory says that If your gradients are really high, then a high learning rate is going to take you to a spot that's so far away from the minimum you will probably be worse than before in terms of loss. Even on something as simple as a parabola, see how a high learning rate quickly gets you further and further away from the minima.

![Lr_1](./assets/Lr_1.png)

So we have to pick exactly the right value, not too high and not too low. For a long time, it's been a game of try and see, but in this article another approach is presented. Over an epoch begin your SGD with a very low learning rate (like 10−8) but change it (by multiplying it by a certain factor for instance) at each mini-batch until it reaches a very high value (like 1 or 10). Record the loss each time at each iteration and once you're finished, plot those losses against the learning rate. You'll find something like this:
	
![Lr_2](./assets/Lr_2.png)
	
The loss decreases at the beginning, then it stops and it goes back increasing, usually extremely quickly. That's because with very low learning rates, we get better and better, especially since we increase them. Then comes a point where we reach a value that's too high and the phenomenon shown before happens. Looking at this graph, what is the best learning rate to choose? Not the one corresponding to the minimum.

Why? Well the learning rate that corresponds to the minimum value is already a bit too high, since we are at the edge between improving and getting all over the place. We want to go one order of magnitude before, a value that's still aggressive (so that we train quickly) but still on the safe side from an explosion. In the example described by the picture above, for instance, we don't want to pick 10−1 but rather 10−2. This method can be applied on top of every variant of SGD, and any kind of network. We just have to go through one epoch (usually less) and record the values of our loss to get the data for our plot.


---
## How do we code it ?

Well it's pretty simple when we have used the fastai library.

```python
learner.lr_find()
learner.sched.plot()
```
The first one is that we won't really plot the loss of each mini-batch, but some smoother version of it. If we tried to plot the raw loss, we would end up with a graph like this one:

![Lr_3](./assets/Lr_3.png)
	
Even if we can see a global trend (and that's because I truncated the part where it goes up to infinity on the right), it's not as clear as the previous graph. To smooth those losses we will take their exponentially weighed averages. It sounds far more complicated that it is and if you're familiar with the momentum variant of SGD it's exactly the same. At each step where we get a loss, we define this average loss by

![Lr_4](./assets/Lr_4.png)
	
where β is a parameter we get to pick between 0 and 1. This way the average losses will reduce the noise and give us a smoother graph where we'll definitely be able to see the trend. This also also explains why we are too late when we reach the minimum in our first curve: this averaged loss will stay low when our losses start to explode, and it'll take a bit of time before it starts to really increase.

If you don't see the exponentially weighed behind this average, it's because it's hidden in our recursive formula. If our losses are l0,…,ln then the exponentially weighed loss at a given index i is

![Lr_5](./assets/Lr_5.png)

so the weights are all powers of β. If remember the formula giving the sum of a geometric sequence, the sum of our weights is

![Lr_6](./assets/Lr_6.png)
	
so to really be an average, we have to divide our average loss by this factor. In the end, the loss we will plot is

![Lr_7](./assets/Lr_7.png)
	
this doesn't really change a thing when i is big, because βi+1 will be very close to 0. But for the first values of i, it insures we get better results. This is called the bias-corrected version of our average.

The next thing we will change in our training loop is that we probably won't need to do one whole epoch: if the loss is starting to explode, we probably don't want to continue. The criteria that's implemented in the fastai library and that seems to work pretty well is:

```
current smoothed loss > 4 × minimum smoothed loss
```
Lastly, we need just a tiny bit of math to figure out by how much to multiply our learning rate at each step. If we begin with a learning rate of lr0 and multiply it at each step by then at the i-th step, our learning rate will be

```
ri = lr0 × qi
```
Now, we want to figure out q knowing lr0 and lr N−1 (the final value after N steps) so we isolate it:

![Lr_8](./assets/Lr_8.png)
	
Why go through this trouble and not just take learning rates by regularly splitting the interval between our initial value and our final value? We have to remember we will plot the loss against the logs of the learning rates at the end, and if we take the log of our lri we have
	
![Lr_9.png](./assets/Lr_9.png)
	
which corresponds to regularly splitting the interval between our initial value and our final value... but on a log scale! That way we're sure to have evenly spaced points on our curve, whereas by taking

![Lr_10.png](./assets/Lr_10.png)

we would have had all the points concentrated near the end (since lrN−1 is much bigger than lr0

---
### Augumented Images

![augumented_images](./assets/augumented_images.png)

---
### Graphs
	
![graphs](./assets/graphs.png)

---
### Confusion Matrix
	
![cw](./assets/cw.png)

---

### Incorrect Predictions
	
![images](./assets/images.png)
	
---

### References
* https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
* https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
* https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06	
* https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
* https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html#how-do-you-find-a-good-learning-rate