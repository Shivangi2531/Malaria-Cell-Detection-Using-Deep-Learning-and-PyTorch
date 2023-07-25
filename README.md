# Malaria-Cell-Detection-Using-Deep-Learning-and-PyTorch


The parasite Plasmodium causes the potentially fatal infectious disease malaria, which is spread to people via the bite of infected female Anopheles mosquitoes. Tropical and subtropical areas are affected by the disease, which poses a serious threat to world health, especially in sub-Saharan Africa, Southeast Asia, and parts of South America.

There are recurrent cycles of fever, chills, and flu-like symptoms brought on by the malaria parasite, which multiplies in the liver and infects red blood cells. Malaria primarily affects young children, pregnant women, and people with compromised immune systems. In extreme cases, it can result in anaemia, organ failure, and even death.

Insecticide-treated bed nets, indoor residual spraying, and antimalarial medications for both prevention and therapy are all used in the fight against malaria. Malaria control tactics have become more challenging as drug-resistant strains and insecticide-resistant insects have emerged.



In the fight against malaria, research, diagnosis, and early detection are essential elements. Modern developments in medical imaging and deep learning offer new options for more effective and accurate malaria cell detection, diagnosis, and monitoring. Rapid diagnostic tests and microscopic analysis of blood samples remain crucial tools for detecting the disease.


Our deep learning project aims to develop an efficient and accurate system for malaria cell detection using the ResNet architecture, incorporating regularization techniques, and enhancing the dataset with data augmentation in the PyTorch framework.

We are going to do it in the following steps:
1. Pick a Dataset
2. Download the dataset
3. Import the dataset using PyTorch
4. Prepare the dataset for training
5. Move the dataset to the GPU
6. Define a neural networks
7. Train the model

Make predictions on sample images iterate on it with different networks & hyperparameters


## Summary and Further Reading

Here's a summary of the different techniques used in this notebook to improve our model performance and reduce the training time:


* **Data normalization**: We normalized the image tensors by subtracting the mean and dividing by the standard deviation of pixels across each channel. Normalizing the data prevents the pixel values from any one channel from disproportionately affecting the losses and gradients. [Learn more](https://medium.com/@ml_kid/what-is-transform-and-transform-normalize-lesson-4-neural-networks-in-pytorch-ca97842336bd)

* **Data augmentation**: We applied random transformations while loading images from the training dataset. Specifically, we will pad each image by 4 pixels, and then take a random crop of size 32 x 32 pixels, and then flip the image horizontally with a 50% probability. [Learn more](https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/)

* **Residual connections**: One of the key changes to our CNN model was the addition of the resudial block, which adds the original input back to the output feature map obtained by passing the input through one or more convolutional layers. We used the ResNet9 architecture [Learn more](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec).

* **Batch normalization**: After each convolutional layer, we added a batch normalization layer, which normalizes the outputs of the previous layer. This is somewhat similar to data normalization, except it's applied to the outputs of a layer, and the mean and standard deviation are learned parameters. [Learn more](https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd)

* **Learning rate scheduling**: Instead of using a fixed learning rate, we will use a learning rate scheduler, which will change the learning rate after every batch of training. There are [many strategies](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) for varying the learning rate during training, and we used the "One Cycle Learning Rate Policy". [Learn more](https://sgugger.github.io/the-1cycle-policy.html)

* **Weight Decay**: We added weight decay to the optimizer, yet another regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function. [Learn more](https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab)

* **Gradient clipping**: We also added gradient clippint, which helps limit the values of gradients to a small range to prevent undesirable changes in model parameters due to large gradient values during training.  [Learn more.](https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48#63e0)

* **Adam optimizer**: Instead of SGD (stochastic gradient descent), we used the Adam optimizer which uses techniques like momentum and adaptive learning rates for faster training. There are many other optimizers to choose froma and experiment with. [Learn more.](https://ruder.io/optimizing-gradient-descent/index.html)


To improve our accuracy we can try different types of CNN architectures like Resnets-50, Xception and ResNeXt-50 etc.

Our model trained to 96.34% accuracy
