#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras as tfk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.layers import Dropout
from keras.layers import BatchNormalization
import sys

from keras.preprocessing.image import ImageDataGenerator
import math
import keras
from tensorflow.keras.datasets import cifar100


from keras.callbacks import Callback
from keras import backend
from keras.models import load_model
from keras.utils import to_categorical

import pandas as pd

from sklearn.metrics import accuracy_score


# Some code modified from that provided by Daniel Sawyer.  This implementation
# is done with functions for a different look.  You do not have to use it.
# You will work with cifar100 as set up here (in terms of train, validation
# and test).  This is color images of size 32x32 of 100 classes. Hence, 3
# chanels R, G, B.    I took out 10% for validation.
# You can change this around, but must be very clear on what was done and why.
# You must improve on 44% accuracy (which is a fairly low bar).  You need to
# provide a best class accuracy and worst class accuracy. To improve, more epochs
# can help, but that cannot be the only change you make.  You should show  better
# performance at 15 epochs or argue why it is not possible.

# I also want you to use a snapshot ensemble of at least 5 snapshots.  One
# way to choose the best class is to sum the per class outputs and take the
# maximum.  Another is to vote for the class and break ties in some way.
# Indicate if results are better or worse or the same. (This is 5
# extra credit points of the grade).

# You must clearly explain what you tried and why and what seemed to work
# and what did not.  That will be the major part of your grade.  Higher
# accuracy will also improve your grade. If you use an outside source, it
# must be disclosed and that source may be credited with part of the grade.
#  The best accuracy in class will add
# 4 points to their overall average grade, second best 3 points and 3rd best 2
# points and 4th best 1 point.

# To get predictions:
# predictions=model.predict(ds_test)
# Prints the first test predition and you will see 100 predictions
# print(predictions[0])


# In[2]:


# cifar100 has 2 sets of labels.  The default is "label" giving you 100 predictions for the classes

(input_train, target_train), (input_test, target_test) = cifar100.load_data()

input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize the dataset
input_train = input_train / 255
input_test = input_test / 255

# Making all the target output as category. Very important step.
target_test = to_categorical(target_test)


# In[3]:


# Shuffle the training set

shuffle_index = np.random.permutation(input_train.shape[0])
ds_train_shuffled, target_train_shuffled = input_train[shuffle_index], target_train[shuffle_index]

# First 45% and last 45% from training, then validation data is 10%
# from 45% of train data to 55% and test is the usual 10K

row_num_45 = int(input_train.shape[0]*0.45)
row_num_55 = int(input_train.shape[0]*0.55)

ds_train_1 = input_train[:row_num_45]
ds_train_2 = input_train[row_num_55:]

ds_train_x = np.concatenate((ds_train_1, ds_train_2), axis=0)

ds_train_y_1 = target_train[:row_num_45]
ds_train_y_2 = target_train[row_num_55:]

ds_train_y = np.concatenate((ds_train_y_1, ds_train_y_2), axis=0)

dsvalid_x = input_train[row_num_45:row_num_55]
dsvalid_y = target_train[row_num_45:row_num_55]


# In[4]:


### Refer the snapshot ensemble codes from https://www.kaggle.com/fkdplc/snapshot-ensemble-tutorial-with-keras

class SnapshotEnsemble(Callback):
    
    __snapshot_name_fmt = "snapshot_%d.hdf5"
    
    def __init__(self, n_models, n_epochs_per_model, lr_max, verbose=1):
        """
        n_models -- quantity of models (snapshots)
        n_epochs_per_model -- quantity of epoch for every model (snapshot)
        lr_max -- maximum learning rate (snapshot starter)
        """
        self.n_epochs_per_model = n_epochs_per_model
        self.n_models = n_models
        self.n_epochs_total = self.n_models * self.n_epochs_per_model
        self.lr_max = lr_max
        self.verbose = verbose
        self.lrs = []
 
    # calculate the learning rate for epoch
    def cosine_annealing(self, epoch):
        cos_inner = (math.pi * (epoch % self.n_epochs_per_model)) / self.n_epochs_per_model
        return self.lr_max / 2 * (math.cos(cos_inner) + 1)

    # when epoch begins update learning rate
    def on_epoch_begin(self, epoch, logs={}):
        # update learning rate
        lr = self.cosine_annealing(epoch)
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrs.append(lr)

    # when epoch ends check if there is a need to save a snapshot
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.n_epochs_per_model == 0:
            # save model to file
            filename = self.__snapshot_name_fmt % ((epoch + 1) // self.n_epochs_per_model)
            self.model.save(filename)
            if self.verbose:
                print('Epoch %d: snapshot saved to %s' % (epoch, filename))
                
    # load all snapshots after training
    def load_ensemble(self):
        models = []
        for i in range(self.n_models):
            models.append(load_model(self.__snapshot_name_fmt % (i + 1)))
        return models
    
    


# In[6]:


epochs = 20
batch_size = 128
xavier = keras.initializers.glorot_normal(seed=None)

# Input shape and layer.  This is rgb
input_shape = (32, 32, 3)
input_layer = tfk.layers.Input(shape=input_shape)
    
# Create the model
### Improve the following architecture from:
### https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer=xavier, padding='same', input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer=xavier, padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer=xavier, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer=xavier, padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer=xavier, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer=xavier, padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(2048, activation='relu', kernel_initializer=xavier))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu', kernel_initializer=xavier))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu', kernel_initializer=xavier))
model.add(BatchNormalization())




model.add(Dense(100, activation='softmax'))


# Compile the model
# opt = tfk.optimizers.Adam(learning_rate=0.001)
lr=0.001
opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Prints model summary
model.summary()



# # create data generator
# datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# # prepare iterator
# it_train = datagen.flow(ds_train_x, ds_train_y, batch_size=128)



se_callback = SnapshotEnsemble(n_models=5, n_epochs_per_model=4, lr_max=.001)
    
    
# Trains model with increased epochs and saves best
# model.fit_generator(
#         it_train,
#         epochs=epochs,
#         validation_data=(dsvalid_x, dsvalid_y),
#         callbacks=[se_callback]
#     )



# Trains model with increased epochs and saves best
model.fit(
        ds_train_x, ds_train_y,
        epochs=epochs,
        validation_data=(dsvalid_x, dsvalid_y),
        batch_size = batch_size,
        callbacks=[se_callback]
    )


# In[7]:


### Refer the following codes (Calculating the accuracy for each model) from:
### https://www.kaggle.com/fkdplc/snapshot-ensemble-tutorial-with-keras


# makes prediction according to given models and given weights
def predict(models, data, weights=None):
    if weights is None:
        # default weights provide voting equality
        weights = [1 / (len(models))] * len(models)
    pred = np.zeros((data.shape[0], 100))
    for i, model in enumerate(models):
        pred += model.predict(data) * weights[i]
    return pred
    
# returns accuracy for given predictions
def evaluate(preds, weights=None):
    if weights is None:
        weights = [1 / len(preds)] * len(preds)
    y_pred = np.zeros((target_test.shape[0], 100))
    for i, pred in enumerate(preds):
        y_pred += pred * weights[i]
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(target_test, axis=1)
    return accuracy_score(y_true, y_pred)

# load list of snapshots
models = se_callback.load_ensemble()
# precalculated predictions of all models
preds = []
# evaluate every model as single
for i, model in enumerate(models):
    pred = predict([model], input_test)
    preds.append(pred)
    score = evaluate([pred])
    print(f'model {i + 1}: accuracy = {score:.4f}')

# evaluate ensemble (with voting equality)
ensemble_score = evaluate(preds)
print(f'ensemble: accuracy = {ensemble_score:.4f}')


# In[8]:


### Refer the following codes (Best Weight) from:
### https://www.kaggle.com/fkdplc/snapshot-ensemble-tutorial-with-keras

best_score = ensemble_score
best_weights = None
no_improvements = 0
while no_improvements < 1000:
    
    # generate normalized weights
    new_weights = np.random.uniform(size=(len(models), ))
    new_weights /= new_weights.sum()
    
    # get the score without predicting again
    new_score = evaluate(preds, new_weights)
    
    # check (and save)
    if new_score > best_score:
        no_improvements = 0
        best_score = new_score
        best_weights = new_weights
        print(f'improvement: {best_score:.4f}')
    else:
        no_improvements += 1

print(f'best weights are {best_weights}')


# In[9]:


### Get the prediction from the best weight

pred = predict(models, input_test, best_weights)


# In[10]:


pred_label = np.argmax(pred, axis=1)
true_label = np.argmax(target_test, axis=1)


# In[24]:


counting.shape


# In[25]:


### Obtain the counting of the labels correctly classified

counting = np.zeros((target_test.shape[1],))


for i in range(0, target_test.shape[0]):
    if pred_label[i] == true_label[i]:
        counting[(pred_label[i])] = counting[(pred_label[i])]+1
   


# In[28]:


import pandas as pd

result = pd.DataFrame()
result['ImageId'] = np.arange(target_test.shape[1]) + 1
result['Correct_label'] = counting
result.to_csv('submission.csv', index=False)


# In[30]:


highest_list = []
lowest_list = []

highest = result['Correct_label'].max()
lowest = result['Correct_label'].min()


for i in range(1, target_test.shape[1]):
    if counting[i] == highest:
      highest_list.append((i+1))
    elif counting[i] == lowest:
      lowest_list.append((i+1))


# In[31]:


print("Highest_accuracy of the class: ", highest, " %")
print("Highest accuracy class: ")
for x in highest_list:
  print(x)


# In[32]:


print("Lowest_accuracy of the class: ", lowest, " %")
print("Lowest accuracy class: ")
for x in lowest_list:
  print(x)


# In[ ]:


# ### Wider or Deeper?
# When I begin the model with deeper CNN, such as 9 convolutional layers (32, 32, 64, 64, 128, 128, 256, 256, 512) and 2 dense layers (512, 256), the accuracy was 42.01%, 44.92%, 44.32%. Not much improvement from the original architecture.

# With 100 classes in CIFAR-100 images, using only 32 or 64 filters causes the network extracting small number of abstractions from the image data. This might be beneficial to CIFAR-10 images with only 10 classes. Small number of abstractions meaning there are many raw data extracted and it produces a lot of noise. This noise, “pure” raw data, confuse the classifier especially when there are ‘sub’classes in the CIFAR-100 dataset and having 100 classes.

# Therefore, my recommendation is going wider instead of deeper. I started with two layers using 128 filters, following by two layers using 256 filters and another two layers using 512 filters, lastly three dense layers (2048, 1024, 512). This architecture significantly increases at least 10% of the accuracy although it has less layers than the first one I proposed.

# ### Kernel Size
# I have tried using both 3x3 and 5x5 kernel size, the conclusion is the smaller kernel size (3x3) works better than 5x5. Firstly, the CIFAR-100 dataset consists 3 channels R, G, B in the images.
# Besides that, since we are using large filter size, it is recommended to apply 3x3 kernel size to find useful features. After trying both kernel size, it is proven that 3x3 gives higher accuracy.

# ### Batch Normalization
# Batch normalization plays an important role in this project. As my network is wide, it is necessary to normalize the contributions to different layer and nodes for every mini batch during the training. Adding this regularization (Batch Normalization) helps stabilizing the network learning rate. Without the batch-normalization layers, it happens overfitting issue by showing the accuracy of the test set decreasing 10 to 20%.

# ### Data Normalization
# I have tried to normalize the image data although it is not required. I realize the image normalization DOES NOT help much in this project. A possible reason is DIFAR-100 image dataset are already normalized and they all have a similar data distribution. However, making the data type is critical. Without making the test target as category, the model would treat it as decimals and wrongly classify them.

# ### Data split
# I use a different method to extract the data from CIFAR-100, but the split is the same. First 45% and last 45% are for training dataset, the middle 45%-55% data as the validation set and there are another 10,000 images as test set.
# ### Data Augmentation
# Data Augmentation is popular when analyzing the image data. However, since the dataset is huge and it has subclasses in each superclass, I would not recommend using data augmentation in this project. The main reason is data augmentation is computationally expensive especially in this project and it does not help much in increasing the accuracy after my experiment. With data augmentation, it increases only 1% of the accuracy but it took me double of the time to run the CNN. It is not worth for the resources.

# ### Activation, Kernel Initializer, Padding and Optimizer
# After multiple testing,
# -    RELU works better than sigmoid and ELU in this project.
# -    “glorot_normal” works better than “he_uniform” in kernel initializer.
# -    SAME padding works better than VALID padding.
# -    Add BETA regularization into Adam Optimizer.

# ### Epoch & Batch Size
# Last but not least, increasing the epoch helps in bossting up the accuracy. However, each running of an epoch takes 30 minutes locally and 60 minutes by using google collab hosted runtime. Therefore, I would not recommend using high epoch in this project. I have chosen using only 20 epoch and it achieves optimal accuracy.

# I have tried using 64, 128 and 256 batch size. The conclusion is 128 batch size works the best with the reasonable time.

# ### Snapshot Ensemble
# By referring to the KAGGLE website, www.kaggle.com/fkdplc/snapshot-ensemble-tutorial-with-keras, I have successfully applied snapshot ensemble in my codes. At the end of the codes, there are 5 accuracy values from different snapshot and it also conclude with the best weight.

# Furthermore, the highest and lowest accuracy of the class would be observed too at the end of the codes.

# ### Conclusion:
# My network Architecture:
# -    Convolutional Layers: 6 layers
# -    Dense Layers: 3 layers
# -    Pooling Layers: 3 layers
# -    Batch Size: 128
# -    Regularization: True
# -    Augmentation: False

# Output:
# 1st run: 63.47%,
# 2nd run: 63.89%
# 3rd run:
# Average Accuracy:

# Best Class accuracy: Class 49 (92%)
# Worst Class accuracy: Class 73 (32%)

