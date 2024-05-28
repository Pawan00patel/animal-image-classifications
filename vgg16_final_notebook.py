#!/usr/bin/env python
# coding: utf-8

# ## __Objective:__ Create a multiclass image classifier
# 
# ## __Purpose:__ Can be used to classify  species of animal
# 
# ### Use transfer learning and vgg16 model

# ### importing necessary libraries

# In[3]:


import pandas as pd
import numpy as np 
import itertools
import keras
import keras
print(keras.__version__)

import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator

from keras.models import Sequential 
from keras import optimizers
from keras.utils import img_to_array, load_img
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import math  
import datetime
import time





# Loading up our image datasets

# In[4]:


#Default dimensions we found online
img_width, img_height = 224, 224  
   
#Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5' 

# loading up our datasets
train_data_dir = 'data/train'  
validation_data_dir = 'data/validation'  
test_data_dir = 'data/test'
   
# number of epochs to train top model  
epochs = 7 #this has been changed after multiple model run  
# batch size used by flow_from_directory and predict_generator  
batch_size = 50  


# In[5]:


#Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet') 


# In[6]:


vgg16 = applications.VGG16(include_top=False, weights=None)
vgg16.load_weights("C:\\Users\\Pawan\Downloads\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")  # Load the downloaded weights


# In[ ]:


# from visualkeras import visualizer

# visualizer(vgg16, format="png", title="VGG16 Architecture",
#            layout="layered", layer_filter=lambda layer: "conv" in layer.name)


# In[ ]:


# from IPython.display import Image
# from keras.applications import VGG16
# from keras.utils import plot_model

# # Load the pre-trained VGG16 model
# model = VGG16(weights="imagenet", include_top=False)

# # Display the model architecture as a PNG image
# plot_model(model, to_file="vgg16_architecture.png", show_shapes=True)

# # Open the image in your default image viewer
# Image(filename="vgg16_architecture.png")


# In[7]:


datagen = ImageDataGenerator(rescale=1. / 255)  #needed to create the bottleneck .npy files


# # Creation of weights/features with VGG16

# In[8]:


#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()
   
generator = datagen.flow_from_directory(  
     train_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_train_samples = len(generator.filenames)  
num_classes = len(generator.class_indices)  
   
predict_size_train = int(math.ceil(nb_train_samples / batch_size))  
   
bottleneck_features_train = vgg16.predict(generator, predict_size_train)  
   
np.save('bottleneck_features_train.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[9]:


print('-'*117)


# In[10]:


#__this can take half an hour to run so only run it once. once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     validation_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_validation_samples = len(generator.filenames)  
   
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))  
   
bottleneck_features_validation = vgg16.predict(  
     generator, predict_size_validation)  
   
np.save('bottleneck_features_validation.npy', bottleneck_features_validation) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[11]:


print('-'*117)


# In[12]:


#__this can take half an hour to run so only run it once. once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     test_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_test_samples = len(generator.filenames)  
   
predict_size_test = int(math.ceil(nb_test_samples / batch_size))  
   
bottleneck_features_test = vgg16.predict(  
     generator, predict_size_test)  
   
np.save('bottleneck_features_test.npy', bottleneck_features_test) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# # Loading training, validation and testing data

# In[13]:


#training data
generator_top = datagen.flow_from_directory(  
         train_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode='categorical',  
         shuffle=False)  
   
nb_train_samples = len(generator_top.filenames)  
num_classes = len(generator_top.class_indices)  
   
# load the bottleneck features saved earlier  
train_data = np.load('bottleneck_features_train.npy')  
   
# get the class lebels for the training data, in the original order  
train_labels = generator_top.classes  
   
# convert the training labels to categorical vectors  
train_labels = to_categorical(train_labels, num_classes=num_classes) 


# In[14]:


#validation data
generator_top = datagen.flow_from_directory(  
         validation_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_validation_samples = len(generator_top.filenames)  
   
validation_data = np.load('bottleneck_features_validation.npy')  
   

validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes)  


# In[15]:


#testing data
generator_top = datagen.flow_from_directory(  
         test_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_test_samples = len(generator_top.filenames)  
   
test_data = np.load('bottleneck_features_test.npy')  
   

test_labels = generator_top.classes  
test_labels = to_categorical(test_labels, num_classes=num_classes)


# # Training of model

# In[37]:


import datetime
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras import optimizers

start = datetime.datetime.now()

model = Sequential()  
model.add(Flatten(input_shape=train_data.shape[1:]))  
model.add(Dense(100, activation='relu'))  # Changed activation to 'relu' for simplicity
model.add(Dropout(0.5))  
model.add(Dense(50, activation='relu'))  # Changed activation to 'relu' for simplicity
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))  

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])  

history = model.fit(train_data, train_labels,  
                    epochs=200,
                    batch_size=batch_size,  
                    validation_data=(validation_data, validation_labels))  

# Evaluate the model on test data
(eval_loss, eval_accuracy) = model.evaluate(test_data, test_labels)

print("[INFO] Test accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("[INFO] Test Loss: {}".format(eval_loss))  
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[19]:


#Model summary
model.summary()


# In[16]:


# import visualkeras
# # visualkeras.layered_view(vgg16)


# In[17]:


# from PIL import ImageFont
# font = ImageFont.truetype("arial.ttf", 12)
# from keras import layers
# from collections import defaultdict
# color_map = defaultdict(dict)#customize the colours
# color_map[layers.Conv2D]['fill'] = '#00f5d4'
# color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
# color_map[layers.Dropout]['fill'] = '#03045e'
# color_map[layers.Dense]['fill'] = '#fb5607'
# color_map[layers.Flatten]['fill'] = '#ffbe0b'
# visualkeras.layered_view(vgg16, legend=True, font=font,color_map=color_map)


# In[ ]:


# import tensorflow as tf
# from keras.applications import VGG16
# from keras.utils import plot_model

# model = VGG16(weights='imagenet')


# In[ ]:


# import visualkeras
# visualkeras.layered_view(model)


# In[ ]:


# plot_model(model, to_file='pretrained_vgg16.png', show_shapes=True)


# In[ ]:


# from PIL import Image
# img = Image.open('pretrained_vgg16.png')
# img.show()


# In[20]:


#Graphing our training and validation
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')  
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')  
plt.xlabel('epoch')
plt.legend()
plt.show()


# ## Model Evaluation on Testing Set

# In[21]:


model.evaluate(test_data, test_labels)


# ## Classification metrics and Confusion Matrix

# ### Classification Metrics

# In[22]:


print('test data', test_data)
preds = np.round(model.predict(test_data),0) 
#to fit them into classification metrics and confusion metrics, some additional modificaitions are required
print('rounded test_labels', preds)


# In[11]:


# import visualkeras
# visualkeras.layered_view(model)


# In[23]:


animals =  ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat', 'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra']
classification_metrics = metrics.classification_report(test_labels, preds, target_names=animals )
print(classification_metrics)


# ### Confusion Matrix

# In[24]:


#Since our data is in dummy format we put the numpy array into a dataframe and call idxmax axis=1 to return the column
# label of the maximum value thus creating a categorical variable
#Basically, flipping a dummy variable back to it's categorical variable
categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)


# In[25]:


confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)


# In[26]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(10, 8)):  # Adjust the figsize as per your preference
    # Add Normalization Option
    '''prints pretty confusion metric with normalization option '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    # Set figure size
    plt.figure(figsize=figsize)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    
    # Rotate x-labels by 90 degrees
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Increase x-coordinate for more horizontal space
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[27]:


plot_confusion_matrix(confusion_matrix,  ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat', 'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra'])


# In[28]:


#Those numbers are all over the place. Now turning normalize= True
plot_confusion_matrix(confusion_matrix, 
                       ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat', 'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra'],
                     normalize=True)


# ## Testing images on model

# In[29]:


def read_image(file_path):
    print("[INFO] loading and preprocessing image...")  
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image


# In[31]:


def test_single_image(path):
    animals = ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat', 'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra']
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)
    preds = model.predict(bt_prediction)
    #print("BT ",bt_prediction)
    for idx, animal, x in zip(range(0, 6), animals, preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100, 2)))
    print('Final Decision:')
    time.sleep(.5)
    for x in range(3):
        print('.'*(x+1))
        time.sleep(.2)
    class_predicted = model.predict(bt_prediction)
    class_dictionary = generator_top.class_indices
    #print("class_dictionary ",class_dictionary)
    inv_map = {v: k for k, v in class_dictionary.items()}
    #print("inv_map ",inv_map)
    class_prob = list(class_predicted[0])
    #print(class_prob)
    max_prob = max(class_prob)
    #print(max_prob)
    pred_class = class_prob.index(max_prob)
    #print(pred_class)
    print("ID: {}, Label: {}".format(class_dictionary[inv_map[pred_class]], inv_map[pred_class]))
    return load_img(path)


# In[34]:


path = 'data/test/bear/1_96.jpg'


# In[35]:


test_single_image(path)


# In[1]:


# import tensorflow as tf
# print(tf.__version__)


# In[12]:


# from keras.utils.vis_utils import plot_model
# from keras.applications import VGG16

# # Load VGG16 model
# model = VGG16()

# # Print model summary
# model.summary()

# # Plot model architecture
# plot_model(model, to_file='vgg16_architecture.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




