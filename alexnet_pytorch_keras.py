#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.utils import to_categorical
from keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import math
import datetime
import time


# In[26]:


import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# Default dimensions we found online
img_width, img_height = 224, 224

# Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model_alexnet.h5'

# Loading up our datasets
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'

# Number of epochs to train top model
epochs = 100  # This has been changed after multiple model runs
# Batch size used by flow_from_directory and predict_generator
batch_size = 50


# In[27]:


# Load pre-trained AlexNet model in PyTorch
pytorch_model = models.alexnet(pretrained=True)

# Modify the last layer for feature extraction
num_ftrs = pytorch_model.classifier[6].in_features
pytorch_model.classifier[6] = nn.Identity()

# Define transforms for images
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(train_data_dir, transform=transform)
validation_dataset = ImageFolder(validation_data_dir, transform=transform)
test_dataset = ImageFolder(test_data_dir, transform=transform)

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Feature extraction for training data
start = datetime.datetime.now()
train_features = []
train_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        features = pytorch_model(images)
        train_features.append(features)
        train_labels.append(labels)

train_features = torch.cat(train_features)
train_labels = torch.cat(train_labels)
np.save('train_features_alex.npy', train_features.numpy())
np.save('train_labels_alex.npy', train_labels.numpy())
end = datetime.datetime.now()
elapsed = end - start
print('Feature extraction time for training data:', elapsed)

# Feature extraction for validation data
start = datetime.datetime.now()
validation_features = []
validation_labels = []

with torch.no_grad():
    for images, labels in validation_loader:
        features = pytorch_model(images)
        validation_features.append(features)
        validation_labels.append(labels)

validation_features = torch.cat(validation_features)
validation_labels = torch.cat(validation_labels)
np.save('validation_features_alex.npy', validation_features.numpy())
np.save('validation_labels_alex.npy', validation_labels.numpy())
end = datetime.datetime.now()
elapsed = end - start
print('Feature extraction time for validation data:', elapsed)

# Feature extraction for test data
start = datetime.datetime.now()
test_features = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        features = pytorch_model(images)
        test_features.append(features)
        test_labels.append(labels)

test_features = torch.cat(test_features)
test_labels = torch.cat(test_labels)
np.save('test_features_alex.npy', test_features.numpy())
np.save('test_labels_alex.npy', test_labels.numpy())
end = datetime.datetime.now()
elapsed = end - start
print('Feature extraction time for test data:', elapsed)

# Load features and labels
train_data = np.load('train_features_alex.npy')
train_labels = np.load('train_labels_alex.npy')
validation_data = np.load('validation_features_alex.npy')
validation_labels = np.load('validation_labels_alex.npy')
test_data = np.load('test_features_alex.npy')
test_labels = np.load('test_labels_alex.npy')

# Convert target labels to one-hot encoded format
train_labels = to_categorical(train_labels, num_classes=20)
validation_labels = to_categorical(validation_labels, num_classes=20)
test_labels = to_categorical(test_labels, num_classes=20)


# In[28]:


# Define the model architecture
from keras import regularizers
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(512, activation=keras.layers.LeakyReLU(
    alpha=0.3), kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dropout(0.5))
model.add(Dense(128, activation=keras.layers.LeakyReLU(
    alpha=0.3), kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dropout(0.4))
model.add(Dense(units=20, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc'])

early_stopping = EarlyStopping(
    monitor='val_acc', patience=20, verbose=1, mode='max')

# Train the model
history = model.fit(train_data, train_labels,
                    epochs=200,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels),
                    callbacks=[early_stopping])

# Save the trained model
model.save_weights(top_model_weights_path)

# Evaluate the model on the test data
(eval_loss, eval_accuracy) = model.evaluate(
    test_data, test_labels, batch_size=batch_size, verbose=1)

# Print the evaluation results
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


# In[50]:


filepath = 'model_save_alexnet.h5'
keras.models.save_model(model, filepath)


# In[51]:


model.save('savealexnet.h5')


# In[29]:


# Model summary
model.summary()


# In[30]:


# Graphing our training and validation
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


# In[31]:


model.evaluate(test_data, test_labels)


# In[32]:


print('test data', test_data)
preds = np.round(model.predict(test_data), 0)
# to fit them into classification metrics and confusion metrics, some additional modificaitions are required
print('rounded test_labels', preds)


# In[33]:


animals = ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat',
           'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra']
classification_metrics = metrics.classification_report(
    test_labels, preds, target_names=animals)
print(classification_metrics)


# In[34]:


# Since our data is in dummy format we put the numpy array into a dataframe and call idxmax axis=1 to return the column
# label of the maximum value thus creating a categorical variable
# Basically, flipping a dummy variable back to it's categorical variable
categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)


# In[35]:


confusion_matrix = confusion_matrix(categorical_test_labels, categorical_preds)


# In[36]:


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
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[37]:


plot_confusion_matrix(confusion_matrix, ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe',
                      'goat', 'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra'])


# In[38]:


# Those numbers are all over the place. Now turning normalize= True
plot_confusion_matrix(confusion_matrix,
                      ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat', 'gorilla',
                          'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra'],
                      normalize=True)


# In[39]:


model.load_weights(top_model_weights_path)


# In[40]:


def read_image(file_path):
    print("[INFO] loading and preprocessing image...")
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image


# In[41]:


def test_single_image(path):
    animals = ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat',
               'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra']
    image = read_image(path)
    time.sleep(.5)
    image = torch.tensor(image, dtype=torch.float32)
    # Change the dimension order to [batch_size, channels, height, width]
    image = image.permute(0, 3, 1, 2)
    image = image.to('cpu')  # Set to 'cuda' if you're using GPU
    features = pytorch_model(image)
    features = features.detach().numpy()
    preds = model.predict(features)
    # print("BT ",bt_prediction)
    for idx, animal, x in zip(range(0, 20), animals, preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100, 2)))
    print('Final Decision:')
    time.sleep(.5)
    for x in range(3):
        print('.'*(x+1))
        time.sleep(.2)
    class_prob = list(preds[0])
    max_prob = max(class_prob)
    pred_class = class_prob.index(max_prob)
    print("ID: {}, Label: {}".format(pred_class, animals[pred_class]))
    return load_img(path)


# In[47]:


path = 'data/test/gorilla/4_74.jpg'


# In[48]:


test_single_image(path)


# In[49]:


from sklearn import metrics

# Compute precision, recall, F1 score
precision = metrics.precision_score(
    np.argmax(test_labels, axis=1), np.argmax(preds, axis=1), average='weighted')
recall = metrics.recall_score(
    np.argmax(test_labels, axis=1), np.argmax(preds, axis=1), average='weighted')
f1_score = metrics.f1_score(np.argmax(test_labels, axis=1), np.argmax(
    preds, axis=1), average='weighted')

# Compute confusion matrix
conf_matrix = metrics.confusion_matrix(
    np.argmax(test_labels, axis=1), np.argmax(preds, axis=1))

# Calculate FAR and FRR
far = np.sum(conf_matrix.sum(axis=0) - np.diag(conf_matrix)) / \
    np.sum(conf_matrix.sum(axis=1))
frr = np.sum(conf_matrix.sum(axis=1) - np.diag(conf_matrix)) / \
    np.sum(conf_matrix.sum(axis=1))

# Print out the results
print("Accuracy: {:.2f}%".format(eval_accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1_score))
print("False Acceptance Rate (FAR): {:.2f}".format(far))
print("False Rejection Rate (FRR): {:.2f}".format(frr))

# Calculate mAP
average_precision = metrics.average_precision_score(
    test_labels, preds, average=None)
mAP = np.mean(average_precision)
print("Mean Average Precision (mAP): {:.2f}".format(mAP))


# In[ ]:




