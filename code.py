#!/usr/bin/env python
# coding: utf-8

# ### Fashion MNIST

# In[4]:


import struct
import numpy as np

def read_idx(filename):
    """Credit: https://gist.github.com/tylerneylon"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# ### We use the function to extact our training and test datasets

# In[5]:


x_train = read_idx("./fashion_mnist/train-images-idx3-ubyte")
y_train = read_idx("./fashion_mnist/train-labels-idx1-ubyte")
x_test = read_idx("./fashion_mnist/t10k-images-idx3-ubyte")
y_test = read_idx("./fashion_mnist/t10k-labels-idx1-ubyte")


# ### Let's inspect our dataset

# In[6]:


# printing the number of samples in x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))

print ("Number of samples in our training data: " + str(len(x_train)))
print ("Number of labels in our training data: " + str(len(y_train)))
print ("Number of samples in our test data: " + str(len(x_test)))
print ("Number of labels in our test data: " + str(len(y_test)))
print()
print ("Dimensions of x_train:" + str(x_train[0].shape))
print ("Labels in x_train:" + str(y_train.shape))
print()
print ("Dimensions of x_test:" + str(x_test[0].shape))
print ("Labels in y_test:" + str(y_test.shape))


# In[7]:


from keras.datasets import mnist
from keras.utils import np_utils
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

# Training Parameters
batch_size = 128
epochs = 3

# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Let's count the number columns in our hot encoded matrix 
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# create MODEL ARCHITECTURE

import numpy as np
import hyper_parameters1 as hp
model = Sequential()

def addCRPs(i):
    if i==1: #1conv and 1 maxpool
        model.add(Conv2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))
        model.add(Dropout(0.25))
        
    elif i==2:  # 2Conv and 1 MaxPool
        model.add(Conv2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=hp.no_of_filters*2,
        kernel_size=(hp.kernel_size+2,hp.kernel_size+2),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))
        model.add(Dropout(0.25))
        
    elif i==3:  # 2Conv and 2Pool
        model.add(Conv2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=hp.no_of_filters*2,
        kernel_size=(hp.kernel_size+2,hp.kernel_size+2),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(hp.pool_size+2,hp.pool_size+2)))
        model.add(Dropout(0.25))
        
    else: # 3Conv and 1 MaxPool
        model.add(Conv2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=hp.no_of_filters*2,
        kernel_size=(hp.kernel_size+2,hp.kernel_size+2),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=hp.no_of_filters*4,
        kernel_size=(hp.kernel_size+4,hp.kernel_size+4),activation='relu',
        input_shape=input_shape))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))
        model.add(Dropout(0.25))

addCRPs(hp.i)

    
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

print(model.summary())


# In[8]:


print(hp.i)


# ### Let's train our model

# In[ ]:


history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


accuracy=float(score[1]*100)
print(f"accuracy {accuracy}%")


# In[ ]:


#Printing accuracy into a separate file

from os import system

if accuracy > 90:
    system("echo True accuracy={}% > Accuracy.txt".format(accuracy))
    #system("echo 'True' > /Accuracy.txt")  # for linux
else:
    system("echo False accuracy={}% > Accuracy.txt".format(accuracy))
    #system("echo 'False' > /Accuracy.txt")


# In[ ]:


"""f1=open("Accuracy.txt" , "r")
x1=f1.readline()
ac=float(x1)

f2=open()"input.txt","r")
a=[]
file=f2.readlines()
for l in f2:
    
"""


# In[ ]:





# In[ ]:


# Code to send email

import smtplib, ssl

port = 465 #For SSL
smtp_server ="smtp.gmail.com"
sender_email="iampalakjain01@gmail.com"    #Sender's Mail Address
receiver_email="itspalak19@gmail.com"      #Receiver's Mail Address
password="xecbeupbulzfwpos"
if accuracy > 90:
    message="""    Subject: Report | Prediction Program
    
    CONGRATULATIONS! 
    Your code achieved{}% accuracy.""".format(accuracy)
else:
    message="""    Subject: Report | Prediction Program
    
    Train Again!
    Your code got {}% accuracy.""".format(accuracy)
    
context=ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email,password)
    server.sendmail(sender_email, receiver_email, message)    


# ### Let's test out our model

# In[ ]:


import cv2
import numpy as np

def getLabel(input_class):
    number = int(input_class)
    if number == 0:
        return "T-shirt/top "
    if number == 1:
        return "Trouser"
    if number == 2:
        return "Pullover"
    if number == 3:
        return "Dress"
    if number == 4:
        return "Coat"
    if number == 5:
        return "Sandal"
    if number == 6:
        return "Shirt"
    if number == 7:
        return "Sneaker"
    if number == 8:
        return "Bag"
    if number == 9:
        return "Ankle boot"

def draw_test(name, pred, actual, input_im):
    BLACK = [0,0,0]

    res = getLabel(pred)
    actual = getLabel(actual)   
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, 4*imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, "Predicted - " + str(res), (152, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0), 1)
    cv2.putText(expanded_image, "   Actual - " + str(actual), (152, 90) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 1)
    cv2.imshow(name, expanded_image)


for i in range(0,10):
    rand = np.random.randint(0,len(x_test))
    input_im = x_test[rand]
    actual = y_test[rand].argmax(axis=0)
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    input_im = input_im.reshape(1,28,28,1) 
    
    ## Get Prediction
    res = str(model.predict_classes(input_im, 1, verbose = 0)[0])

    draw_test("Prediction", res, actual, imageL) 
    cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




