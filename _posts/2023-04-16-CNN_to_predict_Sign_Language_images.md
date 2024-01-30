<a href="https://colab.research.google.com/github/gain16/portfolioAI/blob/master/CNN_to_predict_Sign_Language_images.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# CNN to predict Sign Language images

### Your own image classification system
- Create an image classification system using a variety of sign Language images, rather than relying solely on the pre-existing sample data provided by Tensorflow.
- Compare the two cases as you change the activation function.(Relu and Softmax)

## Setup

Import TensorFlow and other necessary libraries:


```python
# TensorFlow and tf.keras
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Helper libraries
import numpy as np # matrix 연산
import matplotlib.pyplot as plt # 그림 그리기
```

## Data downLoad and prepare (Google Drive Mount)


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    


```python
import shutil

shutil.copy('/content/gdrive/MyDrive/Colab Notebooks/CV_sign_language_filterted/sign_language_filtered.zip', '/content/')
```




    '/content/sign_language_filtered.zip'




```python
# It will delete existing generated files on repeated runs.
# 반복적인 실행시 기존의 생성된 파일을 삭제해 줍니다.
root_dir = '/content'

import os

if os.path.exists(os.path.join(root_dir, 'sign_language_filtered')):
    shutil.rmtree(os.path.join(root_dir, 'sign_language_filtered'))
```


```python
# Import the compressed file module.
# 압축파일 모듈 import해 줍니다.
import zipfile

# Use the Python with function to make the sign_language_filtered.zip file accessible as target_file.
# 파이썬 with함수를 사용해서 sign_language_filtered.zip 파일을 target_file으로 접근하게 만듭니다.
with zipfile.ZipFile(os.path.join(root_dir, 'sign_language_filtered.zip'), mode='r') as target_file: # mode='r' (읽기 모드)

  target_file.extractall(os.path.join(root_dir, 'sign_language_filtered')) # Unzip the zip file to that path.(해당 경로에 zip파일을 해제)
```


```python
# Convert HEIC files to jpegs
# HEIC파일을 jpeg파일로 변환

!pip install pyheif
!pip install Pillow

import os
import pyheif
from PIL import Image

root_dir = '/content'
dataset_class = ['train','test']

for dataset_name in dataset_class:
  folder_path = os.path.join(root_dir, 'sign_language_filtered', dataset_name)
  for label in range(1, 11):
    class_folder = os.path.join(folder_path, str(label))
    for filename in os.listdir(class_folder):
        if filename.lower().endswith(".heic"):
           input_path = os.path.join(class_folder, filename)
           output_path = os.path.join(class_folder, f"{os.path.splitext(filename)[0]}.jpeg")

           heif_file = pyheif.read(input_path)
           image = Image.frombytes(
              heif_file.mode,
              heif_file.size,
              heif_file.data,
              "raw"
           )
           image.save(output_path, "jpeg")
           os.remove(input_path) # Remove existing HEIC files (기존 HEIC파일 제거)
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pyheif in /usr/local/lib/python3.10/dist-packages (0.7.1)
    Requirement already satisfied: cffi>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from pyheif) (1.15.1)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.0.0->pyheif) (2.21)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (8.4.0)
    

Check the data quantity after downloading

* There are 777 images in the train folder
* There are 330 images in the test folder

There are 1107 images in total.


```python
train_dir = os.path.join(root_dir, 'sign_language_filtered/train/')
test_dir = os.path.join(root_dir, 'sign_language_filtered/test/')
```


```python
import pathlib
from pathlib import Path

train_dir = pathlib.Path(train_dir)
test_dir = pathlib.Path(test_dir)

# Check the number of images in the training and test datasets.
# 훈련용데이터셋 과 테스트용데이터셋의 이미지수량을 확인한다.
train_image_count = len(list(train_dir.glob('*/*.*'))) #glob는 문자열 객체에서 호출할수 없다. 그래서 pathlib.path를 사용
test_image_count = len(list(test_dir.glob('*/*.*')))   #왜 */*으로 가운데 '/'가 들어가는지 확인하기

print(train_image_count)
print(test_image_count)
total = int(train_image_count) + int(test_image_count)
print(total)
```

    777
    330
    1107
    

Functions for turning data into numpy array format


```python
def load_images_from_folder(folder_path, target_size):
    images = []
    labels = []
    for label in range(1, 11):
        class_folder = os.path.join(folder_path, str(label))
        for filename in os.listdir(class_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                image_path = os.path.join(class_folder, filename)

                # Load the image and resize it to the desired size
                # 이미지를 불러온 후, 원하는 크기로 리사이즈한다
                image = Image.open(image_path).resize(target_size)

                # Convert the resized image to a Numpy array
                # 리사이즈된 이미지를 NumPy 배열로 변환한다
                numpy_array = np.array(image)
                images.append(numpy_array)
                labels.append(label-1)  # 레이블을 0부터 시작하게 조정
    return np.array(images), np.array(labels)
```


```python
target_size = (28, 28)  # Specify the desired image size (원하는 이미지 크기를 지정한다)

train_dir = os.path.join(root_dir, 'sign_language_filtered', 'train')
train_images, train_labels = load_images_from_folder(train_dir, target_size)

test_dir = os.path.join(root_dir, 'sign_language_filtered', 'test')
test_images, test_labels = load_images_from_folder(test_dir, target_size)
```


```python
# Check Train Dataset Numpy array Format
print(train_images.shape)
print(train_labels.shape)
print(type(train_images))
print(type(train_labels))
```

    (777, 28, 28, 3)
    (777,)
    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>
    


```python
# Check Test Dataset Numpy array Format
print(test_images.shape)
print(test_labels.shape)
print(type(test_images))
print(type(test_labels))
```

    (330, 28, 28, 3)
    (330,)
    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>
    

# Image Classification with CNN

### Create a dataset

Define some parameters for the loader:


```python
batch_size = 32
img_height = 160
img_width = 160
```

It's good practice to use a validation split when developing your model. Use 80% of the images for training and 20% for validation.


```python
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2, # 80% train / 20% validation
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```

    Found 777 files belonging to 10 classes.
    Using 622 files for training.
    


```python
val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```

    Found 777 files belonging to 10 classes.
    Using 155 files for validation.
    

You can find the class names in the `class_names` attribute on these datasets. These correspond to the directory names in alphabetical order.


```python
class_names = train_ds.class_names
print(class_names)
```

    ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
    


```python
train_np = list(train_ds)
val_np = list(val_ds)
```


```python
len(train_np[0])
```




    2




```python
train_np[0][0].shape
```




    TensorShape([32, 160, 160, 3])




```python
train_np[0][1]
```




    <tf.Tensor: shape=(32,), dtype=int32, numpy=
    array([8, 1, 1, 4, 2, 5, 0, 3, 3, 1, 5, 4, 7, 6, 5, 3, 2, 9, 2, 0, 1, 4,
           2, 2, 8, 3, 2, 4, 1, 7, 1, 6], dtype=int32)>



## Visualize the data

Here are the first ten images from the training dataset:


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(10):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
```


    
![png](output_32_0.png)
    


## Configure the dataset for performance


```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

## Standardize the data

The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network; in general you should seek to make your input values small.

Here, you will standardize values to be in the `[0, 1]` range by using `tf.keras.layers.Rescaling`:


```python
normalization_layer = layers.Rescaling(1./255)
```

There are two ways to use this layer. You can apply it to the dataset by calling `Dataset.map`:


```python
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
```

    0.0 0.9798034
    

# Case1) Activation function of Dense layer → relu

## Data augmentation

Overfitting generally occurs when there are a small number of training examples. [Data augmentation](./data_augmentation.ipynb) takes the approach of generating additional training data from your existing examples by augmenting them using random transformations that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better.


```python
# Data Augmenation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
```

Visualize a few augmented examples by applying data augmentation to the same image several times:


```python
# Data Augmenation Output the results
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(10):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
```


    
![png](output_45_0.png)
    


## Create the convolutional base

Building a simple CNN model with Conv2D, Maxpool, Conv2D

Introduced [dropout](https://developers.google.com/machine-learning/glossary#dropout_regularization) regularization to the network to reduce overfitting.


```python
num_classes = len(class_names)

model = Sequential([
  data_augmentation, # 데이터 어그멘테이션 추가
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # dropout 사용
  layers.Flatten(),
  layers.Dense(128, activation='relu'), # Activation function을 relu로 사용
  layers.Dense(num_classes)
])
```

## Compile and train the model


```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     sequential_3 (Sequential)   (None, 160, 160, 3)       0         
                                                                     
     rescaling_4 (Rescaling)     (None, 160, 160, 3)       0         
                                                                     
     conv2d_6 (Conv2D)           (None, 160, 160, 16)      448       
                                                                     
     max_pooling2d_6 (MaxPooling  (None, 80, 80, 16)       0         
     2D)                                                             
                                                                     
     conv2d_7 (Conv2D)           (None, 80, 80, 32)        4640      
                                                                     
     max_pooling2d_7 (MaxPooling  (None, 40, 40, 32)       0         
     2D)                                                             
                                                                     
     conv2d_8 (Conv2D)           (None, 40, 40, 64)        18496     
                                                                     
     max_pooling2d_8 (MaxPooling  (None, 20, 20, 64)       0         
     2D)                                                             
                                                                     
     dropout_2 (Dropout)         (None, 20, 20, 64)        0         
                                                                     
     flatten_2 (Flatten)         (None, 25600)             0         
                                                                     
     dense_4 (Dense)             (None, 128)               3276928   
                                                                     
     dense_5 (Dense)             (None, 10)                1290      
                                                                     
    =================================================================
    Total params: 3,301,802
    Trainable params: 3,301,802
    Non-trainable params: 0
    _________________________________________________________________
    


```python
epochs=100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

    Epoch 1/100
    20/20 [==============================] - 11s 471ms/step - loss: 2.3241 - accuracy: 0.1543 - val_loss: 2.2717 - val_accuracy: 0.1806
    Epoch 2/100
    20/20 [==============================] - 1s 32ms/step - loss: 2.2633 - accuracy: 0.1785 - val_loss: 2.3010 - val_accuracy: 0.1806
    Epoch 3/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2259 - accuracy: 0.1897 - val_loss: 2.2249 - val_accuracy: 0.1806
    Epoch 4/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.1790 - accuracy: 0.1977 - val_loss: 2.2834 - val_accuracy: 0.1484
    Epoch 5/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.1196 - accuracy: 0.2428 - val_loss: 2.1671 - val_accuracy: 0.1742
    Epoch 6/100
    20/20 [==============================] - 0s 21ms/step - loss: 2.0289 - accuracy: 0.2540 - val_loss: 2.0592 - val_accuracy: 0.2452
    Epoch 7/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.9261 - accuracy: 0.2942 - val_loss: 2.0327 - val_accuracy: 0.2710
    Epoch 8/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.8571 - accuracy: 0.3392 - val_loss: 1.9731 - val_accuracy: 0.2774
    Epoch 9/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.8239 - accuracy: 0.3569 - val_loss: 1.9894 - val_accuracy: 0.3290
    Epoch 10/100
    20/20 [==============================] - 0s 24ms/step - loss: 1.7493 - accuracy: 0.3971 - val_loss: 1.8063 - val_accuracy: 0.4129
    Epoch 11/100
    20/20 [==============================] - 1s 25ms/step - loss: 1.7060 - accuracy: 0.3971 - val_loss: 1.8705 - val_accuracy: 0.4000
    Epoch 12/100
    20/20 [==============================] - 0s 23ms/step - loss: 1.5732 - accuracy: 0.4518 - val_loss: 1.7771 - val_accuracy: 0.4000
    Epoch 13/100
    20/20 [==============================] - 0s 25ms/step - loss: 1.5398 - accuracy: 0.4469 - val_loss: 2.0873 - val_accuracy: 0.3548
    Epoch 14/100
    20/20 [==============================] - 0s 25ms/step - loss: 1.5010 - accuracy: 0.4920 - val_loss: 1.6585 - val_accuracy: 0.4516
    Epoch 15/100
    20/20 [==============================] - 0s 24ms/step - loss: 1.4056 - accuracy: 0.5177 - val_loss: 1.7149 - val_accuracy: 0.4129
    Epoch 16/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.5106 - accuracy: 0.4920 - val_loss: 1.7744 - val_accuracy: 0.4516
    Epoch 17/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.3776 - accuracy: 0.5016 - val_loss: 1.6723 - val_accuracy: 0.4710
    Epoch 18/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.2911 - accuracy: 0.5547 - val_loss: 1.7622 - val_accuracy: 0.4645
    Epoch 19/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.2369 - accuracy: 0.5691 - val_loss: 1.7576 - val_accuracy: 0.4839
    Epoch 20/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.1607 - accuracy: 0.5981 - val_loss: 1.4524 - val_accuracy: 0.5355
    Epoch 21/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.0878 - accuracy: 0.6141 - val_loss: 1.7102 - val_accuracy: 0.5290
    Epoch 22/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.1051 - accuracy: 0.6206 - val_loss: 1.6052 - val_accuracy: 0.5355
    Epoch 23/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.9748 - accuracy: 0.6672 - val_loss: 1.4525 - val_accuracy: 0.5613
    Epoch 24/100
    20/20 [==============================] - 0s 22ms/step - loss: 1.0570 - accuracy: 0.6495 - val_loss: 1.5923 - val_accuracy: 0.5419
    Epoch 25/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.9478 - accuracy: 0.6720 - val_loss: 1.7746 - val_accuracy: 0.5419
    Epoch 26/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.9150 - accuracy: 0.6704 - val_loss: 1.9042 - val_accuracy: 0.5161
    Epoch 27/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.9810 - accuracy: 0.6640 - val_loss: 1.7715 - val_accuracy: 0.5484
    Epoch 28/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.8398 - accuracy: 0.7106 - val_loss: 2.0354 - val_accuracy: 0.5290
    Epoch 29/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.8724 - accuracy: 0.6849 - val_loss: 1.6763 - val_accuracy: 0.5290
    Epoch 30/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.7876 - accuracy: 0.7299 - val_loss: 1.9525 - val_accuracy: 0.5419
    Epoch 31/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.9141 - accuracy: 0.6720 - val_loss: 1.6522 - val_accuracy: 0.5419
    Epoch 32/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.7998 - accuracy: 0.7283 - val_loss: 1.5258 - val_accuracy: 0.5677
    Epoch 33/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.7116 - accuracy: 0.7476 - val_loss: 1.9832 - val_accuracy: 0.5484
    Epoch 34/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.7094 - accuracy: 0.7733 - val_loss: 1.4473 - val_accuracy: 0.5548
    Epoch 35/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.6919 - accuracy: 0.7444 - val_loss: 2.0650 - val_accuracy: 0.5419
    Epoch 36/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.6604 - accuracy: 0.7765 - val_loss: 1.6809 - val_accuracy: 0.5742
    Epoch 37/100
    20/20 [==============================] - 0s 25ms/step - loss: 0.6387 - accuracy: 0.7717 - val_loss: 1.6194 - val_accuracy: 0.5806
    Epoch 38/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.5311 - accuracy: 0.8055 - val_loss: 1.6718 - val_accuracy: 0.5871
    Epoch 39/100
    20/20 [==============================] - 1s 25ms/step - loss: 0.6356 - accuracy: 0.7894 - val_loss: 1.6033 - val_accuracy: 0.6258
    Epoch 40/100
    20/20 [==============================] - 1s 26ms/step - loss: 0.5327 - accuracy: 0.8232 - val_loss: 1.5790 - val_accuracy: 0.6258
    Epoch 41/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.5193 - accuracy: 0.8489 - val_loss: 1.8384 - val_accuracy: 0.5871
    Epoch 42/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.5460 - accuracy: 0.8248 - val_loss: 1.6660 - val_accuracy: 0.6452
    Epoch 43/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.5594 - accuracy: 0.8183 - val_loss: 1.5267 - val_accuracy: 0.6581
    Epoch 44/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.6105 - accuracy: 0.8006 - val_loss: 1.6175 - val_accuracy: 0.6194
    Epoch 45/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4740 - accuracy: 0.8296 - val_loss: 1.5857 - val_accuracy: 0.6194
    Epoch 46/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4803 - accuracy: 0.8328 - val_loss: 1.5215 - val_accuracy: 0.6645
    Epoch 47/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4694 - accuracy: 0.8248 - val_loss: 1.5766 - val_accuracy: 0.6645
    Epoch 48/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4203 - accuracy: 0.8553 - val_loss: 1.8034 - val_accuracy: 0.6129
    Epoch 49/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.3942 - accuracy: 0.8698 - val_loss: 1.4778 - val_accuracy: 0.6258
    Epoch 50/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4032 - accuracy: 0.8617 - val_loss: 1.6364 - val_accuracy: 0.6839
    Epoch 51/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.3928 - accuracy: 0.8617 - val_loss: 1.4809 - val_accuracy: 0.7032
    Epoch 52/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4878 - accuracy: 0.8441 - val_loss: 1.5021 - val_accuracy: 0.6645
    Epoch 53/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.3759 - accuracy: 0.8650 - val_loss: 1.8422 - val_accuracy: 0.6194
    Epoch 54/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.3752 - accuracy: 0.8730 - val_loss: 1.5582 - val_accuracy: 0.6581
    Epoch 55/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4182 - accuracy: 0.8585 - val_loss: 1.9785 - val_accuracy: 0.6387
    Epoch 56/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.4045 - accuracy: 0.8682 - val_loss: 1.5283 - val_accuracy: 0.6645
    Epoch 57/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.3447 - accuracy: 0.8698 - val_loss: 1.5673 - val_accuracy: 0.6194
    Epoch 58/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.3428 - accuracy: 0.8826 - val_loss: 1.8609 - val_accuracy: 0.6581
    Epoch 59/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2813 - accuracy: 0.9051 - val_loss: 1.7635 - val_accuracy: 0.6516
    Epoch 60/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2938 - accuracy: 0.8923 - val_loss: 2.1269 - val_accuracy: 0.6323
    Epoch 61/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.3095 - accuracy: 0.8971 - val_loss: 1.7226 - val_accuracy: 0.6774
    Epoch 62/100
    20/20 [==============================] - 1s 26ms/step - loss: 0.2817 - accuracy: 0.9132 - val_loss: 1.6342 - val_accuracy: 0.6452
    Epoch 63/100
    20/20 [==============================] - 1s 25ms/step - loss: 0.2943 - accuracy: 0.8971 - val_loss: 1.7971 - val_accuracy: 0.6645
    Epoch 64/100
    20/20 [==============================] - 0s 23ms/step - loss: 0.3097 - accuracy: 0.8987 - val_loss: 1.5645 - val_accuracy: 0.6645
    Epoch 65/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.3354 - accuracy: 0.8955 - val_loss: 1.9068 - val_accuracy: 0.6452
    Epoch 66/100
    20/20 [==============================] - 1s 25ms/step - loss: 0.2747 - accuracy: 0.9148 - val_loss: 1.6942 - val_accuracy: 0.6581
    Epoch 67/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.3014 - accuracy: 0.9035 - val_loss: 1.9937 - val_accuracy: 0.6452
    Epoch 68/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2851 - accuracy: 0.8939 - val_loss: 1.9181 - val_accuracy: 0.6774
    Epoch 69/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2523 - accuracy: 0.9164 - val_loss: 1.8624 - val_accuracy: 0.6581
    Epoch 70/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2677 - accuracy: 0.9132 - val_loss: 2.0224 - val_accuracy: 0.6065
    Epoch 71/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2557 - accuracy: 0.9100 - val_loss: 1.6450 - val_accuracy: 0.6452
    Epoch 72/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2329 - accuracy: 0.9116 - val_loss: 1.8229 - val_accuracy: 0.6903
    Epoch 73/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2670 - accuracy: 0.9132 - val_loss: 1.6190 - val_accuracy: 0.6452
    Epoch 74/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2541 - accuracy: 0.9277 - val_loss: 1.6706 - val_accuracy: 0.7032
    Epoch 75/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2671 - accuracy: 0.9100 - val_loss: 1.9038 - val_accuracy: 0.6645
    Epoch 76/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1794 - accuracy: 0.9437 - val_loss: 1.8540 - val_accuracy: 0.6774
    Epoch 77/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2160 - accuracy: 0.9228 - val_loss: 1.6928 - val_accuracy: 0.6968
    Epoch 78/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2432 - accuracy: 0.9116 - val_loss: 1.7868 - val_accuracy: 0.7097
    Epoch 79/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2349 - accuracy: 0.9373 - val_loss: 1.8322 - val_accuracy: 0.6903
    Epoch 80/100
    20/20 [==============================] - 0s 23ms/step - loss: 0.1814 - accuracy: 0.9357 - val_loss: 1.9648 - val_accuracy: 0.6903
    Epoch 81/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2447 - accuracy: 0.9180 - val_loss: 1.8417 - val_accuracy: 0.7097
    Epoch 82/100
    20/20 [==============================] - 0s 23ms/step - loss: 0.2341 - accuracy: 0.9196 - val_loss: 1.8670 - val_accuracy: 0.6581
    Epoch 83/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2462 - accuracy: 0.9132 - val_loss: 1.9938 - val_accuracy: 0.6839
    Epoch 84/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2707 - accuracy: 0.9164 - val_loss: 1.5018 - val_accuracy: 0.7290
    Epoch 85/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1963 - accuracy: 0.9357 - val_loss: 1.7165 - val_accuracy: 0.6968
    Epoch 86/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1366 - accuracy: 0.9566 - val_loss: 2.0390 - val_accuracy: 0.6516
    Epoch 87/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2328 - accuracy: 0.9196 - val_loss: 1.9439 - val_accuracy: 0.6581
    Epoch 88/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.2011 - accuracy: 0.9277 - val_loss: 2.0360 - val_accuracy: 0.7032
    Epoch 89/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.2322 - accuracy: 0.9084 - val_loss: 1.7971 - val_accuracy: 0.6452
    Epoch 90/100
    20/20 [==============================] - 0s 24ms/step - loss: 0.2055 - accuracy: 0.9357 - val_loss: 1.9844 - val_accuracy: 0.6323
    Epoch 91/100
    20/20 [==============================] - 0s 25ms/step - loss: 0.1635 - accuracy: 0.9437 - val_loss: 2.0740 - val_accuracy: 0.6839
    Epoch 92/100
    20/20 [==============================] - 1s 25ms/step - loss: 0.1911 - accuracy: 0.9325 - val_loss: 2.2594 - val_accuracy: 0.6194
    Epoch 93/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.2600 - accuracy: 0.9051 - val_loss: 2.0101 - val_accuracy: 0.6903
    Epoch 94/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1595 - accuracy: 0.9550 - val_loss: 1.7172 - val_accuracy: 0.7032
    Epoch 95/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1756 - accuracy: 0.9389 - val_loss: 1.6861 - val_accuracy: 0.7097
    Epoch 96/100
    20/20 [==============================] - 0s 23ms/step - loss: 0.1569 - accuracy: 0.9502 - val_loss: 1.8313 - val_accuracy: 0.6710
    Epoch 97/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1950 - accuracy: 0.9196 - val_loss: 1.8279 - val_accuracy: 0.7161
    Epoch 98/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1746 - accuracy: 0.9534 - val_loss: 2.2156 - val_accuracy: 0.6710
    Epoch 99/100
    20/20 [==============================] - 0s 21ms/step - loss: 0.2354 - accuracy: 0.9244 - val_loss: 2.0141 - val_accuracy: 0.7097
    Epoch 100/100
    20/20 [==============================] - 0s 22ms/step - loss: 0.1849 - accuracy: 0.9357 - val_loss: 1.8712 - val_accuracy: 0.7032
    

## Visualize training results

After applying data augmentation and `tf.keras.layers.Dropout`, there is less overfitting than before, and training and validation accuracy are closer aligned:


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```


    
![png](output_54_0.png)
    


### Case2) Activation function of Dense layer → softmax


```python
num_classes = len(class_names)

model = Sequential([
  data_augmentation, #데이터 어그멘테이션 추가
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), #dropout 사용
  layers.Flatten(),
  layers.Dense(128, activation='softmax'), # Activation function을 softmax로 사용
  layers.Dense(num_classes)
])
```

## Compile and train the model


```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     sequential_3 (Sequential)   (None, 160, 160, 3)       0         
                                                                     
     rescaling_5 (Rescaling)     (None, 160, 160, 3)       0         
                                                                     
     conv2d_9 (Conv2D)           (None, 160, 160, 16)      448       
                                                                     
     max_pooling2d_9 (MaxPooling  (None, 80, 80, 16)       0         
     2D)                                                             
                                                                     
     conv2d_10 (Conv2D)          (None, 80, 80, 32)        4640      
                                                                     
     max_pooling2d_10 (MaxPoolin  (None, 40, 40, 32)       0         
     g2D)                                                            
                                                                     
     conv2d_11 (Conv2D)          (None, 40, 40, 64)        18496     
                                                                     
     max_pooling2d_11 (MaxPoolin  (None, 20, 20, 64)       0         
     g2D)                                                            
                                                                     
     dropout_3 (Dropout)         (None, 20, 20, 64)        0         
                                                                     
     flatten_3 (Flatten)         (None, 25600)             0         
                                                                     
     dense_6 (Dense)             (None, 128)               3276928   
                                                                     
     dense_7 (Dense)             (None, 10)                1290      
                                                                     
    =================================================================
    Total params: 3,301,802
    Trainable params: 3,301,802
    Non-trainable params: 0
    _________________________________________________________________
    


```python
epochs=100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

    Epoch 1/100
    20/20 [==============================] - 3s 33ms/step - loss: 2.2937 - accuracy: 0.1704 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 2/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2884 - accuracy: 0.1752 - val_loss: 2.2918 - val_accuracy: 0.1806
    Epoch 3/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2852 - accuracy: 0.1752 - val_loss: 2.2904 - val_accuracy: 0.1806
    Epoch 4/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2833 - accuracy: 0.1752 - val_loss: 2.2888 - val_accuracy: 0.1806
    Epoch 5/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2818 - accuracy: 0.1752 - val_loss: 2.2879 - val_accuracy: 0.1806
    Epoch 6/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2808 - accuracy: 0.1752 - val_loss: 2.2870 - val_accuracy: 0.1806
    Epoch 7/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2796 - accuracy: 0.1752 - val_loss: 2.2863 - val_accuracy: 0.1806
    Epoch 8/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2788 - accuracy: 0.1752 - val_loss: 2.2858 - val_accuracy: 0.1806
    Epoch 9/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2782 - accuracy: 0.1752 - val_loss: 2.2853 - val_accuracy: 0.1806
    Epoch 10/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2776 - accuracy: 0.1752 - val_loss: 2.2852 - val_accuracy: 0.1806
    Epoch 11/100
    20/20 [==============================] - 0s 24ms/step - loss: 2.2770 - accuracy: 0.1752 - val_loss: 2.2851 - val_accuracy: 0.1806
    Epoch 12/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2768 - accuracy: 0.1752 - val_loss: 2.2845 - val_accuracy: 0.1806
    Epoch 13/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2763 - accuracy: 0.1752 - val_loss: 2.2845 - val_accuracy: 0.1806
    Epoch 14/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2761 - accuracy: 0.1752 - val_loss: 2.2847 - val_accuracy: 0.1806
    Epoch 15/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2758 - accuracy: 0.1752 - val_loss: 2.2847 - val_accuracy: 0.1806
    Epoch 16/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2756 - accuracy: 0.1752 - val_loss: 2.2847 - val_accuracy: 0.1806
    Epoch 17/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2755 - accuracy: 0.1752 - val_loss: 2.2848 - val_accuracy: 0.1806
    Epoch 18/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2754 - accuracy: 0.1752 - val_loss: 2.2852 - val_accuracy: 0.1806
    Epoch 19/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2753 - accuracy: 0.1752 - val_loss: 2.2846 - val_accuracy: 0.1806
    Epoch 20/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2751 - accuracy: 0.1752 - val_loss: 2.2848 - val_accuracy: 0.1806
    Epoch 21/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2751 - accuracy: 0.1752 - val_loss: 2.2851 - val_accuracy: 0.1806
    Epoch 22/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2751 - accuracy: 0.1752 - val_loss: 2.2850 - val_accuracy: 0.1806
    Epoch 23/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2851 - val_accuracy: 0.1806
    Epoch 24/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2750 - accuracy: 0.1752 - val_loss: 2.2851 - val_accuracy: 0.1806
    Epoch 25/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2853 - val_accuracy: 0.1806
    Epoch 26/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2857 - val_accuracy: 0.1806
    Epoch 27/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2750 - accuracy: 0.1752 - val_loss: 2.2858 - val_accuracy: 0.1806
    Epoch 28/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2858 - val_accuracy: 0.1806
    Epoch 29/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2856 - val_accuracy: 0.1806
    Epoch 30/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2858 - val_accuracy: 0.1806
    Epoch 31/100
    20/20 [==============================] - 0s 24ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2861 - val_accuracy: 0.1806
    Epoch 32/100
    20/20 [==============================] - 1s 25ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2859 - val_accuracy: 0.1806
    Epoch 33/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2862 - val_accuracy: 0.1806
    Epoch 34/100
    20/20 [==============================] - 1s 25ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2861 - val_accuracy: 0.1806
    Epoch 35/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2862 - val_accuracy: 0.1806
    Epoch 36/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2863 - val_accuracy: 0.1806
    Epoch 37/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2863 - val_accuracy: 0.1806
    Epoch 38/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2863 - val_accuracy: 0.1806
    Epoch 39/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2866 - val_accuracy: 0.1806
    Epoch 40/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2866 - val_accuracy: 0.1806
    Epoch 41/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2867 - val_accuracy: 0.1806
    Epoch 42/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2867 - val_accuracy: 0.1806
    Epoch 43/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2867 - val_accuracy: 0.1806
    Epoch 44/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2870 - val_accuracy: 0.1806
    Epoch 45/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 46/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 47/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2869 - val_accuracy: 0.1806
    Epoch 48/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2869 - val_accuracy: 0.1806
    Epoch 49/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2868 - val_accuracy: 0.1806
    Epoch 50/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2868 - val_accuracy: 0.1806
    Epoch 51/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2869 - val_accuracy: 0.1806
    Epoch 52/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 53/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 54/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2870 - val_accuracy: 0.1806
    Epoch 55/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 56/100
    20/20 [==============================] - 0s 25ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 57/100
    20/20 [==============================] - 0s 24ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 58/100
    20/20 [==============================] - 1s 26ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2870 - val_accuracy: 0.1806
    Epoch 59/100
    20/20 [==============================] - 0s 24ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 60/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 61/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 62/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2873 - val_accuracy: 0.1806
    Epoch 63/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 64/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 65/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 66/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 67/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2877 - val_accuracy: 0.1806
    Epoch 68/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 69/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2873 - val_accuracy: 0.1806
    Epoch 70/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 71/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2874 - val_accuracy: 0.1806
    Epoch 72/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2874 - val_accuracy: 0.1806
    Epoch 73/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2878 - val_accuracy: 0.1806
    Epoch 74/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 75/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2874 - val_accuracy: 0.1806
    Epoch 76/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2873 - val_accuracy: 0.1806
    Epoch 77/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2873 - val_accuracy: 0.1806
    Epoch 78/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 79/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 80/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 81/100
    20/20 [==============================] - 0s 24ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 82/100
    20/20 [==============================] - 0s 24ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2878 - val_accuracy: 0.1806
    Epoch 83/100
    20/20 [==============================] - 1s 26ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2876 - val_accuracy: 0.1806
    Epoch 84/100
    20/20 [==============================] - 1s 26ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 85/100
    20/20 [==============================] - 1s 25ms/step - loss: 2.2749 - accuracy: 0.1752 - val_loss: 2.2874 - val_accuracy: 0.1806
    Epoch 86/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 87/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2874 - val_accuracy: 0.1806
    Epoch 88/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2873 - val_accuracy: 0.1806
    Epoch 89/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2870 - val_accuracy: 0.1806
    Epoch 90/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2874 - val_accuracy: 0.1806
    Epoch 91/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2877 - val_accuracy: 0.1806
    Epoch 92/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 93/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2875 - val_accuracy: 0.1806
    Epoch 94/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2876 - val_accuracy: 0.1806
    Epoch 95/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2748 - accuracy: 0.1752 - val_loss: 2.2876 - val_accuracy: 0.1806
    Epoch 96/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 97/100
    20/20 [==============================] - 0s 23ms/step - loss: 2.2746 - accuracy: 0.1752 - val_loss: 2.2872 - val_accuracy: 0.1806
    Epoch 98/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2874 - val_accuracy: 0.1806
    Epoch 99/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2871 - val_accuracy: 0.1806
    Epoch 100/100
    20/20 [==============================] - 0s 22ms/step - loss: 2.2747 - accuracy: 0.1752 - val_loss: 2.2877 - val_accuracy: 0.1806
    

## Visualize training results


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```


    
![png](output_62_0.png)
    


## Conclusion:
- It is difficult to learn with small amounts of data.

- There are different models for different data characteristics.
  For this data, relu was more appropriate than softmax for activation.

- Typically, when we apply deep learning, we apply models with a specific structure, such as CNN or Transformer, and these models may not be the right structure to learn the data for this particular sale.

- Just like the right tool for the right situation, the right machine learning model for the right problem will vary (deep learning is not one-size-fits-all).

