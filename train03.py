import tensorflow as tf
import numpy as np
import pandas as pd
import os
import imageio
import shutil
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import Image
from shutil import copy

data_dir = 'D:/20201/AI/leedsbutterfly/'
working_dir = 'D:/20201/AI/working'
model_dir = 'D:/20201/AI/working/models'
image_dir = os.path.join(data_dir, "images")
segmentation_dir = os.path.join(data_dir, "segmentations")

image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
segmentation_files = [f for f in os.listdir(segmentation_dir) if os.path.isfile(os.path.join(segmentation_dir, f))]
labels = np.array([f[:3] for f in image_files]).astype('int32') - 1

image_files = [os.path.join(image_dir, f) for f in image_files]
segmentation_files = [os.path.join(segmentation_dir, f) for f in segmentation_files]

try: 
    os.mkdir(model_dir) 
except OSError as error:
    print("")

def copy_files(filenames, dest_dir, labelname):
    label_dest_dir = os.path.join(dest_dir, str(label))

    if os.path.isdir(label_dest_dir):
        shutil.rmtree(label_dest_dir, ignore_errors=True)

    os.makedirs(label_dest_dir)

    
    [copy(file , label_dest_dir) for file in filenames]
    
    return

def copy_images_and_masks(samples, masks, sample_dir, mask_dir, labels, current_label):
    sample_idx = np.where(labels == label)[0]
    sample_data = np.array(samples)[sample_idx]
    masks_data = np.array(masks)[sample_idx]

    copy_files(sample_data, sample_dir, label)
    copy_files(masks_data, mask_dir, label)

image_segment_files = list(zip(image_files, segmentation_files))
X_train_seg, X_test_seg, y_train, y_test = train_test_split(image_segment_files, labels, train_size=0.8, random_state=5634)
X_train, X_train_segment = zip(*X_train_seg)
X_test, X_test_segment = zip(*X_test_seg)

unique_labels = np.unique(labels)
train_dest_dir = os.path.join(working_dir, "train")
train_segment_dest_dir = os.path.join(working_dir, "train_segment")
test_dest_dir = os.path.join(working_dir, "test")
test_segment_dest_dir = os.path.join(working_dir, "test_segment")

for label in unique_labels:
    copy_images_and_masks(X_train, X_train_segment, train_dest_dir, train_segment_dest_dir, y_train, label)
    copy_images_and_masks(X_test, X_test_segment, test_dest_dir, test_segment_dest_dir, y_test, label)

def show_images_from_file(base_dir, label, num_images=25, num_per_row=5):
    image_dir = os.path.join(base_dir, str(label))

    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    files = files[:num_images]
    
    images_per_row = min(num_images, num_per_row)
    n_rows = (num_images - 1) // images_per_row + 1

    row_images = []
    
    for row in range(n_rows):
        current_files = files[row * images_per_row : (row + 1) * images_per_row]
        images = [tf.image.resize(imageio.imread(file), [100, 100], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) for file in current_files]
        row_images.append(np.concatenate(images, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.figure(figsize = (images_per_row * 2, n_rows * 2))
    plt.imshow(image, interpolation='nearest')
    plt.axis("off")

    return
for label in np.unique(labels):
    print("-----------------------------")
    print("Label: {}".format(label))
    print("")
    show_images_from_file(test_dest_dir, label, num_images=1)
    plt.show()

def create_data_generator(data_dir, batch_size, data_seed, target_size, args):
    datagen = ImageDataGenerator(**args)
    
    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=data_seed
    )
    
    return data_generator

def create_sample_mask_generator(sample_dir, mask_dir, batch_size, data_seed, target_size, args):
    sample_datagen = create_data_generator(sample_dir, batch_size, data_seed, target_size, args)
    mask_datagen = create_data_generator(mask_dir, batch_size, data_seed, target_size, args)
    
    return [sample_datagen, mask_datagen]

batch_size = 32
data_seed = 432
target_size = (100,100)

train_data_gen_args = dict(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_data_gen_args = dict(
    rescale=1./255
)

train_generator, train_mask_generator = create_sample_mask_generator(
    train_dest_dir,
    train_segment_dest_dir,
    batch_size,
    data_seed,
    target_size,
    train_data_gen_args
)

test_generator, test_mask_generator = create_sample_mask_generator(
    test_dest_dir,
    test_segment_dest_dir,
    batch_size,
    data_seed,
    target_size,
    test_data_gen_args
)

image_shape = target_size + (3,)

model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=image_shape, name='conv_1'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='conv_2'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv_3'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu', name='dense_1'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='sigmoid', name='outputs')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

model.fit_generator(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(X_test) // batch_size
)

model_filename = os.path.join(model_dir, "model_no_mask_{}.hdf5".format(timestamp))

print("Saving model in file {}".format(model_filename))
model.save(model_filename)
'''
checkpoint = ModelCheckpoint('D:/20201/AI/weights22.hdf5',monitor='val_loss',verbose=1,save_best_only=True,mode='min')
history=model.fit(train_data,train_target,epochs=20, batch_size= 16 ,callbacks=[checkpoint],validation_split=0.2)'''