from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.models import Model

inputs = Input((1024, 1024, 3))

#Encoder
c1 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(inputs)
c1 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(p1)
c2 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(p2)
c3 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(p3)
c4 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(c4)
p4 = MaxPooling2D((2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(p4)
c5 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(c5)

#Decoder
u6 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(u6)
c6 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(u7)
c7 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(u8)
c8 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = 'same')(c8)
u9 = concatenate([u9, c1])
c9 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(u9)
c9 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(c9)

outputs = Conv2D(1, (1, 1), activation = 'sigmoid')(c9)

model = Model(inputs= [inputs], outputs= [outputs])

model.summary()


from keras import backend as K

def Iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


seed = 24
batch_size = 4

img_data_gen_args = dict(rescale = 1/255.)

mask_data_gen_args = dict(rescale = 1/255.)


image_data_generator = ImageDataGenerator(**img_data_gen_args)
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

image_generator = image_data_generator.flow_from_directory("/content/drive/MyDrive/GIRS Data/Aerial Dataset/train_images",
                                                           seed = seed,
                                                           batch_size = batch_size,
                                                           target_size = (1024, 1024),
                                                           class_mode = None)

mask_generator = mask_data_generator.flow_from_directory("/content/drive/MyDrive/GIRS Data/Aerial Dataset/train_masks",
                                                          seed = seed,
                                                          batch_size = batch_size,
                                                          target_size = (1024, 1024),
                                                          color_mode = 'grayscale',
                                                          class_mode = None)

from matplotlib import pyplot as plt
import numpy as np

# Get the next batch of images and masks
x = next(image_generator)
y = next(mask_generator)

# Debugging step
print(f"Image batch shape: {x.shape}")
print(f"Mask batch shape: {y.shape}")

# Ensure both the image and mask batches have the same length
assert x.shape[0] == y.shape[0], "Mismatch in number of images and masks"

# Visualize the first pair of image and mask
for i in range(1):  # Adjust the range if you want to visualize more images/masks
    image = x[i]
    mask = y[i]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap='gray')  # Ensure the mask is displayed correctly in grayscale
    plt.title("Mask")
    plt.axis('off')

    plt.show()

from tensorflow.keras.optimizers.legacy import Adam
model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = [Iou, 'accuracy'])


history = model.fit_generator(train_generator, validation_data = val_generator,
                              steps_per_epoch = 45,
                              validation_steps = 5,
                              epochs = 20,
                              callbacks = callbacks)









