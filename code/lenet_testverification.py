# loading trained Lenet module
import tensorflow as tf

model = tf.keras.models.load_model('lenet_model.h5')
model.summary()

import cv2
import matplotlib.pyplot as plt

# Step 1: read the image
img = cv2.imread('6.jpg')
print(img.shape)

# Step 2: convert this image into grey image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
# plt.imshow(img,cmap='Greys')
# plt.show()

# Step 3: bitwise the color of digit and background
img = cv2.bitwise_not(img)
# plt.imshow(img,cmap='Greys')
# plt.show()
# Step 4: make the background pure black and digit pure white
img[img<=100]=0
img[img>140]=255  # 130

# Displaying the image
# plt.imshow(img,cmap='Greys')
# plt.show()
# Part 5: resize the given image
img = cv2.resize(img,(32,32))
plt.imshow(img, cmap='Greys')
plt.show()
# Step 6: converting the type to float32
img = img.astype('float32')

# Step 7: normalisation
img /= 255

# Step 8: increase the number of demention
img = img.reshape(1, 32, 32, 1)
print(img.shape)


# Step 9: prediction
pred = model.predict(img)

# Step 10: result
print(pred.argmax())