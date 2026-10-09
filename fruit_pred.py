import os
import cv2
import numpy as np

data = "fruit_images"

X = []
y = []

classes = os.listdir(data)

for label, fruit in enumerate(classes):
    print(label, "=", fruit)

print("Total classes:", len(classes))

for labels, fruits in enumerate(classes):

    fruits_path = os.path.join(data, fruits)

    for image_name in os.listdir(fruits_path):

        image_path = os.path.join(fruits_path, image_name)

        image = cv2.imread(image_path)


        if image is None:
            continue


        image = cv2.resize(image, (128, 128))

        X.append(image)

        y.append(labels)


X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42,)

from tensorflow import keras
from tensorflow.keras import layers

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.Sequential([
    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(9,activation="softmax"),
    # layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train,y_train,epochs=20)


# image = cv2.imread("apple_image2.jpg")
# # image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
# image = cv2.resize(image,(128,128))
# image = image / 255.0
# image = np.expand_dims(image, axis=0)
#
# pred = model.predict(image)
# prediction = np.argmax(pred)
#
# print("Predicted class number:", prediction)
#
# print(f"This Fruit is {classes[prediction]}")
#
# print(f"This Fruit is {np.argmax(pred)}")


