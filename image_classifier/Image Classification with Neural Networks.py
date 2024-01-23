import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers

def load_and_preprocess(images, labels, batch_size):
    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_images = images[start_idx:end_idx] / 255.0
        batch_labels = labels[start_idx:end_idx]
        yield batch_images, batch_labels

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

batch_size = 1000

train_generator = load_and_preprocess(training_images, training_labels, batch_size)
test_generator = load_and_preprocess(testing_images, testing_labels, batch_size)

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(10, 10))

for batch_images, batch_labels in train_generator:
    for i in range(min(16, batch_images.shape[0])):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(batch_images[i])
        label_index = batch_labels[i][0]

        if label_index < len(class_names):
            plt.xlabel(class_names[label_index])
        else:
            plt.xlabel("Unknown Class")

    plt.show()
    break

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')

loaded_model = models.load_model('image_classifier.model')

img = cv.imread('D:\michine Learning Projects\pythonProject\image_classifier.model\deer.jpg')
if img is not None:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Correcting color channels

    img = cv.resize(img, (32, 32))

    img = img / 255.0

    prediction = loaded_model.predict(np.array([img]))

    index = np.argmax(prediction)

    print(f'Prediction: {class_names[index]}')
else:
    print("Error: Unable to read the image.")
