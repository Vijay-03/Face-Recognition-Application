import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('vegita.jpg')
gray_image = cv2.imread('vegita.jpg', 0)
# print(image.shape)

cv2.imshow('Vegita', image)
cv2.imshow('Vegita gray', gray_image)

cv2.waitKey((0))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(image)
#plt.title("Vegita")
#plt.show()

#plt.imshow(image[:, :970])
#plt.title("Vegita left half")
#plt.show()

#plt.imshow(image[:, 970:])
#plt.title("Vegita right half")
#plt.show()
