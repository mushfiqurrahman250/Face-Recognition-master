import cv2

src = cv2.imread('D:/untitled/Face-Recognition-master/images', cv2.IMREAD_UNCHANGED)

print(src)

image = cv2.resize(src, (224, 224))

# dsize

dsize = image

# resize image
output = cv2.resize(src, dsize)

cv2.imwrite('D:/untitled/Face-Recognition-master/test1', output)