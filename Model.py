import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from challenge2017.lib.record import *
from sklearn.metrics import accuracy_score
from random import shuffle
import random
import cv2
from challenge2017.lib.dataset import record_reader, get_unique_labels
#!/usr/bin/env python3



def preProcess(img):

    img = cv2.resize(img, (0,0), fx=0.75, fy=0.75);
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    detector = cv2.FeatureDetector_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)


    #img = img[:90][:140]

    #print(len(skp))
    #print(dst.shape)

    List = []
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            List.append(dst[i][j])

    #skp.reshape(1, -1)
   # for feature in skp:
    #    List.append(feature.pt[0])
    return List



# browse filetree and find all data records
records = record_reader("C:\Users\Valio\Desktop\Hackatum_data\car")
all_labels = get_unique_labels(records)

Y = []
X = []
empty = [None]
print("Available records {}".format(len(records)))

# check that all labels naming one cathegory are of the same size
for i in range(11000,12000):
    rec = records[i]
    #img = get_image_corner(rec.get_image(), (True, True))
    if (rec.get_image() is None):
        print("Problem")
    y = len(rec.labels)
#    img = np.reshape(img,(90*140,-1))
    if y is 0:
        List = preProcess(rec.get_image())
        #print(img.shape)
        Y.append(y)
        X.append(List)

# Logo positive data
# s=selected i = increment
s=0
i=0
len_y = len(Y)
while s < len_y and i < 4000:
    i = i+1
    rec = records[i]
    #img = get_image_corner(rec.get_image(), (True, True))
    if (rec.get_image() is None):
        print("Problem")
    y = len( rec.labels)
#    img = np.reshape(img,(90*140,-1))
    if y is not 0:
        s = s+1
        List = preProcess(rec.get_image())
        #print(img.shape)
        Y.append(1)
        X.append(List)


#print(len(Y))

clf = KNeighborsClassifier()

#nsamples, nx, ny = X.shape
#d2_train_dataset = X.reshape((nsamples,nx*ny))
bound = 80

Together = [X, Y]
Together = shuffle(Together)

combined = list(zip(X, Y))
random.shuffle(combined)

X[:], Y[:] = zip(*combined)

#print(Y.count(True))
#print(len(Y))

clf.fit(X[:bound], Y[:bound])


fake = cv2.imread("C:\Users\Valio\Desktop\Hackatum_data\car\prosieben2_fake.jpg", 0)
real = cv2.imread("C:\Users\Valio\Desktop\Hackatum_data\car\prosieben2.jpg", 0)

X2 = [preProcess(fake), preProcess(real)]

Y2 = [1, 0]

pred2 = clf.predict(X2)
print(pred2)
#precision = accuracy_score(pred2, Y2)

#print(precision)



pred = clf.predict(X[bound:])
print(pred)
precision = accuracy_score(pred, Y[bound:])

print(precision)



if __name__ == "__main__":
   pass
