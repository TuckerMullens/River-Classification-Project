import numpy as np
import os
import cv2
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score

#Loading Images and Labels into training_data
DATADIR = "RiverImages"
CATEGORIES = ["river", "noriver"] #0 for river and 1 for noriver

training_data = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        training_data.append([img_array, class_num])

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

Y = np.array(Y).reshape(len(Y), 1)

hog_images = []
hog_features = []

#Creating the hog features for the images
for sample in training_data:
    fd, hog_image = hog(sample[0], orientations=8, pixels_per_cell=(10, 10), cells_per_block=(4, 4), block_norm='L2-Hys',
                        visualize=True)
    hog_images.append(hog_image)
    hog_features.append(fd)

hog_features = np.array(hog_features)

#Finding the best values for C and gamma for the classifier
def svc_param_selection(X, Y):
    Y = Y.ravel()
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
    grid_search.fit(X, Y)
    grid_search.best_params_
    return grid_search.best_params_

#Returning the best values for C and gamma
best_param = svc_param_selection(hog_features, Y)

#Moving hog_features and labels into a data_frame and shuffling it
data_frame = np.hstack((hog_features, Y))

np.random.shuffle(data_frame)

#Creating the SMV classifier
clf = svm.SVC(C=0.001, gamma=0.1)

#Partitioning data_frame into training and testing data
x_train, x_test = data_frame[:90, :-1], data_frame[90:, :-1]
y_train, y_test = data_frame[:90, -1:].ravel(), data_frame[90:, -1:].ravel()

#Fitting model to training data
clf.fit(x_train, y_train)

#Generating predictions for test data
y_pred = clf.predict(x_test)

#Printing Results
print("Results:")
for i,j in zip(y_pred, y_test):
    print("Predicted:", end=" ")
    print(i, end=" ")
    print("Actual:", end=" ")
    print(j)

print("Accuracy: " + str(accuracy_score(y_test, y_pred)))