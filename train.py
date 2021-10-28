import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

'''
input_list = np.array(pickle.load(open('./pkls/keypoint_list.pkl','rb')))
output_list = np.array(pickle.load(open('./pkls/label_list.pkl','rb')))

kf = KFold(n_splits=10, shuffle=True, random_state=1)

acc_list = []
for train_idx, test_idx in kf.split(input_list):
    X_train, X_test = input_list[train_idx], input_list[test_idx]
    y_train, y_test = output_list[train_idx], output_list[test_idx]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(256,32), random_state=1, early_stopping=True)
    clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    acc = accuracy_score(y_test, y_prediction)
    acc_list.append(acc)

print(sum(acc_list)/len(acc_list))
print(acc_list)
'''
X_train = np.array(pickle.load(open('./pkls/keypoint_list.pkl','rb')))
y_train = np.array(pickle.load(open('./pkls/label_list.pkl','rb')))
X_test = np.array(pickle.load(open('./pkls/test_keypoint_list.pkl','rb')))
y_test = np.array(pickle.load(open('./pkls/test_label_list.pkl','rb')))

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,64,32), random_state=1, early_stopping=True)
#clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)
acc = accuracy_score(y_test, y_prediction)
print(acc)

pickle.dump(clf, open('./pkls/trained_model.pkl','wb'))
pickle.dump(scaler, open('./pkls/trained_scaler.pkl','wb'))

