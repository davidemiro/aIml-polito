from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import sklearn.neighbors as knn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from matplotlib.colors import ListedColormap
import sklearn.svm as svm
import sklearn.metrics as mc
import sklearn.model_selection as ms





# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
def printKNNBoundaries(X,y,clf,n,cmap_light,cmap_bold):
    h = .02
    # calculate min, max and limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # predict class using data and kNN classifier
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % (n))
    plt.show()
def printSVM(X,y,clf,n):
    h =.02
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y,cmap=cmap_bold)
    plt.show()

X = load_wine()


#stampo i primi due attributi del datasets con classe
plt.scatter(X.data[:, 0], X.data[:, 1], c=X.target, cmap=cmap_bold)


#Splitto il dataset in train-validation-test

X_train,X_t,Y_train,Y_t= train_test_split(X.data,X.target,test_size=0.5,random_state=1)
X_validation,X_test,Y_validation,Y_test = train_test_split(X_t,Y_t,test_size=0.6,random_state=1)
plt.scatter(X_train[:,0],X_train[:, 1],c = Y_train,cmap=cmap_bold)
X_fold = np.concatenate((X_train,X_validation))
Y_fold = np.concatenate((Y_train,Y_validation))



#k = 1
knn_1 =knn.KNeighborsClassifier(1)
knn_1.fit(X_train[:,:2],Y_train)
printKNNBoundaries(X_train,Y_train,knn_1,1,cmap_light,cmap_bold)
Y_predict = knn_1.predict(X_validation[:,:2])
score1 = mc.accuracy_score(Y_validation,Y_predict)
#print(score1)

#k = 3
knn_3 =knn.KNeighborsClassifier(3)
knn_3.fit(X_train[:,:2],Y_train)
printKNNBoundaries(X_train,Y_train,knn_3,3,cmap_light,cmap_bold)
Y_predict = knn_3.predict(X_validation[:,:2])
score3 = mc.accuracy_score(Y_validation,Y_predict)
#print(score3)


#k = 5
knn_5 =knn.KNeighborsClassifier(5)
knn_5.fit(X_train[:,:2],Y_train)
printKNNBoundaries(X_train,Y_train,knn_5,5,cmap_light,cmap_bold)
Y_predict = knn_5.predict(X_validation[:,:2])
score5 = mc.accuracy_score(Y_validation,Y_predict)
#print(score5)

#k = 7
knn_7 =knn.KNeighborsClassifier(7)
knn_7.fit(X_train[:,:2],Y_train)
printKNNBoundaries(X_train,Y_train,knn_7,7,cmap_light,cmap_bold)
Y_predict = knn_7.predict(X_validation[:,:2])
score7 = mc.accuracy_score(Y_validation,Y_predict)
#print(score7)

scores = [score1,score3,score5,score7]
models = [knn_1,knn_3,knn_5,knn_7]
plt.plot([1,3,5,7],scores)

max = .0
n = 0

#select the best k
for i in scores:
    if i > max:
        max = i
        n = scores.index(i)*2 + 1

#print(n)
#testin
final_model =knn.KNeighborsClassifier(n)
final_model.fit(X_fold[:,:2],Y_fold)
Y_predict = final_model.predict(X_test[:,:2])
final_score = mc.accuracy_score(Y_test,Y_predict)
print(final_score)
print("KNN %i : %f" % (models[(n-1)//2].get_params()["n_neighbors"],final_score))
plt.show()

#C= 0.001
svm_001 = svm.SVC(kernel='linear',C=0.001)
svm_001.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm_001,0.001)
Y_predict = svm_001.predict(X_validation[:,:2])
score_001 = mc.accuracy_score(Y_validation,Y_predict)
#print(score_001)

#C= 0.01
svm_01 = svm.SVC(kernel='linear',C=0.01)
svm_01.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm_01,0.01)
Y_predict = svm_01.predict(X_validation[:,:2])
score_01 = mc.accuracy_score(Y_validation,Y_predict)
#print(score_01)

#C= 0.1
svm_1 = svm.SVC(kernel='linear',C=0.1)
svm_1.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm_1,0.1)
Y_predict = svm_1.predict(X_validation[:,:2])
score_1 = mc.accuracy_score(Y_validation,Y_predict)
#print(score_1)

#C= 1
svm1 = svm.SVC(kernel='linear',C=1)
svm1.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm1,1)
Y_predict = svm1.predict(X_validation[:,:2])
score1 = mc.accuracy_score(Y_validation,Y_predict)
#print(score1)

#C=10
svm10 = svm.SVC(kernel='linear',C=10)
svm10.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm10,10)
Y_predict = svm1.predict(X_validation[:,:2])
score10 = mc.accuracy_score(Y_validation,Y_predict)
#print(score10)

#C=100
svm100 = svm.SVC(kernel='linear',C=100)
svm100.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm100,100)
Y_predict = svm1.predict(X_validation[:,:2])
score100 = mc.accuracy_score(Y_validation,Y_predict)
#print(score100)

#C=1000
svm1000 = svm.SVC(kernel='linear',C=1000)
svm1000.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm1000,1000)
Y_predict = svm1.predict(X_validation[:,:2])
score1000 = mc.accuracy_score(Y_validation,Y_predict)
#print(score1000)

models =[svm_001, svm_01, svm_1, svm1, svm10, svm100, svm1000]
scores = [score_001,score_01,score_1,score1,score10,score100,score1000]
C = [0.001,0.01,0.1,1,10,100,1000]

plt.semilogx()
plt.plot(C,scores)

max = .0
n = 0

#select the best C
for i in scores:
    if i > max:
        max = i
        n = scores.index(i)

#testing
final_model =svm.SVC(kernel='linear',C=C[n])
final_model.fit(X_fold[:,:2],Y_fold)
Y_predict = final_model.predict(X_test[:,:2])
final_score = mc.accuracy_score(Y_test,Y_predict)
print(final_score)
plt.show()

#SVM with rbf kernel

#C= 0.001
svm_001 = svm.SVC(kernel='rbf',C=0.001,gamma='auto')
svm_001.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm_001,0.001)
Y_predict = svm_001.predict(X_validation[:,:2])
score_001 = mc.accuracy_score(Y_validation,Y_predict)
#print(score_001)

#C= 0.01
svm_01 = svm.SVC(kernel='rbf',C=0.01,gamma='auto')
svm_01.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm_01,0.01)
Y_predict = svm_01.predict(X_validation[:,:2])
score_01 = mc.accuracy_score(Y_validation,Y_predict)
#print(score_01)

#C= 0.1
svm_1 = svm.SVC(kernel='rbf',C=0.1,gamma='auto')
svm_1.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm_1,0.1)
Y_predict = svm_1.predict(X_validation[:,:2])
score_1 = mc.accuracy_score(Y_validation,Y_predict)
#print(score_1)

#C= 1
svm1 = svm.SVC(kernel='rbf',C=1,gamma='auto')
svm1.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm1,1)
Y_predict = svm1.predict(X_validation[:,:2])
score1 = mc.accuracy_score(Y_validation,Y_predict)
#print(score1)

#C=10
svm10 = svm.SVC(kernel='rbf',C=10,gamma='auto')
svm10.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm10,10)
Y_predict = svm1.predict(X_validation[:,:2])
score10 = mc.accuracy_score(Y_validation,Y_predict)
#print(score10)

#C=100
svm100 = svm.SVC(kernel='rbf',C=100,gamma='auto')
svm100.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm100,100)
Y_predict = svm1.predict(X_validation[:,:2])
score100 = mc.accuracy_score(Y_validation,Y_predict)
#print(score100)

#C=1000
svm1000 = svm.SVC(kernel='rbf',C=1000,gamma='auto')
svm1000.fit(X_train[:,:2],Y_train)
printSVM(X_train,Y_train,svm1000,1000)
Y_predict = svm1.predict(X_validation[:,:2])
score1000 = mc.accuracy_score(Y_validation,Y_predict)
#print(score1000)

models =[svm_001, svm_01, svm_1, svm1, svm10, svm100, svm1000]
scores = [score_001,score_01,score_1,score1,score10,score100,score1000]


plt.semilogx()
plt.plot(C,scores)

max = .0
n = 0

#select the best C
for i in scores:
    if i > max:
        max = i
        n = scores.index(i)

#testing
final_model =svm.SVC(kernel='rbf',C=C[n],gamma='auto')
final_model.fit(X_fold[:,:2],Y_fold)
Y_predict = final_model.predict(X_test[:,:2])
final_score = mc.accuracy_score(Y_test,Y_predict)
print(final_score)

#cross validation
C_range = 10. ** np.arange(-3, 8)
gamma_range = 10. ** np.arange(-5, 4)
max =.0
gamma_max = .0
C_max = .0
for c in C_range:
    for g in gamma_range:
        svf = svm.SVC(kernel='rbf',gamma=g,C=c)
        svf.fit(X_train[:,:2],Y_train)
        score = mc.accuracy_score(Y_validation,svf.predict(X_validation[:,:2]))
        if(score > max):
            max = score
            gamma_max = g
            C_max = c
svf = svm.SVC(kernel='rbf',gamma=gamma_max,C=C_max)
svf.fit(X_fold[:,:2],Y_fold)
score = mc.accuracy_score(Y_test,svf.predict(X_test[:,:2]))
print(max)
print(score)

printSVM(X_fold,Y_fold,svf,c)

#5-fold cross validation
max =.0
max_gamma =.0
max_C =.0

C_range = 10. ** np.arange(-3, 8)
gamma_range = 10. ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = ms.GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5,iid=False)
grid.fit(X_fold[:,:2],Y_fold)

#print(grid.score(X_test[:,:2],Y_test))
#print(grid.best_params_)
score_dict = grid.cv_results_
scores = score_dict['mean_test_score'].reshape(len(C_range), len(gamma_range))

#cross validation graph
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest')
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.show()


gamma_range = np.linspace(grid.best_params_['gamma']/10,grid.best_params_['gamma'] *10,100)
C_range = np.linspace(grid.best_params_['C']/10,grid.best_params_['C']*10,100)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = ms.GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5,iid=False)
grid.fit(X_fold[:,:2],Y_fold)
model = svm.SVC(kernel='rbf',gamma=grid.best_params_['gamma'],C=grid.best_params_[ 'C'])
model.fit(X_fold[:,:2],Y_fold)
printSVM(X_fold[:,:2],Y_fold,model,grid.best_params_['C'])
print(grid.score(X_test[:,:2],Y_test))
print(grid.best_params_)


X = load_wine()
X_train,X_test,Y_train,Y_test= train_test_split(X.data,X.target,test_size=0.3,random_state=1)
gamma =[0.001,0.01,0.1,1,10,100,1000]
C = [0.001,0.01,0.1,1,10,100,1000]
k =[1,3,5,7]
paramitersSVM = [{'kernel':['rbf','linear'],'gamma':gamma,'C':C}]
paramitersKNN =[{'n_neighbors':k}]
foldKNN = ms.GridSearchCV(knn.KNeighborsClassifier(),paramitersKNN,cv=5,iid=False)
foldSVM = ms.GridSearchCV(svm.SVC(),paramitersSVM,cv=5,iid=False)

foldKNN.fit(X_train[:,2:4],Y_train)
foldSVM.fit(X_train[:,2:4],Y_train)
KNN = knn.KNeighborsClassifier(foldKNN.best_params_['n_neighbors'])
SVM = svm.SVC(kernel=foldSVM.best_params_['kernel'],C=foldSVM.best_params_['C'],gamma=foldSVM.best_params_['gamma'])
KNN.fit(X_train[:,2:4],Y_train)
SVM.fit(X_train[:,2:4],Y_train)

printKNNBoundaries(X_train[:,2:4],Y_train,KNN,foldKNN.best_params_['n_neighbors'],cmap_light,cmap_bold)
printSVM(X_train[:,2:4],Y_train,SVM,foldSVM.best_params_['C'])






