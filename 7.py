from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset=load_iris()
X=pd.DataFrame(dataset.data,columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
Y=pd.DataFrame(dataset.target,columns=['Targets'])
print(X)
print(Y)

colormap=np.array(['red','lime','black'])

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[Y.Targets],s=40)
plt.title("Real Clustering")
plt.show()

model1=KMeans(n_clusters=3)
model1.fit(X)

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model1.labels_],s=40)
plt.title("KMean Clustering")
plt.show()

model2=GaussianMixture(n_components=3)
model2.fit(X)

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model2.predict(X)],s=40)
plt.title("EM Clustering")
plt.show()

print("Actual Target :\n",dataset.target)
print("KMean :\n",model1.labels_)
print("EM :\n",model2.predict(X))
print("Kmean accuracy :\n",sm.accuracy_score(Y,model1.labels_))
print("EM accuracy :\n",sm.accuracy_score(Y,model2.predict(X)))
