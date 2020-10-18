import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('customers.csv')
X = dataset.iloc[:, [3,4,5]].values
#X = dataset.iloc[:, 3:6].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Scaled = sc_X.fit_transform(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plt.plot(range(1, 11), wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()

def segment(cluster_number):
    if cluster_number == 0:
        return "Fan"
    elif cluster_number == 1:
        return "Roamer"
    elif cluster_number == 2:
        return "Supporter"
    else :
        return "Alienated"

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_Scaled)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'red', label = 'Cluster 0')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'cyan', label = 'Cluster 3')


#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Customer Clusters')
plt.xlabel('Satisfaction')
plt.ylabel('Service Spending')
#plt.legend()
plt.show()

#Roamer
prediction_data = [[5,450,4]]

prediction_data_scaled = sc_X.transform(prediction_data)
prediction = kmeans.predict(prediction_data_scaled)
print(segment(prediction[0]))

#3D Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s =5, c = 'red')
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s =5, c = 'blue')
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s =5, c = 'green')
ax.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], X[y_kmeans == 3, 2], s =5, c = 'cyan')

ax.set_xlabel('Satisfaction')
ax.set_ylabel('Spend')
ax.set_zlabel('Visits')
plt.show()




import pickle

pickle_out = open("customer_kmeans_segmentation.pickle","wb")
pickle.dump(kmeans, pickle_out)
pickle_out.close()

pickle_out = open("customer_kmeans_scaler.pickle","wb")
pickle.dump(sc_X, pickle_out)
pickle_out.close()












