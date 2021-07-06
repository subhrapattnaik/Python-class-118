#How to choose the right K?

#We use the WCSS perimeter to to evaluate the choice of K. WCSS stands for Within Cluster Sum of Squares. What this means is that we are going to choose a center point for a cluster, from where all the points falling inside that cluster will be closest.

#Then, we will calculate the distance of all the points from the center, add up all the distances and then note the value.

#We will then take 2 centre points and do the same. We will choose the value of K to be the one which has the minimum sum of all the distances.


#The Elbow method can be used to choose the best value for K. Let's see how it works!


#Here, we are going to take up a data of some of the flowers, and we want to cluster them to know how many species of flower's data do we have.

#-------------------------------------------------------------

import pandas as pd
import plotly.express as px

df = pd.read_csv("./118/petals_sepals.csv")

print(df.head())

fig = px.scatter(df, x="petal_size", y="sepal_size")
fig.show()

#-------------------------------------------------
#Now, let's find the best K value by using the WCSS perimeter and the Elbow method.

from sklearn.cluster import KMeans

X = df.iloc[:, [0,1]].values

print(X)

wcss = []
#we just need 10 cluster points so taken till 11
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
#--------------------------------------------------------------------------
#

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#----------------------------------------------
#Now, if we look at the scatterplot we plotted earlier, we can see that their might be around 3 clusters.


#In the elbow chart above, we can see that the WCSS value is decreasing significantly until the K = 3. Hence, we can see that our K is 3 for the data given.
#------------------------------------------------
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
#----------------------------------------------------
plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids',s=100,marker=',')
plt.grid(False)
plt.title('Clusters of Flowers')
plt.xlabel('Petal Size')
plt.ylabel('Sepal Size')
plt.legend()
plt.show()
#----------------------------------------------------
#Thus, we can see that our model has identified 3 clusters, which means that we had data for 3 different species of flowers.

