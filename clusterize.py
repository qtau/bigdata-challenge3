from adjusted_rand_index import rand_index
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt


# Map a 32-bit hexadecimal hashing into a vector of length 32 of integers from 0 to 15
def hashToAttributes(hashing):
    dicMapping = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}
    result = []
    for c in hashing:
        result.append(dicMapping[c])
    return result

	
# Create the inputs for the 9700 videos: each observation is a 32-element vector of integers
def createInput(hash_videos):
    videos_indice = []
    inputX = []
    for key in hash_videos:
        videos_indice.append(key)
        inputX.append(hashToAttributes(hash_videos[key]))
    return videos_indice, inputX

	
# Returns the results of the clustering for each video into a variable result that can be used by the function rand_index
def createClusterList(clusterPredict, videos_indice):
    result = [None]*970
    for i, c in enumerate(clusterPredict):
        if result[c] is not None:
            result[c].add(videos_indice[i])
        else:
            result[c] = set([videos_indice[i]])
    return result
	

# Clustering algorithm using K-means
def clusterize_kmeans(videos_hash, nb_clusters):
    videos_indice, X = createInput(videos_hash)
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0, n_init=5, max_iter=300).fit(X)
    listCluster = createClusterList(kmeans.predict(X), videos_indice)
    ri = rand_index(listCluster)
    print(ri)
    return ri, videos_indice, kmeans, listCluster
	
	
# Clustering algorithm using agglomerative clustering with ward linkage
def clusterize_aggroCluster(videos_hash, nb_clusters):
    videos_indice, X = createInput(videos_hash)
    aggroCluster = AgglomerativeClustering(n_clusters=nb_clusters, linkage='ward', affinity='euclidean').fit(X)
    listCluster = createClusterList(aggroCluster.fit_predict(X), videos_indice)
    ri = rand_index(listCluster)
    print(ri)
    return ri, videos_indice, aggroCluster, listCluster
	
	
	
##### MAIN #####

# Open the file containing all the hashing for all the videos
with open("videos_hash.json", "r") as file:
    videos_hash = json.load(file)

# clusterize the videos	
ri, videos_indice, aggroCluster, listCluster = clusterize_aggroCluster(videos_hash,970)
# it prints 0.65


# Plot the distribution of the numbers of videos in each cluster
plt.hist([len(listCluster[i]) for i in range(len(listCluster))], bins=50)
plt.xlabel("Number of videos per cluster")
plt.ylabel("Number of clusters")
plt.savefig("distribution_cluster.png")
plt.show()