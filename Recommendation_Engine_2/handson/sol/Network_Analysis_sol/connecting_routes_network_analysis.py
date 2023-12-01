# import libraries
import pandas as pd
import networkx as nx

# Load the dataset
column_names = ["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]
Airport = pd.read_csv("D:/360digi/DS/Sharath/Recommendation_Engine_2/handson/Datasets_Network Analytics/flight_hault.csv", names = column_names, header= None)
Airport.info()


column_names = ["flights","ID","main_Airport", "main_AirportID","Destination","DestinationID","Dummy","haults","machinary"]
R = pd.read_csv("D:/360digi/DS/Sharath/Recommendation_Engine_2/handson/Datasets_Network Analytics/connecting_routes.csv", names= column_names, header = None)

R.drop(["Dummy"], axis = 1, inplace= True)
R1 = R.iloc[0:500, 1:]
R1.info()

#checking for missing values
R1.isnull().sum()

g = nx.Graph()
g = nx.from_pandas_edgelist(R1, source = 'main_Airport', target = 'Destination')

print(nx.info(g)) #Graph with 227 nodes and 264 edges

#Degree Centrality
d = nx.degree_centrality(g)
print(d)

pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')

#Closeness Centrality
closeness = nx.closeness_centrality(g)
print(closeness)

#Betweenness Centrality
b = nx.betweenness_centrality(g)
print(b)

m = max(b, key=b.get)
res = Airport.loc[Airport['IATA_FAA'] == m]
Airport.iloc[res.index,:]


#Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g)
print(evg)
max(evg)

#'ZRH'
res = Airport.loc[Airport['IATA_FAA'] == 'ZRH']
print(res.index)
print(Airport.iloc[res.index,:])


#Cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

#Average clustering
cc = nx.average_clustering(g)
print(cc)
