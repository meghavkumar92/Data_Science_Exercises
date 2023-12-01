# import libraries
import pandas as pd
import networkx as nx
import numpy as np

fb = pd.read_csv("D:/360digi/DS/Sharath/Recommendation_Engine_2/handson/Datasets_Network Analytics/facebook.csv")
fb.columns = [k for k in np.arange(9)]
fb.info()

insta = pd.read_csv("D:/360digi/DS/Sharath/Recommendation_Engine_2/handson/Datasets_Network Analytics/instagram.csv")
insta.columns = [p for p in np.arange(8)]
insta.info()

LIn = pd.read_csv("D:/360digi/DS/Sharath/Recommendation_Engine_2/handson/Datasets_Network Analytics/linkedin.csv")
LIn.columns = [q for q in np.arange(13)]
LIn.info()

# undirected graph
g = nx.Graph()
g = nx.from_pandas_adjacency(fb)
print(nx.info(g))
# Graph with 9 nodes and 9 edges


# Visualization of Facebook data in circular network
nx.draw_circular(g, node_size = 300, node_color = 'red', with_labels=True)
# or
nx.draw_networkx(g, pos= nx.circular_layout(g), node_size = 15, node_color = 'red' )



g1 = nx.from_pandas_adjacency(insta)
print(nx.info(g1))
# Graph with 8 nodes and 7 edges

# Visualization of Instagram data in star network
G = nx.star_graph(g1)
nx.draw(G, node_color = 'orange', node_size = 300, with_labels=True)




l1 = nx.from_pandas_adjacency(LIn)
print(nx.info(l1))
# Graph with 13 nodes and 18 edges

# Visualization of LinkedIn data in star network
G = nx.star_graph(l1)
nx.draw(G, node_color = 'grey', node_size = 300, with_labels=True)






