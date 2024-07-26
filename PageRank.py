"""
Auburn University -- Data Mining -- COMP 5130

Group 7:
Bobby Bevis, Ben Buchanan, Jonghyun Jung

PageRank Implementation
"""
import pandas as pd
import networkx as nx

MAX_ITERATIONS = 100
DAMPING_FACTOR = 0.85
CONVERGENCE_THRESHOLD = 1e-6

# Bonus dataset (13.9 MB) 
bonusFilePath = "Dataset/com-dblp.ungraph.txt"

# Dataset that will be used 352 KB
filePath = "Dataset/CA-GrQc.txt"

# Create a dataframe from the given data using the pandas library
# To change the dataset used, change "filePath" to "bonusFilePath"
df = pd.read_csv(filePath, delimiter='\t', comment='#', names=['FromNodeId', 'ToNodeId'])

# Generate a directed graph using the networkx library
G = nx.from_pandas_edgelist(df, "FromNodeId", "ToNodeId", create_using=nx.DiGraph)

# Intialize all nodes to have the same pagerank
pageRank = {node: 1/len(G) for node in G.nodes}

# For each node (page) calculate its specific page rank based on the the nodes and edges of the graph.
for iteration in range(MAX_ITERATIONS):
    new_pageRank = {}
    for node in G.nodes:
        rank_sum = sum(pageRank[neighbor] / len(list(G.neighbors(neighbor))) for neighbor in G.predecessors(node))
        new_pageRank[node] = (1 - DAMPING_FACTOR) / len(G) + DAMPING_FACTOR * rank_sum
    # During every iteration, check if the ranks have surpassed the convergence threshold.
    if max(abs(new_pageRank[node] - pageRank[node]) for node in G.nodes) < CONVERGENCE_THRESHOLD:
        print(f"\n*** PageRank has surpassed the convergence threshold at iteration: {iteration} ***")
        break
    # Update the Pagerank value for the next iteration.
    pageRank = new_pageRank

# Normalization of PageRank Values sum to 100%
rank_sum = sum(pageRank.values())
pageRank = {node: rank / rank_sum for node, rank in pageRank.items()}

# Ensure the sum of the rank values == 100%
# PageRank values should sum to 1, so we multiply by 100 to get the percentage
# Rounded the number up, since the number will equal 9.99 repeating
totalRank = (100 * sum(pageRank.values()))
if (totalRank != 100):
    print(f"The total rank does NOT equal 100. The actual total rank is: {totalRank}%\n")
else:
    print(f"The rank sum has the correct value of: {totalRank}%\n")
    
# Prints the first ten values of the pageRank dictionary
print("Here are the first 10 values:\n")
print("Node | Rank")
print("------------------------------")
for i, (key, value) in enumerate(pageRank.items()):
    if i >= 10:
        break
    print(key, value)


# print(pageRank)
