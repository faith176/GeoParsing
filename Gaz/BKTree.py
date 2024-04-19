import os
import pickle
import Levenshtein
import networkx as nx
import matplotlib.pyplot as plt
from alive_progress import alive_bar

class BKNode:
    def __init__(self, word, geoID=0):
        self.word = word
        self.geoID = geoID
        self.children = {}

    def add_child(self, child_node, distance):
        self.children[distance] = child_node

class BKTree:
    def __init__(self, values = None, sample=False):
        self.root = None
        if os.path.exists('data/saved_data/Gaz/bktree.pkl') and sample==False:
            print("Retrieving BK Tree from Saved Data")
            self.retrieve_bktree()
        else:
            if values:
                print("Creating BK Tree")
                with alive_bar(len(set(values)), force_tty=True) as bar:
                    for loc in set(values):
                        self.add(loc)
                        bar()
                    if sample == False:
                        self.save_bk_tree()
                        
    def save_bk_tree(self):
        with open('data/saved_data/Gaz/bktree.pkl','wb') as f:
            pickle.dump(self.root, f)
            print("Saved Data")
            
    def retrieve_bktree(self):
        with open('data/saved_data/Gaz/bktree.pkl','rb') as f:
            loaded = pickle.load(f)
            self.root = loaded

    def add(self, loc) -> None:
        if self.root is None:
            self.root = BKNode(loc[0], loc[1])
        else:
            self._add_to_node(self.root, loc)

    def _add_to_node(self, node, loc):
        distance = Levenshtein.distance(loc[0], node.word)

        if distance in node.children:
            self._add_to_node(node.children[distance], loc)
        else:
            node.add_child(BKNode(loc[0], loc[1]), distance)

    def search(self, query, max_distance) -> list:
        if self.root is None:
            return []

        result = []

        self._search_in_node(self.root, query.lower(), max_distance, result)
        result.sort(key=lambda x: x[1], reverse=True) # Sort the final result based on highest to lowest score

        return result

    def _search_in_node(self, node, query, max_distance, result):
        distance = Levenshtein.distance(query, node.word)
        score = Levenshtein.ratio(query, node.word)

        if distance <= max_distance:
            result.append((node.word, score, distance, node))

        for i in range(distance - max_distance, distance + max_distance + 1):
            if i in node.children:
                self._search_in_node(node.children[i], query, max_distance, result)
    

    def graph_10_nodes(self):
        G = nx.Graph()
        nodes = []

        def traverse(node):
            if len(nodes) >= 10:
                return
            if node:
                nodes.append(node)
                for child_distance, child_node in sorted(node.children.items()):
                    traverse(child_node)

        traverse(self.root)

        closest_nodes = sorted(nodes, key=lambda x: Levenshtein.distance(x.word, self.root.word))[:10]
        
        for i, node in enumerate(closest_nodes):
            G.add_node(i, label=node.word)  # Use node.word as label

        for i, node in enumerate(closest_nodes):
            for child_distance, child_node in node.children.items():
                if child_node in closest_nodes:  # Check if child is in closest_nodes list
                    j = closest_nodes.index(child_node)
                    G.add_edge(i, j, distance=child_distance)  # Add Levenshtein distance as edge attribute

        labels = {i: data["label"] for i, data in G.nodes(data=True)}  # Extract labels from node data
        edge_labels = nx.get_edge_attributes(G, 'distance')  # Extract distances
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color="skyblue", font_size=10, arrowsize=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')  # Draw edge labels
        plt.title("Location Entities String Similarity Using Levenshtein Distance", fontsize=14)
        plt.show()

def main():
    bktree = BKTree(sorted([("book", 0), ("books",0),("bookd", 0), ("cake",0), ("boo",0), ("cape",0), ("boon",0), ("cook",0), ("cart",0)]), sample=True)
    query_word = "book"
    max_distance = 1
    result = bktree.search(query_word, max_distance)
    print(f"Words within {max_distance} distance from '{query_word}': {result}")
    bktree.graph_10_nodes()
    
if __name__ == "__main__":
    main()