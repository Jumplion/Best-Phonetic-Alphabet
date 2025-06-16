import os
import csv
import logging
import string
from collections import defaultdict, Counter

# Scientific and Data Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Plotting and Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Graph Libraries
import networkx as nx

# Machine Learning and Clustering
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram

WORD_PAIR_FILENAME_TEMPLATE:str = "word_pairs_{0}_{1}_data.csv"
LETTER_PAIR_FILENAME:str = "letter_pair_averages.csv"

# -----------------------------
# Graphing Options
# -----------------------------
SPRING_FACTOR:float = 0.5    # Factor to control the spring force in the graph layout
ITERATIONS:int = 50         # Number of iterations for the spring layout algorithm
MAX_DISTANCE:float = 3.0        # Maximum distance between nodes in the graph layout
# -----------------------------

"""_summary_
Log a header message to the console
"""
def log_console_header(message, data=None):
    logging.info("--------------------------------")
    if data:
        logging.info("%s: %s", message, data)
    else:
        logging.info(message)
    logging.info("--------------------------------")

# -----------------------------
# Visualization Functions
# -----------------------------

def plot_graph(spring_factor = 0.15, iters = 50, max_distance = 2.0):
    
    # Build the graph with networkx for easy Plotly integration
    G = nx.Graph()
    base_dir = os.path.join("Gephi Files")
    
    logging.info("Building Graph from CSV Files in '%s'", base_dir)
    logging.info("Spring Factor: %.2f | Iterations: %i | Score Range: %.2f", spring_factor, iters, max_distance)
    
    words = set()
    letters = string.ascii_uppercase
    
    for l1 in tqdm(letters, total=26, desc="Reading Word Distances", unit=" letter", colour="green"):
        
        if G.number_of_nodes() > 10000:
            break
        
        l1_index = letters.index(l1)
        check_list = letters[l1_index:]  # Only check letters after the current letter to avoid duplicates

        for l2 in tqdm(check_list, total=len(check_list), desc=f"- Reading {l1}-Word Distances | {len(G.nodes)} Nodes | {len(G.edges)} Edges", unit=" letter", leave=False, colour="yellow"):
            filename = os.path.join(base_dir, WORD_PAIR_FILENAME_TEMPLATE.format(l1, l2))
            edges = defaultdict(list)

            # Sort the csv file by distance before processing
            with open(filename, newline="") as f:
                reader = csv.reader(f)
                next(reader) # Skip header

                # Process the sorted rows
                for w1, w2, dist in tqdm(reader, desc=f"Processing {l1}-{l2} Distances", unit=" word pair", leave=False, colour="red"):
                    if w1 != w2 and np.abs(float(dist)) <= max_distance:
                        words.add(w1)
                        words.add(w2)
                        edges[dist].append((w1, w2))
                    
                    if len(words) > 10000:
                        break
            
                # Add nodes and edges to the graph
                # Doing this in the for loop so we can see progress and avoid memory issues with large graphs
                G.add_nodes_from(tqdm(words, total=len(words), desc="Adding Nodes", unit="word", leave=False))
                for d in tqdm(edges.keys(), total=len(edges), desc="Processing Edges", unit="distance", leave=False):
                    G.add_edges_from(edges[d], weight=float(d))
    
    logging.info("Total Words: %d | Total Edges: %d", G.number_of_nodes(), G.number_of_edges())

    # Assign colors by first letter
    letters = sorted(set(w[0].upper() for w in words))
    letter_to_color = {le: f"hsl({int(360*i/len(letters))},70%,50%)" for i, le in enumerate(letters)}
    node_colors = [letter_to_color[w[0].upper()] for w in G.nodes()]
    
    log_console_header("Calculating Node Positions for Scatter Plot (This will take a while...)")
    logging.info("Using spring layout with factor: %.2f and iterations: %d", spring_factor, iters)
    # Use spring layout for positions
    pos = nx.spring_layout(G, k=spring_factor, iterations=iters)

    # Build edge traces
    edge_x = edge_y = []
    for e in tqdm(G.edges(), total=len(G.edges()), desc="Building Edge Traces", unit="edge"):
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    log_console_header("Calculating Edge Traces for Scatter Plot (This will take a minute...)")
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="rgb(50, 50, 50)"),
        hoverinfo='none',
        mode='lines'
    )

    # Build node traces
    node_x = []
    node_y = []
    node_text = []
    for node in tqdm(G.nodes(), total=len(G.nodes()), desc="Building Node Traces", unit="node"):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=8,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Cluster Graph: Iterations: {iters}, Score Range: {max_distance}',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    #fig.write_image(f"cluster_graph_iter_{iters}_score_range_{max_distance}.png", width=1200, height=800, scale=2)
    fig.write_html(f"cluster_graph_iter_{iters}_score_range_{max_distance}.html")
    fig.show()

"""_summary_
Plots 4 bar graphs:
1. Number of words in each letter group (A-Z)
2. Number of words for each word length
3. Number of words for each phoneme length
4. Number of words for each syllable count
"""
def plot_dictionary_stats(words_by_letter, words_by_length, words_by_phoneme_length, words_by_syllable):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 1. Words per letter group
    letters = sorted(words_by_letter.keys())
    counts = [len(words_by_letter[le]) for le in letters]
    bars = axs[0, 0].bar(letters, counts, color='tab:blue')
    axs[0, 0].set_title("Number of Words by First Letter")
    axs[0, 0].set_xlabel("First Letter")
    axs[0, 0].set_ylabel("Word Count")
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        axs[0, 0].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # 2. Words per word length
    lengths = sorted(words_by_length.keys())
    length_counts = [len(words_by_length[le]) for le in lengths]
    bars = axs[0, 1].bar(lengths, length_counts, color='tab:orange')
    axs[0, 1].set_title("Number of Words by Word Length")
    axs[0, 1].set_xlabel("Word Length")
    axs[0, 1].set_ylabel("Word Count")
    for bar in bars:
        height = bar.get_height()
        axs[0, 1].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # 3. Words per phoneme length
    phoneme_lengths = sorted(words_by_phoneme_length.keys())
    phoneme_counts = [len(words_by_phoneme_length[le]) for le in phoneme_lengths]
    bars = axs[1, 0].bar(phoneme_lengths, phoneme_counts, color='tab:green')
    axs[1, 0].set_title("Number of Words by Phoneme Length")
    axs[1, 0].set_xlabel("Phoneme Length")
    axs[1, 0].set_ylabel("Word Count")
    for bar in bars:
        height = bar.get_height()
        axs[1, 0].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # 4. Words per syllable count
    syllable_counts = sorted(words_by_syllable.keys())
    syllable_word_counts = [len(words_by_syllable[s]) for s in syllable_counts]
    bars = axs[1, 1].bar(syllable_counts, syllable_word_counts, color='tab:red')
    axs[1, 1].set_title("Number of Words by Syllable Count")
    axs[1, 1].set_xlabel("Syllable Count")
    axs[1, 1].set_ylabel("Word Count")
    for bar in bars:
        height = bar.get_height()
        axs[1, 1].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    plt.suptitle("Dictionary Statistics", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("dictionary_stats.png")
    plt.show()  

def visualize_word_length_distribution(words_by_letter):
    """
    Visualize the distribution of word lengths for each letter group in words_by_letter.
    Shows both a heatmap and a grouped bar chart.
    """
    # Count word lengths for each letter
    length_counts_by_letter = {}
    for letter, words in words_by_letter.items():
        lengths = [len(word) for word in words]
        length_counts_by_letter[letter] = Counter(lengths)

    # Find all unique word lengths
    all_lengths = set()
    for counts in length_counts_by_letter.values():
        all_lengths.update(counts.keys())
    all_lengths = sorted(all_lengths)

    # Build DataFrame: rows=word lengths, columns=letters
    df = pd.DataFrame(
        {letter: [length_counts_by_letter[letter].get(le, 0) for le in all_lengths] for letter in words_by_letter},
        index=all_lengths
    )
    df.index.name = "Word Length"

    # Plot heatmap
    plt.figure(figsize=(16, 6))
    sns.heatmap(df.T, cmap="Blues", annot=True, fmt="d")
    plt.title("Word Length Distribution by Letter (Heatmap)")
    plt.xlabel("Word Length")
    plt.ylabel("Starting Letter")
    plt.tight_layout()
    plt.show()

    # Plot grouped bar chart
    df.T.plot(kind="bar", stacked=False, figsize=(16, 6))
    plt.title("Word Length Distribution by Letter (Bar Chart)")
    plt.xlabel("Starting Letter")
    plt.ylabel("Count")
    plt.legend(title="Word Length")
    plt.tight_layout()
    plt.show()

def plot_phoneme_distance_matrix(matrix, labels, title="Phoneme Difference Matrix"):
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="magma", square=True)
    plt.title(title)
    plt.xlabel("Phoneme")
    plt.ylabel("Phoneme")
    plt.tight_layout()
    plt.show()

def plot_phoneme_dendrogram(matrix, labels):
    linkage_matrix = linkage(matrix, method='average')
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title("Phoneme Clustering Dendrogram")
    plt.ylabel("Acoustic Distance")
    plt.tight_layout()
    plt.show()

def plot_mds(matrix, labels, dim=2):
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(matrix)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = coords[i][:2]
        plt.scatter(x, y)
        plt.text(x, y, label, fontsize=10)
    plt.title("MDS Projection of Phonemes")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid()
    plt.tight_layout()
    plt.show()
