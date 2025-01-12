import time
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
from drugXdisease.dXd_results import write_results_file, write_nn_results_file, plot_graph, do_roc
from drugXdisease.dXd_logging import log_performance
from drugXdisease.dXd_graph_metrics import compute_graph_features, enhanced_features
from drugXdisease.dXd_randomForest import train_classifier_rf
from drugXdisease.dXd_neuralNet import train_classifier_nn
from drugXdisease.dXd_read_input import read_input_data


def generate_node_embeddings(graph, dimensions=128, walk_length=30, num_walks=200, workers=4):
    """
    Generate node embeddings using the Node2Vec algorithm.

    Parameters:
    - graph: networkx.Graph, the input graph.
    - dimensions: int, size of the embedding vectors.
    - walk_length: int, length of each random walk.
    - num_walks: int, number of random walks per node.
    - workers: int, number of parallel workers.

    Returns:
    - embeddings: dict, node embeddings.
    """
    node2vec = Node2Vec(
        graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    return embeddings


def load_embeddings(embeddings_file):
    """
    Load precomputed embeddings from the embeddings file.

    Parameters:
    - embeddings_file: str, path to the embeddings file.

    Returns:
    - embeddings: dict, mapping node ID to its embedding as a numpy array.
    """
    embeddings = {}
    current_id = None
    current_embedding_lines = []

    print(f"Reading embeddings from {embeddings_file}...")
    with open(embeddings_file, 'r') as f:
        next(f)  # Skip the header
        for line in f:
            # Check if the line starts with a new entry ID (integer at the beginning)
            if line.strip() and line.split(',')[0].isdigit():
                # Process the previous entry if one exists
                if current_id is not None:
                    # Join lines and parse the embedding
                    full_embedding = " ".join(current_embedding_lines).strip('"[]')
                    embedding_array = np.array([float(x) for x in full_embedding.split()])
                    embeddings[current_id] = embedding_array
                
                # Start a new entry
                parts = line.strip().split(',')
                current_id = parts[1]  # Node ID
                current_embedding_lines = [parts[-1]]  # First line of embedding
            else:
                # Continue collecting lines for the current embedding
                current_embedding_lines.append(line.strip())

        # Process the last entry
        if current_id is not None:
            full_embedding = " ".join(current_embedding_lines).strip('"[]')
            embedding_array = np.array([float(x) for x in full_embedding.split()])
            embeddings[current_id] = embedding_array

    print(f"Loaded {len(embeddings)} embeddings.")
    return embeddings


def create_dataset(graph, embeddings, graph_features):
    """
    Create a dataset for drug-disease links prediction.

    Parameters:
    - graph: networkx.Graph, the input graph.
    - embeddings: dict, node embeddings.
    - graph_features: dict, graph-based features for each node.

    Returns:
    - X: numpy array, feature matrix.
    - y: numpy array, labels.
    - pairs: list of (drug, disease) pairs.
    """
    positive_pairs = [(u, v) for u, v, d in graph.edges(data=True) if d.get("relation") == "treats"]
    negative_pairs = [(u, v) for u, v, d in graph.edges(data=True) if d.get("relation") == "unaffected"]
    
    print(f"Number of positive pairs (treats): {len(positive_pairs)}")
    print(f"Number of negative pairs (unaffected): {len(negative_pairs)}")

    def pair_features(pair):
        embedding_features = np.concatenate([embeddings[pair[0]], embeddings[pair[1]]])
        graph_features_pair = np.array([
            graph_features[pair[0]]["degree"],
            graph_features[pair[1]]["degree"],
            graph_features[pair[0]]["betweenness"],
            graph_features[pair[1]]["betweenness"],
            graph_features[pair[0]]["clustering"],
            graph_features[pair[1]]["clustering"],
        ])
        enhanced_features_pair = enhanced_features(graph, pair[0], pair[1])
        return np.concatenate([embedding_features, graph_features_pair, enhanced_features_pair])

    X = np.array([pair_features(pair) for pair in positive_pairs + negative_pairs])
    y = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))
    return X, y, positive_pairs + negative_pairs


def create_graph_from_data(nodes, edges):
    """
    Create a NetworkX graph using the filtered nodes and edges.
    """
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node['id'], **node)
    for edge in edges:
        graph.add_edge(edge['subject'], edge['object'], **edge)
    return graph

def evaluate_model(classifier, X_test, y_test, is_nn=False):
    """
    Evaluate the model on the test set and print performance metrics.

    Parameters:
    - classifier: the trained model (can be RF or NN).
    - X_test: numpy array, test feature matrix.
    - y_test: numpy array, true labels.
    - is_nn: bool, whether the classifier is a neural network.
    """
    if is_nn:
        # Predict probabilities and threshold for binary output
        y_pred = (classifier.predict(X_test) > 0.5).astype("int32").flatten()
    else:
        y_pred = classifier.predict(X_test)
    
    target_names = ["unaffected", "treats"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    # Write evaluation report to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_nn:
        with open(f"evaluation_{timestamp}_nn.txt", "w") as eval_file:
            eval_file.write("Accuracy: " + str(accuracy) + "\n")
            eval_file.write("Classification Report:\n" + report)
    else:
        with open(f"evaluation_{timestamp}_rf.txt", "w") as eval_file:
            eval_file.write("Accuracy: " + str(accuracy) + "\n")
            eval_file.write("Classification Report:\n" + report)


def main():
    # File paths
    nodes_file = "data/SNodes.csv"
    edges_file = "data/SEdges.csv"
    ground_truth_file = "data/Ground_Truth.csv"
    embeddings_file = "data/Embeddings.csv"

    # Step 1: Read and process input data
    nodes, edges, ground_truth, graph, embeddings_df = read_input_data(
        nodes_file, edges_file, ground_truth_file, embeddings_file
    )
    print(f"Number of nodes in the graph: {graph.number_of_nodes()}")
    print(f"Number of edges in the graph: {graph.number_of_edges()}")

    # Step 2: Plot and save the graph
    print("Plotting graph...", datetime.now().strftime("%Y%m%d_%H%M%S"))
    plot_graph(graph)

    # Step 3: Compute graph features
    print("Computing graph features...", datetime.now().strftime("%Y%m%d_%H%M%S"))
    graph_features = compute_graph_features(graph)

    # Step 4: Load embeddings
    print("Loading embeddings...", datetime.now().strftime("%Y%m%d_%H%M%S"))
    embeddings = load_embeddings(embeddings_file)

    # Step 5: Create dataset with enhanced features
    print("Creating dataset...", datetime.now().strftime("%Y%m%d_%H%M%S"))
    X, y, pairs = create_dataset(graph, embeddings, graph_features)

    # Step 6: Split the dataset
    X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(
        X, y, pairs, test_size=0.3, random_state=42
    )

    # Step 7: Train classifiers
    print("Training Random Forest...", datetime.now().strftime("%Y%m%d_%H%M%S"))
    classifier_rf = train_classifier_rf(X_train, y_train)

    print("Training Neural Network...", datetime.now().strftime("%Y%m%d_%H%M%S"))
    input_dim = X_train.shape[1]
    classifier_nn = train_classifier_nn(X_train, y_train, input_dim, epochs=20, batch_size=16)

    # Step 8: Evaluate classifiers
    print("Random Forest Evaluation:")
    evaluate_model(classifier_rf, X_test, y_test, is_nn=False)

    print("Neural Network Evaluation:")
    y_prob_nn = classifier_nn.predict(X_test)
    y_pred_nn = (y_prob_nn > 0.5).astype("int32").flatten()
    evaluate_model(classifier_nn, X_test, y_test, is_nn=True)

    # Step 9: Generate and save ROC curve for the neural network
    print("Generating ROC curve for Neural Network...")
    do_roc(y_test, y_prob_nn)

    # Step 10: Write results
    y_pred_rf = classifier_rf.predict(X_test)
    y_prob_rf = classifier_rf.predict_proba(X_test)
    write_results_file(pairs_test, y_test, y_pred_rf, y_prob_rf)
    write_nn_results_file(pairs_test, y_test, y_pred_nn, y_prob_nn)

    # Step 11: Log performance metrics
    start_time = time.time()
    log_performance(time.time() - start_time)


if __name__ == "__main__":
    main()
