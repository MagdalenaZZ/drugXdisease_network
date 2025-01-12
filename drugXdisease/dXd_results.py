import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

def write_results_file(test_pairs, y_test, y_pred, y_prob):
    """
    Write the results of predictions to a tab-delimited file, sorted by probability.

    Parameters:
    - test_pairs: list of (drug, disease) pairs in the test set.
    - y_test: numpy array, true labels for the test set.
    - y_pred: numpy array, predicted labels for the test set.
    - y_prob: numpy array, predicted probabilities for the test set.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{timestamp}.tsv"

    # Combine the data into a list of tuples and sort by probability
    results = [
        (drug, disease, y_test[i], y_pred[i], y_prob[i][1])
        for i, (drug, disease) in enumerate(test_pairs)
    ]
    results.sort(key=lambda x: x[4], reverse=True)  # Sort by probability (descending)

    # Write to file
    with open(results_file, "w") as f:
        f.write("Drug\tDisease\tActual\tPredicted\tProbability\n")
        for drug, disease, actual, predicted, probability in results:
            f.write(f"{drug}\t{disease}\t{actual}\t{predicted}\t{probability:.4f}\n")

    print(f"Results written to {results_file}")

def write_nn_results_file(pairs, y_test, y_pred, y_prob):
    """
    Write the results of neural network predictions to a text file, sorted by highest probability.

    Parameters:
    - pairs: list of (drug, disease) pairs.
    - y_test: numpy array, true labels.
    - y_pred: numpy array, predicted binary labels.
    - y_prob: numpy array, predicted probabilities.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_nn_{timestamp}.txt"

    # Combine all data into a list of tuples for sorting
    results = [
        (drug, disease, y_test[i], y_pred[i], y_prob[i][0])
        for i, (drug, disease) in enumerate(pairs)
    ]
    
    # Sort by probability in descending order
    results_sorted = sorted(results, key=lambda x: x[4], reverse=True)

    # Write sorted results to file
    with open(results_file, "w") as f:
        f.write("Drug\tDisease\tActual\tPredicted\tProbability\n")
        for drug, disease, actual, predicted, prob in results_sorted:
            f.write(f"{drug}\t{disease}\t{actual}\t{predicted}\t{prob:.4f}\n")

    print(f"Neural network results written to {results_file}")


def do_roc(y_test, y_prob_nn):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_nn)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    roc_file = f"roc_curve_{timestamp}.png"
    plt.savefig(roc_file, dpi=300)
    plt.close()
    print(f"ROC curve saved to {roc_file}")

def plot_graph(graph):
    """
    Generate a visualization of the graph with nodes and edges colored by their types.
    Edges from ground truth are highlighted distinctly, while others share a common color.
    """
    pos = nx.spring_layout(graph, seed=42)  # Generate layout for consistent plotting

    # Separate nodes by type
    node_types = nx.get_node_attributes(graph, 'type')
    drugs = [node for node, node_type in node_types.items() if node_type in ["drug", "biolink:SmallMolecule", "biolink:Drug"]]
    diseases = [node for node, node_type in node_types.items() if node_type in ["disease", "biolink:Disease"]]
    genes = [node for node, node_type in node_types.items() if node_type in ["gene", "biolink:Gene"]]
    others = [node for node in graph.nodes() if node not in drugs + diseases + genes]

    # Draw nodes with distinct colors
    nx.draw_networkx_nodes(graph, pos, nodelist=drugs, node_color="blue", label="Drugs", node_size=10)
    nx.draw_networkx_nodes(graph, pos, nodelist=diseases, node_color="green", label="Diseases", node_size=10)
    nx.draw_networkx_nodes(graph, pos, nodelist=genes, node_color="purple", label="Genes", node_size=10)
    nx.draw_networkx_nodes(graph, pos, nodelist=others, node_color="gray", label="Others", node_size=5)

    # Separate edges by relation type
    treats_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("relation") == "treats"]
    unaffected_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("relation") == "unaffected"]
    other_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("predicate") not in ["treats", "unaffected"]]

    # Draw edges with distinct colors
    nx.draw_networkx_edges(graph, pos, edgelist=treats_edges, edge_color="red", label="Treats", width=1)
    nx.draw_networkx_edges(graph, pos, edgelist=unaffected_edges, edge_color="black", label="Unaffected", width=1)
    nx.draw_networkx_edges(graph, pos, edgelist=other_edges, edge_color="orange", label="Other Relations", width=0.5)

    # Add legend and save the plot
    plt.legend(loc="upper right", fontsize="small")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.title("Drug x Disease overview graph")
    plt.savefig(f"graph_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.clf()

