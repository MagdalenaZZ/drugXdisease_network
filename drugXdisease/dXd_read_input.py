import pandas as pd
import networkx as nx

def read_input_files(edges_file, nodes_file, ground_truth_file, embeddings_file=None):
    """
    Read input files into pandas DataFrames.
    """
    edges_df = pd.read_csv(edges_file)
    nodes_df = pd.read_csv(nodes_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
    embeddings_df = pd.read_csv(embeddings_file) if embeddings_file else None
    return edges_df, nodes_df, ground_truth_df, embeddings_df


def process_nodes(nodes_df, ground_truth_df):
    """
    Filter nodes to include only those relevant to the ground truth.
    """
    #print(nodes_df)
    filtered_nodes = nodes_df[nodes_df['id'].isin(ground_truth_df['source']) |
                              nodes_df['id'].isin(ground_truth_df['target'])]
    #print(filtered_nodes)
    return filtered_nodes


def filter_edges(edges_df, ground_truth_df):
    """
    Filter edges to include only those relevant to the ground truth.
    """
    filtered_edges = edges_df[edges_df['subject'].isin(ground_truth_df['source']) &
                              edges_df['object'].isin(ground_truth_df['target'])]
    return filtered_edges


def process_ground_truth(ground_truth_df):
    """
    Process the ground truth DataFrame.
    Add a 'relation' column based on the 'y' column.
    """
    ground_truth_df = ground_truth_df.copy()
    ground_truth_df['relation'] = ground_truth_df['y'].apply(lambda x: 'treats' if x == 1 else 'unaffected')
    #print(ground_truth_df)
    return ground_truth_df


def process_edges(edges_df, ground_truth_df):
    """
    Combine ground truth edges with existing edges.
    """
    # Prepare ground truth edges with the required attributes
    ground_truth_edges = ground_truth_df[['source', 'target', 'relation']].rename(
        columns={'source': 'subject', 'target': 'object'}
    )

    # Concatenate with the edges file if provided, otherwise use only ground truth
    if not edges_df.empty:
        combined_edges = pd.concat([edges_df, ground_truth_edges], ignore_index=True)
    else:
        combined_edges = ground_truth_edges

    return combined_edges


def build_graph(nodes_df, edges_df):
    """
    Build a NetworkX graph from the nodes and edges DataFrames.
    """
    graph = nx.Graph()

    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        graph.add_node(row['id'], type=row['category'], **row.to_dict())

    # Add edges with attributes
    for _, row in edges_df.iterrows():
        graph.add_edge(row['subject'], row['object'], **row.to_dict())

    return graph


def read_input_data(nodes_file, edges_file, ground_truth_file, embeddings_file=None):
    """
    Complete pipeline for reading and processing input files.
    Returns processed nodes, edges, ground truth, and graph.
    """
    # Step 1: Read files
    edges_df, nodes_df, ground_truth_df, embeddings_df = read_input_files(
        edges_file, nodes_file, ground_truth_file, embeddings_file
    )

    # Step 2: Process ground truth
    processed_ground_truth = process_ground_truth(ground_truth_df)

    # Step 3: Process nodes and edges
    filtered_nodes = process_nodes(nodes_df, processed_ground_truth)
    fewer_edges = filter_edges(edges_df, processed_ground_truth)
    filtered_edges = process_edges(fewer_edges, processed_ground_truth)
    print(f"Number of 'treats' edges: {len(filtered_edges[filtered_edges['relation'] == 'treats'])}")
    print(f"Number of 'unaffected' edges: {len(filtered_edges[filtered_edges['relation'] == 'unaffected'])}")
    
    # Step 4: Build graph
    graph = build_graph(filtered_nodes, filtered_edges)

    return filtered_nodes, filtered_edges, processed_ground_truth, graph, embeddings_df

