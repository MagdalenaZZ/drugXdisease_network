```markdown
# Drug x Disease Interaction Prediction Package

## Create Conda Environment

To set up the environment:

```bash
conda env create -f bio-graph-env.yaml
conda activate bio-graph-env
```

> **Note**: For production, a Docker image might be a better choice for consistency across systems, but the Conda environment should work reasonably well on most setups.

## Running the Script

The script `load_and_Represent_graph.py` can be executed with:

```bash
python load_and_Represent_graph.py
```

## Output Files

### General Performance Metrics
- Random Forest performance metrics: `results/evaluation_20250112_171908_rf.txt`
- Neural Network performance metrics: `results/evaluation_20250112_171909_nn.txt`
- ROC curve overview: `results/roc_curve_20250112_171909.png`
- Computational performance metrics: `results/performance_20250112_171910.txt`

### Quality Check
- Graph image overview (for input nodes and edges): `results/graph_20250112_171602.png`

### Results
- Random Forest predictions: `results/results_20250112_171910.tsv`
- Neural Network predictions: `results/results_nn_20250112_171910.txt`

## Testing

To run tests:

```bash
python -m unittest discover tests
```

> **Note**: This is a simple testing framework designed as a marker rather than a comprehensive structure.

## Main Script

The script `load_and_Represent_graph.py` performs the following tasks:

1. **Generate Embeddings**:
   - Function: `generate_node_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)`

2. **Load Embeddings**:
   - Function: `load_embeddings(embeddings_file)`

3. **Divide Data**:
   - Split embeddings into training and testing sets with `create_dataset(graph, embeddings)` (70% training, 30% testing).

4. **Train Classifier**:
   - Function: `train_classifier(X_train, y_train)`
   - Methods used:
     - Random Forest (via `sklearn` for speed and effectiveness).
     - Neural Network.

5. **Graph Creation**:
   - Function: `create_graph_from_data(nodes, edges)`

6. **Model Evaluation**:
   - Function: `evaluate_model(classifier, X_test, y_test)`
   - Evaluates precision, recall, and F1-score.

7. **Performance Logging**:
   - Function: `log_performance(start_time)`
   - Compares computational costs across approaches.

## Package Structure

The package includes the following modules:

- **Core Module**:
  - `drugXdisease/__init__.py`

- **Graph Handling**:
  - `drugXdisease/dXd_graph.py`
  - `drugXdisease/dXd_graph_metrics.py`

- **Logging**:
  - `drugXdisease/dXd_logging.py`

- **Data Handling**:
  - `drugXdisease/dXd_read_input.py`
  - `drugXdisease/dXd_results.py`

- **Learning Approaches**:
  - Logistic Regression: `drugXdisease/dXd_logisticRegression.py`
  - Neural Network: `drugXdisease/dXd_neuralNet.py`
  - Random Forest: `drugXdisease/dXd_randomForest.py`

## Conclusions

### Feature Engineering
- Added graph-based metrics to embeddings:
  - **Shortest Path Length**: Between drug and disease nodes.
  - **Number of Paths**: Connections between drug and disease nodes.
  - **Shared Genes**: Genes linking a drug to a disease.
  - **Node Degree**: Number of connections for high-degree diseases and genes.
  - **Betweenness Centrality**: Key regulators in the graph.
  - **Clustering Coefficient**: Local connectivity of nodes.

### Embeddings
- Improved embeddings by increasing dimensionality and richness of the knowledge graph.

### Data Choices
- Gradually expanded the graph with additional nodes, edges, weights, and embeddings for complexity.

### Test Data
- Used slices of real-world data for training and testing, due to the complexity and bias in synthetic data.

## Improvements

- Add CPU and system performance metrics for compute cluster integration.
- Enhance graph visualization with more effective 3D representations.
- Shift evaluation focus to prioritize actionable drug-disease associations over overall accuracy.
- Explore more classification approaches to improve predictions.
- Implement stability testing to ensure reproducibility of associations across runs.
- Investigate feature importance to understand key contributors to predictions.
- Validate new drug-disease associations with orthogonal evidence (e.g., literature, experiments).

## Some Results

### Example Predictions

1. **Hyperpigmentation (MONDO:0019290)**:
   - Potential treatment: Azelaic Acid (CHEMBL1238).
   - Supporting Evidence: Azelaic Acid has shown effects on pigmentation, useful for darker-skinned patients with post-inflammatory pigmentation or melasma.

2. **Cystic Fibrosis (MONDO:0009061)**:
   - Prediction: Chlorhexidine Gluconate (CHEMBL4297088).
   - Verdict: Nonsense result due to irrelevance of the drug for systemic CF treatment.

3. **Coronary Aneurysm (MONDO:0006714)**:
   - Prediction: Urokinase (CHEMBL1201420).
   - Assessment: Limited utility, as Urokinase is effective for clots but less so for other aneurysm causes.

4. **Lichen Planus (MONDO:0006572)**:
   - Prediction: Alefacept (CHEMBL1201571).
   - Supporting Evidence: Prior studies showed some efficacy, with authors recommending further research.

### Suggested Improvements
- Weigh results against broad-spectrum drugs and diseases to create a more targeted list.



