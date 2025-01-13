
## Drug x Disease Interaction Prediction Package
A package for predicting some new drug x disease interactions


## Create Conda Environment

To set up the environment:

```bash
conda env create -f bio-graph-env.yaml
```
```bash
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

The script `load_and_Represent_graph.py` performs the following functions:

1. **Generate Embeddings**:
   - Function: `generate_node_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)`
   - Generate embeddings from the sample graph

2. **Load Embeddings**:
   - Function: `load_embeddings(embeddings_file)`
   - Loading the pre-computed embeddings

3. **Divide Data**:
   - Function: `create_dataset(graph, embeddings)`
   - Split embeddings into training and testing sets with (70% training, 30% testing).

4. **Train Classifier**:
   - Function: `train_classifier(X_train, y_train)`
   - Methods used:
     - Random Forest (via `sklearn` for speed and effectiveness).
     - Neural Network.

5. **Graph Creation**:
   - Function: `create_graph_from_data(nodes, edges)`
   - Creates graph from the provided test data

6. **Model Evaluation**:
   - Function: `evaluate_model(classifier, X_test, y_test)`
   - Evaluates precision, recall, and F1-score.

7. **Performance Logging**:
   - Function: `log_performance(start_time)`
   - Compares computational costs for effective compute use.

## Package Structure

The package includes the following modules:

- **Core Module**:
  - `drugXdisease/__init__.py`

- **Graph Handling**:
  - `drugXdisease/dXd_graph.py`

- **Calculate Graph Metrics**:
  - `drugXdisease/dXd_graph_metrics.py`

- **Logging**:
  - `drugXdisease/dXd_logging.py`

- **Data Handling, input**:
  - `drugXdisease/dXd_read_input.py`

- **Data Handling, output text and pictures**:
  - `drugXdisease/dXd_results.py`

- **Learning Approaches**:
  - Logistic Regression: `drugXdisease/dXd_logisticRegression.py`
  - Neural Network: `drugXdisease/dXd_neuralNet.py`
  - Random Forest: `drugXdisease/dXd_randomForest.py`

## Conclusions

### Feature Engineering

This is a way to create more features for learning without adding more data, but just using the existing data.

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
- Gradually expanded the graph with additional nodes, edges, weights, and embeddings for complexity. Data
choice seems to me to be crucial for predictions and performance.

### Test Data
- It is really difficult to create test data, that has some similarity to real data. The real data existing is so
 complex and biased in different ways so it is almost impossible to reconstruct. In this case, it seems to make
 more sense just to use slices of real data for all training, testing and experiementation.
 For visualisation I did "dumb" filtering, of just taking out nodes that are not relevant, and "dumb" subsetting
 if the data for visualisation. But for real implementation it would make more sense to do slices of the real graph,
 during development.

## Improvements

- The performance logging should be updated to include CPU etc which is suitable for the compute cluster we're at
- The graph image that is written is not very helpful, it should be updated to a nicer looking perhaps
 3D graph in order to be effective
- The evaluation could be more skewed from not finding the overall optimal solution, towards finding more
of what we really want; potential new drug-disease associations to explore further. For example a metrics like f1
is a bit helpful, but not optimal in that regard
- Trying more different classification approaches (most likely to create stepwise change)
- Stability testing - it might be that new drug-disease associations that are found using more different classification
approaches, or with the same approach re-run several times may be more reliable
- Reverse engineer the classifications, so we also can understand which features/input data most contributes to the
best solutions. There are elegant and more pragmatic ways of doing that.
- Any new drug-disease associations found needs additional orthogonal evidence to be pursued. For example scientific literature,
animal model experiments, or similar to build a case.


## Some Results

### Example Predictions

1. **Hyperpigmentation (MONDO:0019290)**:
   - Potential treatment: Hyperpigmentation of the skin (MONDO:0019290) could be treated by Azelaic acid (CHEMBL.COMPOUND:CHEMBL1238)
   - There is some evidence to support it, Azelaic acid is currently most used in treatment of acne, but effect on pigmentation
has been noted: "Azelaic acid, a tyrosine inhibitor, also helps reduce pigmentation, therefore is particularly
useful for darker skinned patients whose acne spots leave persistent brown marks
(postinflammatory pigmentation) or who have melasma."

2. **Cystic Fibrosis (MONDO:0009061)**:
   - Cystic fibrosis (MONDO:0009061) could be treated by CHLORHEXIDINE GLUCONATE (skin antiseptic) CHEMBL.COMPOUND:CHEMBL4297088
   - Obviously nonsense.

3. **Coronary Aneurysm (MONDO:0006714)**:
   - Prediction: Coronary aneurysm (MONDO:0006714) could be treated by Urokinase (CHEMBL.COMPOUND:CHEMBL1201420).
   - Assessment: Limited utility, as Urokinase is effective for blood clots, but coronary aneurysm only occasionally involves blood clots. Those cases with Coronary aneurysm with blood clots would currently be treated with aspirin, P2Y12 inhibitors, or general anticoagulants like warfarin. This probably would not be a lead we'd follow up on, because it is a bit non-specific, and it seems good options exists already.

4. **Lichen Planus (MONDO:0006572)**:
   - Prediction: lichen planus (a chronic, recurrent, pruritic inflammatory disorder of unknown etiology that affects the skin and mucus membranes, MONDO:0006572),
could be treated by ALEFACEPT (CHEMBL.COMPOUND:CHEMBL1201571).
   - Supporting Evidence: Looking at the CHEMBL website, "Lichen Planus" is already listed as studied.
https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL1201571 . This is the study: https://clinicaltrials.gov/study/NCT00135733?term=NCT00135733&rank=1 The results were published https://pubmed.ncbi.nlm.nih.gov/18459520/ and showed that 2 out of 7 patients achieved significant improvement, and was well tolerated by all patients. Authors recommended further study.

### Suggested Improvements
- Weigh results against broad-spectrum drugs (eg. anti-inflammatory) and diseases (e.g. headache, skin rash) to create a more targeted list.




