
# Simple Install #

## Create conda env

conda env create -f bio-graph-env.yaml
conda activate bio-graph-env

(for production, a docker image would maybe be a better choice, but this should work reasonable easy on many systems)

## The script load_and_Represent_graph.py can be run 

python load_and_Represent_graph.py

# Testing #

Run:
python -m unittest discover tests
Only a simple testing framework has been created, as a marker more than a comprehensive testing structure


# MAIN SCRIPT 
Run: python load_and_Represent_graph.py 

The script has the following content:

Generate embeddings from the sample graph:  def generate_node_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

Load your embeddings: def load_embeddings(embeddings_file)
To make it easier to compute

Divide the embeddings into train and test data: create_dataset(graph, embeddings)
It uses 70% of data for training, and reserves 30% for testing

Train the classifier: def train_classifier(X_train, y_train)
Here, I chose a random forest classifier from sklearn, as a suitable and fairly fast learning approach, and a neural network approach.

Create a graph from input data: def create_graph_from_data(nodes, edges):

Evaluate the model: evaluate_model(classifier, X_test, y_test)
Evaluating the accuracy of the model with precision, recall and f1-score.

Check the compute metrics: log_performance(start_time)
For instance, compare how computationally expensive two different approaches are.

# PACKAGE 
A package was created to break out some helper routines
drugXdisease/__init__.py
Handle the graph: drugXdisease/dXd_graph.py
Calculate graph metrics: drugXdisease/dXd_graph_metrics.py
Log cluster performance: drugXdisease/dXd_logging.py
Read input data: drugXdisease/dXd_read_input.py
Write results and generate graphs: drugXdisease/dXd_results.py
Different learning approaches:
drugXdisease/dXd_logisticRegression.py
drugXdisease/dXd_neuralNet.py
drugXdisease/dXd_randomForest.py

# OUTPUT FILES 

General performance
Some performance metrics for the learning, random forest: results/evaluation_20250112_171908_rf.txt 
Some performance metrics for the learning, neural network: results/evaluation_20250112_171909_nn.txt
ROC curve overview: results/roc_curve_20250112_171909.png
Comput performance of the script (for making sure it computes fast enough): results/performance_20250112_171910.txt

Quality check:
Graph image overview (to check input nodes and edges): results/graph_20250112_171602.png

Results:
Results of drug x disease predictions (random forest): results/results_20250112_171910.tsv
Results of drug x disease predictions (neural network): results/results_nn_20250112_171910.txt



# Conclusions  

- Higher features. I tried looking at network-based metrics to add into the embeddings. 
For example:
Shortest Path Length: The shortest path in the graph between a drug and a disease.
Number of Paths: Count the number of paths between a drug and a disease.
Shared Genes: The number of genes connecting a drug to a disease.
and graph metrics like:
Node Degree: Include the number of connections a node has, becasue high-degree diseases may be associated with multiple drugs,
and high-degree genes may be central to important pathways.
Betweenness Centrality: Measure of how often a node lies on the shortest path between two other nodes. Becasue genes with high centrality are often key regulators.
Clustering Coefficient: Measures how tightly connected a node's neighbors are.
This is a way to create more features for learning without adding more data, but just using the existing data.

- Embeddings
It seems quite important to me to create embeddings which contains rich dimentionality of the knowledge graph, 
and I did some adjustments to improve the embeddings with more dimensions.

- There are a lot of choices around which data should go into the knowledge graph. 
Just adding in drugs, diseases, and targets, or adding in a wealth of other biological knowledge.
I ended up adding in more and more data, more edges, weights and adjusting the embeddings to make some test
data gradually more complex.  

- Test data
It is really difficult to create test data, that has some similarity to real data. The real data existing is so
 complex and biased in different ways so it is almost impossible to reconstruct. In this case, it seems to make
 more sense just to use slices of real data for all training, testing and experiementation.
 For visualisation I did "dumb" filtering, of just taking out nodes that are not relevant, and "dumb" subsetting 
 if the data for visualisation. But for real implementation it would make more sense to do slices of the real graph, 
 during development.


# Improvements

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



# Some results

Result 1: 
Hyperpigmentation of the skin (MONDO:0019290) could be treated by Azelaic acid (CHEMBL.COMPOUND:CHEMBL1238)
There is some evidence to support it, Azelaic acid is currently most used in treatment of acne, but effect on pigmentation
has been noted: "Azelaic acid, a tyrosine inhibitor, also helps reduce pigmentation, therefore is particularly 
useful for darker skinned patients whose acne spots leave persistent brown marks 
(postinflammatory pigmentation) or who have melasma."

Result 2:
Cystic fibrosis (MONDO:0009061) could be treated by CHLORHEXIDINE GLUCONATE (skin antiseptic) CHEMBL.COMPOUND:CHEMBL4297088
Obviously nonsense.

Result 3:
Coronary aneurysm (MONDO:0006714) for example in Kawasaki disease, could be treated by Urokinase (CHEMBL.COMPOUND:CHEMBL1201420).
Urokinase is also used clinically as a thrombolytic agent in the treatment of severe or massive deep venous thrombosis, 
peripheral arterial occlusive disease, pulmonary embolism, acute myocardial infarction (AMI, heart attack), and occluded dialysis cannulas (catheter clearance).
Coronary aneurysm in Kawasaki disease is currently treated with intravenous gamma globulin (IVIG), and high-dose aspirin every six hours, to help reduce the 
swelling and inflammation in the blood vessels, so urokinase may not be so helpful for that. Coronary aneurysm can also be caused by atherosclerosis
(buildup of plaques (fatty deposits) in the arterial walls), connective tissue disorders, and infections.
As urokinase primarily removes blot clots, it is probably not so effective for many causes of coronary aneurysm, except for in those cases where there is a 
blood clot formation within the aneurysm. Those would currently be treated with aspirin, P2Y12 inhibitors, or general anticoagulants like warfarin.
This probably would not be a lead we'd follow up on, because it is a bit non-specific, and it seems good options exists already.     

Result 4:
lichen planus (a chronic, recurrent, pruritic inflammatory disorder of unknown etiology that affects the skin and mucus membranes, MONDO:0006572), 
could be treated by ALEFACEPT (CHEMBL.COMPOUND:CHEMBL1201571). Looking at the CHEMBL website, "Lichen Planus" is already listed as studied.
https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL1201571 . This is the study: https://clinicaltrials.gov/study/NCT00135733?term=NCT00135733&rank=1 
The results were published https://pubmed.ncbi.nlm.nih.gov/18459520/ and showed that 2 out of 7 patients achieved significant improvement, and was well 
tolerated by all patients. Authors recommended further study.

The results could probably be improved by weighting against results for broad-spectrum drugs (anti-inflammatory), 
and broad-spectrum diseases (e.g. headache, skin rash), to get a shorter and more targeted list.


