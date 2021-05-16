# Team Formation PyPI Tensorflow Python Library
<p align="center">
  <img width="460" height="300" src="https://i.imgur.com/1W5Y4fU.png">
</p>

The Team Formation PyPI Tensorflow Python Library focuses on the composition of teams of experts that collectively cover 
a set of required skills based on their historical collaboration network and expertise. Prior works are primarily based 
on the shortest path between experts on the expert collaboration network,and suffer from three major shortcomings: 
(1) they are computationally expensive due to the complexity of finding paths on large network structures;
(2) they use a small portion of the entire historical collaboration network to reduce the search space; hence, may form 
sub-optimal teams;
(3) they fall short in sparse networks where the majority of the experts have only participated in a few teams in the 
past. 
Instead of forming a large network of experts, we propose to learn relationships among experts and skills through a 
variational Bayes neural architecture wherein:
- we consider all past team compositions as training instances to predict future teams;
- we bring scalability for large networks of experts due to the neural architecture;
- we address sparsity by incorporating uncertainty on the neural network’s parameters which yields a richer 
representation and more accurate team composition. 

The PyPI python library implements the above-mentioned functionality by pipe-lining its architecture into stages that 
use classes and functions to maintain a fluid data flow.
The pipeline consists of 5 stages that are as follows: (1) data access layer instantiation; (2) dictionaries/embeddings 
generation; (3) train/test dataset split; (4) VAE learning; and, (5) performance evaluation. We empirically demonstrate 
how our proposed model outperforms the state-of-the-art approaches in terms of effectiveness and efficiency based on a 
large DBLP dataset.

## Getting Started

These instructions will get you the Team Formation PyPI library installed on your 
machine and you will be able to use its features in a python compiler.

### Prerequisites

These are the python libraries you need to pre-install before using this package.

```
Python 3.6 (or higher)
Tensorflow 1.15.0 (GPU preferred)
tensorboard 1.15.0
tensorboard-plugin-wit 1.7.0
tensorflow-estimator 1.15.1
tensorflow-probability 0.8.0
Keras 2.0.0
Keras-Applications
Keras-Preprocessing 
gensim
iteration-utilities
keras-metrics
matplotlib
ml-metrics
NLTK 3.5
numpy
pandas
scikit-learn
scipy
sklearn
xlwt 
```
### Dataset
We choose DBLP as the benchmark. However, you can use your own database in similar
fashion to perform team formation.

### Preprocessing

Before starting the project you would need to preprocess your dataset to create the author-skill and team-skill mappings.

### Output files

[Output](/teamFormationLibrary/output) directory stores following data inside.
- Model snapshots 
- Predictions
- Evaluation results

At the end of running session of each model, user will be asked wether if he/she wants to save the model or not. In case of approval model weights and configs will be saved in output folder. They will be accesible for next use.
Also, after running a model, predictions for the test set will be saved into the output folder for the futhur comparison.  You can find final evaluation results for each model in ".csv" individualy. They will be stored in folder.

## Evaluation
Evaluation of predicted files is done using the following metrics:

- Recall @k
- NDCG @k
- MAP @k
- MRR @k

The following diagram is a performance evaluation on the DBLP dataset.
<p align="center">
  <img width="320" height="240" src="https://i.ibb.co/6yN20PF/metric-fig.png">
</p>

## Contributing

This branch is submitted as a public library on the PyPI API.
