# Team Formation 
<p align="center">
  <img width="460" height="300" src="https://i.imgur.com/1W5Y4fU.png">
</p>

We focus on the composition of teams of experts that collectively cover a set of required skills based on their historical collaboration network and expertise. Prior works are primarily based on the shortest path between experts on the expert collaboration network,and suffer from three major shortcomings: 
(1) they are computationally expensive due to the complexity of finding paths on large network structures;
(2) they use a small portion of the entire historical collaboration network to reduce the search space; hence, may form sub-optimal teams;
(3) they fall short in sparse networks where the majority of the experts have only participated in a few teams in the past. 
Instead of forming a large network of experts, we propose to learn relationships among experts and skills through a variational Bayes neural architecture wherein:
- we consider all past team compositions as training instances to predict future teams;
- we bring scalability for large networks of experts due to the neural architecture;
- we address sparsity by incorporating uncertainty on the neural network’s parameters which yields a richer representation and more accurate team composition. 

We empirically demonstrate how our proposed model outperforms the state-of-the-art approaches in terms of effectiveness and efficiency based on a large DBLP dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project.

### Prerequisites

What things you need to install: the software and how to install them

```
Python 3.6
Tensorflow (GPU preferred)
Keras
NLTK
```
### Dataset
We choose DBLP as the benchmark. We consider each author to represent an expert and the authors of each publication to form a team. Presumably publication keywords would be reflective of the skills required for publishing the article. But, often publications in DBLP are not associated with keywords. Thus, we form the skill set S from the set of keywords extracted from the title of the publications. You can download dataset from this [here](https://lfs.aminer.cn/lab-datasets/citation/DBLP_citation_Sep_2013.rar). Downloaded dataset should be moved to [dataset](https://github.com/DoubleBlindRepo/team_formation/dataset) folder of the repository.

### Preprccessing

Before starting the project you may need to preproccess  the DBLP dataset to create the author-skill and team-skill mappings.

## Running the tests

After preparing the input data, models can be run individualy. For this matter, you need to run preferred model from the ML folder. After running each model, a series of question may appear on the screen asking for the embeddings, loading pre-trained weights on the screen. After the training phase, model will be tested on each fold automatically and predicted outputs will be saved in output directory. 


### Output files

[Output](/output) directory stores following data inside.
- Model snapshots 
- Predictions
- Evaluation results

At the end of ruuning session of each model, user will be asked wether if he/she wants to save the model or not. In case of approval model weights and configs will be saved in output folder. They will be accesible for next use.
Also, after running a model, predictions for the test set will be saved into the output folder for the futhur comparison.  You can find final evaluation results for each model in ".csv" individualy. They will be stored in folder.

## Evaluation
Evaluation of predicted files is done these metrics:

- Coverage @k
- NDCG @k
- MAP @k
- MRR @k

In order to evaluate predicted oputputs, you need to run the 
[comparison.py](https://github.com/DoubleBlindRepo/team_formation/eval/comparison.py)  file in [eval](https://github.com/DoubleBlindRepo/team_formation/eval) directory will calculate metrics for each model and will save the scores in the [output](https://github.com/DoubleBlindRepo/team_formation/output) 
 directory in ".csv" format.  
<p align="center">
  <img width="320" height="140" src="https://i.imgur.com/993AYVt.png">
</p>


### Help-Hurt Plot
Help Hurt diagram needs comparison of two methods. Therefore, a script [(HelpHurt.py)](https://github.com/DoubleBlindRepo/team_formation/eval/HelpHurt.py) has been written for this matter. You can find this file in run the [eval](https://github.com/DoubleBlindRepo/team_formation/eval) directory. After finishing the process result will be saved as a ".csv" file and is accessible in [output](https://github.com/DoubleBlindRepo/team_formation/output) directory.

<p align="center">
  <img width="320" height="100" src="https://i.imgur.com/w1qssZQ.png">
</p>


## Contributing

This repository is double-blidined for CIKM 2020 submission.
