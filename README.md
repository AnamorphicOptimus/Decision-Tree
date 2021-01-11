# Decision Tree

## Introduction
"Decision Tree" is an experimental project. Through self-study on the principle of decision tree splitting, I finally completed the realization of the following three models, which are implemented in Python. Some suggestions and questions from the experiment were also put forward later.

## Experimental Results

| Model | Train Acc | Test Acc | Train Time (s)|
| ------ | ------ | ------ | ------ |
| ID3 | 0.99 | 0.91 | 0.15 | 
| RandomForest | 1.0 | 0.91 | 45.4 |
| GBDT | 0.97 | 0.89 | 27.9 |

## Experiment Conclusions:

- 1. ID3: The core of the algorithm is to apply information gain criteria to feature selection on each node of the decision tree. The unimproved ID3 algorithm cannot handle continuous values, and there is no pruning strategy, which is easy to overfit. In this experiment, considering that you want to test on the same data, ID3 is improved, so the focus is on the feature selection method. However, the information gain criterion is still flawed. It has a preference for features with a large number of possible values, and its information gain is close to 1. Finally, ID3 algorithm has no related missing value processing part.

- 2. Random forest: On the basis of the problem of decision trees that often have overfitting, random forest-a combination of multiple random decision trees can prevent this type of problem from occurring and greatly improve the generalization ability of the model. And the random forest is relatively stable, even if a new data point appears, the impact on the entire algorithm will not be too large. However, random forest has the problem of high computational cost, and generally requires more time to train the model. Moreover, the random forest is like a "black box", which requires us to spend a long time looking for parameters to improve the prediction ability. Finally, its time complexity is o(N·P+Depth).

- 3. GBDT: GBDT is relatively flexible to handle various types of data, including continuous values and discrete values. However, due to the dependency between weak learners, it is difficult to train data in parallel. (However, partial parallelism can be achieved through self-sampling SGBT.) At the same time, if we use some robust loss functions, the robustness to outliers is very strong. Such as Huber loss function and Quantile loss.

## Model Improvements & Problems:

- 1. When improving the ID3 algorithm, there is a more detailed area that has been dealt with for a long time. When generating a decision tree, if the current attribute is a continuous attribute, it cannot be deleted from the original data set (compare discrete attributes), because it will also be used later (due to model limitations, ID3 decision trees are not strictly binary trees, so they are often better than binary trees. The processing is more complicated).

- 2. When researching the random forest algorithm on the Internet, I found that there are many random forests that can play a role, such as computing costs. You can consider using multiple trees to calculate in parallel; for example, the adjustment of the number of decision trees affects the model; And the method used for splitting attributes can also have multiple choices, such as information entropy, Gini coefficient, and so on.

- 3. The improvement space of GBDT we can focus on the adjustment of learning_rate and loss function. There are many loss functions to choose from, exponential, logarithmic, Huber, quantile and so on. And the criteria adopted when selecting split attributes include MSE, MAE, and so on.




