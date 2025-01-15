# Machine Learning Assignment 2

## Overview
This assignment consists of three problems involving theoretical and practical aspects of machine learning. You will work with concepts like VC-dimension, the Perceptron algorithm, and Adaboost. Python will be used to implement algorithms and analyze results.

---

## Problem 1: VC-Dimension

### Description:
Given a class \( C \) with VC-dimension \( d \), and a new class \( C' \) that includes all objects formed by intersections and unions (in any order) of \( s \) objects in \( C \):
- Derive an **upper bound** for the VC-dimension of \( C' \).

### Deliverables:
- A theoretical derivation of the upper bound for the VC-dimension of \( C' \).

---

## Problem 2: Perceptron Algorithm

### Description:
Using the **UCI Iris Dataset** (150 flowers, 3 species: Setosa, Versicolor, Virginica):
1. Take only the **second and third features** for each flower.
2. Implement the **Perceptron algorithm** without normalizing the vectors.
3. Perform the following:
   - Run the algorithm for Setosa vs. Versicolor.
     - Report the final vector.
     - Count the number of mistakes made.
     - Calculate the true maximum margin.
   - Run the algorithm for Setosa vs. Virginica.
     - Report the final vector.
     - Count the number of mistakes made.
     - Calculate the true maximum margin.
   - Compare the results of the above runs.
   - Hypothesize what would happen for Versicolor vs. Virginica.

### Deliverables:
- Python code for the Perceptron algorithm.
- Results and analysis, including:
  - Final vector for each classification.
  - Number of mistakes made.
  - Maximum margin values.
  - Explanations of differences in results and hypotheses.

---

## Problem 3: Adaboost with Hypothesis Set

### Description:
Using the Iris dataset (Versicolor and Virginica):
1. Define a **hypothesis set** as the set of all lines passing through pairs of points in the dataset.
2. Implement **Adaboost** as follows:
   - Randomly split the data into 50% train and 50% test.
   - Use the training set to define the hypothesis set.
   - Identify the **8 most important hypotheses** and their weights (\( \alpha_i \)).
   - For each \( k = 1, \ldots, 8 \):
     - Compute the **empirical error** on the training set.
     - Compute the **true error** on the test set.
3. Execute **100 runs** of Adaboost and calculate the averages of:
   - Empirical error \( \bar{e}(H_k) \).
   - True error \( e(H_k) \).
4. Analyze the results:
   - Discuss the behavior of Adaboost on training and testing datasets.
   - Identify any overfitting or exceptional patterns.

### Deliverables:
- Python code for Adaboost implementation.
- Results and analysis, including:
  - Empirical and true error values for each \( H_k \) (average over 100 runs).
  - Observations on the behavior of Adaboost.

---

