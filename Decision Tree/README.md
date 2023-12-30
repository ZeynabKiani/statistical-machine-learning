# 1D Regression with Decision Tree

![Decision Tree Regression](image/sphx_glr_plot_tree_regression_001.png)

## Overview

This repository contains a demonstration of 1D regression using a decision tree. In this example, the decision tree is employed to fit a sine curve with additional noisy observations. The goal is to showcase how decision trees can learn local linear regressions to approximate complex functions such as the sine curve.

## Usage

To run the demonstration, make sure you have the required dependencies installed. You can install them using the following:

```bash
pip install -r requirements.txt
```

Once the dependencies are installed, run the script:

```bash
python decision_tree_regression.py
```

This will execute the decision tree regression and generate visualizations. Feel free to modify parameters and explore how the decision tree behaves with different settings.

## Observations

The key observation from this example is the impact of the maximum depth parameter (`max_depth`) on the decision tree's performance. If the maximum depth is set too high, the tree may learn fine details of the training data, including noise. This phenomenon is known as overfitting, where the model becomes too specific to the training data and fails to generalize well to new, unseen data.

Adjust the `max_depth` parameter to explore its influence on the decision tree's ability to capture the underlying pattern in the data while avoiding overfitting.

## References

- [Scikit-Learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html#tree)

Feel free to experiment and contribute to enhance this example!