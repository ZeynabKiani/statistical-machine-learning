# Decision Tree Regression

This repository contains a demonstration of 1D regression using a decision tree. The decision tree is employed to fit a sine curve with additional noisy observations. The model learns local linear regressions, approximating the sine curve.

## Overview

In the provided example, the decision tree's maximum depth, controlled by the `max_depth` parameter, plays a crucial role. When the maximum depth is set too high, the decision tree tends to capture fine details and noise from the training data, leading to overfitting.

## Example Image

![Decision Tree Regression](sphx_glr_plot_tree_regression_001.png)

## Usage

Clone the repository and explore the example to understand how the decision tree regression behaves under different parameter settings. You can visualize and modify the example as needed.

```bash
git clone https://github.com/your-username/decision-tree-regression.git
cd decision-tree-regression
# Run the example or modify as per your requirements
python decision_tree_regression_example.py
```

## Dependencies

Make sure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to experiment with the decision tree regression and adapt it to your specific use cases. If you encounter any issues or have suggestions, please create an issue or submit a pull request. Happy coding!
