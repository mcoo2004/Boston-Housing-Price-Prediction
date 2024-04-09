# Boston-Housing-Price-Prediction
Boston Housing Price Prediction

Overview
This repository contains a Python script for predicting the median value of owner-occupied homes in the Boston area. Utilizing the Boston housing dataset, the script demonstrates the application of linear regression through both gradient descent and normal equations methods. The goal is to provide an accessible example of basic machine learning techniques in action.

Features:
Linear regression model implementation
Two methods for parameter estimation: gradient descent and normal equations
Data normalization
Mean Squared Error (MSE) calculation for model evaluation

Requirements:
Python 3
Pandas (for data manipulation)
NumPy (for numerical computations)
Scikit-learn (for data splitting and normalization)
To install these dependencies, you can use pip:

"pip install pandas numpy scikit-learn"

Dataset:
The dataset used is the "Boston Housing Dataset", available as boston.csv in this repository. It includes several features such as crime rate, property tax rate, average number of rooms, etc., and the target value is the median value of owner-occupied homes (in $1000's).

Usage Instructions:
Ensure that all requirements listed above are installed in your Python environment.
Clone this repository or download the files to your local machine.
Place the boston.csv file in the same directory as the BostonPricePrediction.py script. If it's located elsewhere, update the file_path variable in the script accordingly.
Run the script using the following command:
"python BostonPricePrediction.py"

The script will train the linear regression model using both the gradient descent and normal equations methods. It then evaluates these models on a test set and prints the MSE for both.
Results, including the final theta values and MSE for each method, are saved to results.txt.

Results:
The final theta values and the MSE of the model on the test set are saved to results.txt. This file will be generated in the same directory as the script upon successful execution.

Contributions:
Contributions are welcome! If you have suggestions for improvements or new features, feel free to fork this repository and submit a pull request.

License:
This project is open source and available under the MIT License.
