# Aircraft engine failure prediction model
I tried to predict the RUL values for the 100 trajectories in the FD003 dataset [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan) using two different models (neural network and support vector machine).
My code can be found in the `turbofan.ipynb` file.

## Data Preparation, Feature Engineering
I computed the RUL value for each row in the training dataset and simplified the model by treating each sample as an independent observations.

I used the formula (3) in [1] to normalize each feature and then selected only features with standarad deviation > 0 to incude in the prediton. The transformed training dataset contains 24720 samples and 20 features.

## Neural Network Regression
The first model is a neural network implemented using `DNNRegressor` from the TensorFlow library. After some experimenting, I decided to use a network with 3 hidden layers containing 15, 30, and 15 units.

## Support Vector Machine
To use a Support vector machine (SVM) is suggested in [2]. The authors recommend to use non-linear radial basis (RBF) function.
SVM model is provided in the Scikit library. I used the default setting with the RBF kernel function.

## Results