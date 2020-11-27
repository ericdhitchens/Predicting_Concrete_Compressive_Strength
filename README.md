# Predicting Concrete Compressive Strength with Deep Neural Networks in TensorFlow 2.0

## Overview
The purpose of this project is to predict the compressive strength of concrete given initial quantities of its components and the age after mixing and installation ("cure time"). The engineering term for the relative amounts of each material contained within a concrete mix is called the "mix design." The following materials comprise a mix design:
* Cementitious materials (e.g. Portland cement, fly ash, etc.)
* Coarse Aggregate (e.g. crushed rock, stone, gravel, etc.)
* Fine aggregate (i.e. sand)
* Water
* Admixtures (materials to increase plasticity, prevent freezing, prevent corrosion, etc.)

## The Dataset
The dataset was retrieved from the UC Irvine Machine Learning Repository from the following URL: <https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength>. The dataset provides laboratory testing data for 1,030 concrete samples, tested at different curing times, with different mix designs.

The dataset was donated to the UCI Repository by Prof. I-Cheng Yeh of Chung-Huah University, who retains copyright for the following published paper: 
* I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). 

Additional papers citing this dataset are listed at the reference link above.

## Repository Structure
### 01_Data
Contains all data files in .xls or .xlsx and .csv format.
1. **Original** - The original data retrieved from the UCI dataset.
2. **Transformed** - The files with my own calculations and reformatting of column headers.
3. **Loaded** - The final data used in the Python Jupyter Notebooks.


### 02_Python_Code
Contains the code that was written in Jupyter Notebooks. Python files are also included for reference. PDF reports generated via LaTeX from the Jupyter notebooks are presented in 04_PDF_Reports.
1. **Exploratory_Data_Analysis**
	* READ THIS JUPYTER NOTEBOOK FIRST.
	* Provides civil engineering domain background to fully understand the dataset.
	* Describes features.
	* Provides initial visualizations of the data.
	* Gives recommendations for analysis, which are pursued in the ANN notebook.
2. **ANN_Modeling**
	* READ THIS JUPYTER NOTEBOOK SECOND.
	* Follows the recommendations provided in the EDA notebook.
	* Constructs ANNs for the data with performance evaluation and optimization.
3. **Model_Analysis**
	* READ THIS JUPYTER NOTEBOOK LAST.
	* Runs linear models on the various concrete constituents independently.
	* Compares linear models' performances with the ANN and gives final conclusions and recommendations.


### 03_Keras_ANN_Models 
Contains the saved deep neural network architectures developed in the ANN_Modeling Jupyter notebook:
1. **Flat_Model** - A deep neural network containing 44 hidden layers, each with 8 units.
2. **Descending Model** - A deep neural network containing 44 hidden layers, with the final 7 hidden layers decreasing by one node per layer from 8 to 2.
3. **Flat_Dropout_Model** - The same model as the Flat_Model, but half of the hidden layers drop out at a 0.5 rate.


### 04_PDF_Reports
Contains PDFs of the Jupyter notebooks from the 02_Python_Code folder. The PDFs were generated via LaTeX from the Jupyter Notebook platform.
