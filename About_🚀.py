#importing necessary libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
import xgboost


# Display details of page 1

st.set_page_config(page_title="About Page - House Price Prediction", page_icon="🚀")
st.title("ABOUT")
st.write("""##### This repo is a part of K136. Kodluyoruz & Istanbul Metropolitan Municipality Data Science Bootcamp. The project aims to produce a machine learning model for home price estimation. The model was built on the Kaggle House Prices - Advanced Regression Techniques competition dataset. """)
st.write(""" The dataset contains the following features: \n

* **MSSubClass**: Identifies the type of dwelling involved in the sale
* **MoSold**: Month Sold (MM)
* **OverallQual**: Rates the overall material and finish of the house
* **OveralCond**: Rates the overall condition of the house
* **YrSold**: Year Sold (YYYY)
* **LotArea**: Lot size in square feet
* **LotFrontage**: Linear feet of street connected to property
* **BsmtUnfSF**: Unfinished square feet of basement area
* **GrLivArea**: Above grade (ground) living area square feet
* **GarageArea**: Size of garage in square feet
* **TotalBsmtSF**: Total square feet of basement area
* **YearBuilt**: Year house was built
* **1stFlrSF**: First Floor square feet
* **BsmtFinSF1**: Type 1 finished square feet
* **OpenPorchSF**: Open porch square feet
* **MasVnrArea**: Masonry veneer square feet
* **YearRemodAdd**: Remodel date (same as construction date if no remodeling or additions)
* **WoodDeckSF**: Wood deck area in square feet
* **GarageYrBlt**: Year garage was built
* **2ndFlrSF**: Second floor square feet
* **SalePrice**: Sale price""")

st.write("""## Model Development
The model is based on a XGBoost algorithm.
### Results
The model is trained on the dataset and tested on the test dataset.""")

st.write("#### R-SQUARED = 0.93")
st.write("#### MAE = 73211.47")
st.write("#### RMSE = 17707.2")

st.write("## Feature Importance")
st.image("Images/outputFI.png", width=800, caption='Feature importance graph')

st.write("\n")

st.write("## Our Dear Supporters")
st.image("Images/logos.jpg", width=800, caption='Our supporters')

