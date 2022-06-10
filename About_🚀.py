#importing necessary libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
import xgboost


# Display details of page 1

st.title("ABOUT")
st.write("""##### This repo is a part of K136. Kodluyoruz & Istanbul Metropolitan Municipality Data Science Bootcamp.The project aims to produce a machine learning model for home price estimation. The model was built on the Kaggle House Prices - Advanced Regression Techniques competition dataset.""")
st.write("""The dataset contains the following features:\n

- MSSubClass : Identifies the type of dwelling involved in the sale\n
- MoSold: Month Sold (MM)\n
- OverallQual: Rates the overall material and finish of the house\n
- OveralCond: Rates the overall condition of the house\n
- YrSold: Year Sold (YYYY)\n
- LotArea: Lot size in square feet\n
- LotFrontage: Linear feet of street connected to property\n
- BsmtUnfSF: Unfinished square feet of basement area\n
- GrLivArea: Above grade (ground) living area square feet\n
- GarageArea: Size of garage in square feet\n
- TotalBsmtSF: Total square feet of basement area\n
- YearBuilt: Year house was built\n
- 1stFlrSF: First Floor square feet\n
- BsmtFinSF1: Type 1 finished square feet\n
- OpenPorchSF: Open porch square feet\n
- MasVnrArea: Masonry veneer square feet\n
- YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)\n
- WoodDeckSF: Wood deck area in square feet\n
- GarageYrBlt: Year garage was built\n
- 2ndFlrSF: Second floor square feet\n
- SalePrice: Sale price""")

st.write("""## Model Development
The model is based on a XGBoost algorithm.
### Results
The model is trained on the dataset and tested on the test dataset.""")

st.write("#### R-SQUARED = 0.93")
st.write("#### MAE = 73211.47")
st.write("#### RMSE = 17707.2")

st.write("## Feature Importance")
st.image("outputFI.png",width=800)

x="https://www.tpfund.org/wp-content/uploads/2019/07/logo-1.png"
y="https://pbs.twimg.com/profile_images/1271062929874530306/xABPmpSo_400x400.jpg"
z="https://www.iogo.org.tr/wp-content/uploads/2020/11/istanbul-buyuksehir-belediyesi-vector-logo.png"
#a="https://iaahbr.tmgrup.com.tr/598bbc/806/378/0/35/678/353?u=https://iahbr.tmgrup.com.tr/2019/07/19/ismek-kurs-kayitlari-ne-zaman-basliyor-ismek-kurs-branslari-nelerdir-1563550981957.jpg"
b="https://media-exp2.licdn.com/dms/image/C4D0BAQFm0BDfiVWhzQ/company-logo_200_200/0/1640783137997?e=1663200000&v=beta&t=Yeys1DYwVuap2iLxCD2HvEl963LOQ4LWbSGvFNK_fNA"
liste=[y,x,z,b]
st.write("\n")
st.image(liste,width=176)

