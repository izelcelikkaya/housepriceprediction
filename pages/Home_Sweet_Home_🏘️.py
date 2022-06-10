#importing necessary libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle


#importing dataframe 
x="https://external-preview.redd.it/wEHXsHI7CTei6iQVuJzS4zyIeW_sVME5akvkvE67CvU.jpg?width=960&crop=smart&auto=webp&s=cd7dad26064e8302c9fa88dbcbf5afb99165dc00" 
st.image(x,width=800)
st.write(" ## Let's Predict Your Dream Home's Value :) ##")
df = pd.read_csv('evfiyati.csv') 

#variables_list contains chosen variables for users, these variables are first 20 most important features

variables_list=['LotArea', 'LotFrontage', 'BsmtUnfSF', 'GrLivArea', 'MSSubClass',
                'GarageArea', 'TotalBsmtSF', 'YearBuilt', '1stFlrSF', 'BsmtFinSF1',
                'OpenPorchSF', 'MasVnrArea', 'MoSold', 'YearRemodAdd', 'WoodDeckSF',
                'OverallQual', 'GarageYrBlt', 'OverallCond', '2ndFlrSF', 'YrSold']

box_desc_list = ['Identifies the type of dwelling involved in the sale', 'Month Sold', 
                 'Rates the overall material and finish of the house', 'Rates the overall condition of the house', 'Year Sold']

slider_desc_list = ['Lot size in square feet', 'Linear feet of street connected to property', 'Unfinished square feet of basement area',
                    'Above grade (ground) living area square feet', 'Size of garage in square feet', 'Total square feet of basement area', 
                    'Original construction date', 'First Floor square feet', 'Type 1 finished square feet', 'Open porch area in square feet',
                    'Masonry veneer area in square feet', 'Remodel date (same as construction date if no remodeling or additions)', 
                    'Wood deck area in square feet', 'Year garage was built', 'Second floor square feet']

box_list = []
slider_list = []

# if amount of unique variables more than 20, then use selectbox
for var in range(len(variables_list)):
    if len(df[variables_list[var]].unique()) < 20:
        box_list.append(variables_list[var])
    elif len(df[variables_list[var]].unique()) >= 20:
        slider_list.append(variables_list[var])

box_overall_dict = {}
slider_overall_dict = {}

# Creating dictionary for value names and their descriptions
for var1, var2 in zip(box_list, box_desc_list):
    box_overall_dict.update({var1: var2})

for var1, var2 in zip(slider_list, slider_desc_list):
    slider_overall_dict.update({var1: var2})

print(box_overall_dict)
print(slider_overall_dict)

# Displaying box and slider with functions
def showing_box(var, desc):
        cycle_option = sorted(list(df[var].unique()))
        box = st.sidebar.selectbox(label= f"{desc}", options=cycle_option)
        return box

def showing_slider(var, desc):
        slider = st.sidebar.slider(label= f"{desc}", min_value=round(df[var].min()), max_value=round(df[var].max()))
        return slider

print(box_list)
print(slider_list)


# Collecting user inputs in dictionaries
box_dict = {}
slider_dict = {}

for key, value in box_overall_dict.items():
    box_dict.update({key: showing_box(key, value)})

for key, value in slider_overall_dict.items():
    slider_dict.update({key: showing_slider(key, value)})

print(box_dict)
print(slider_dict)


#keeping inputs in a dic
input_dict = {**box_dict, **slider_dict}
dictf = pd.DataFrame(input_dict, index=[0])
df = df.append(dictf, ignore_index= True) 

#giving default values to other variables 
for i in df.columns:
    if df[i].dtypes in ["object"]:
        df[i].fillna(df[i].mode(),inplace = True)
    else:
        df[i].fillna(df[i].mean(),inplace = True)

#drop uncessary variables 
df.drop("Id", inplace=True,axis=1)
df.drop("SalePrice", inplace=True,axis=1)
df2 = pd.get_dummies(df)
scaler = StandardScaler()
scaler.fit(df2)
df3 = pd.DataFrame(scaler.transform(df2),index = df2.index,columns = df2.columns)

#selecting only last row. (User input data)
newdata=pd.DataFrame(df3.iloc[[-1]])

#load already trained model (XGBoost)
with open('finalized_model.model' , 'rb') as f:
    lr = pickle.load(f)

#adding a button 
#""" if st.sidebar.button('Show House Price'):
#    ypred = lr.predict(newdata)
#    st.title("Value of your dream house: ")
#    st.title(str(np.round(ypred[0]))+" $") """

ypred = lr.predict(newdata)
st.title("Value of your dream house: ")
st.title(str(np.round(ypred[0]))+" $")
