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
#liste contains chosen variables for users
liste=["HeatingQC","OverallQual","KitchenQual","ExterQual","YearBuilt","GrLivArea","YearRemodAdd","FullBath","LotArea","GarageArea"]

cycle_options = [1,2,3,4,5]
s1 = st.sidebar.selectbox('Heating Quality And Condition',options=cycle_options)

cycle_2 = [1 , 2,  3,  4,  5, 6,  7, 8,  9,10]
s2 = st.sidebar.selectbox("The Overall Material And Finish of The House",options=cycle_2)

cycle_3 = [1,2,3,4,5]
s3 = st.sidebar.selectbox("Kitchen Quality",options=cycle_3)

cycle_4 = [1,2,3,4,5]
s4 = st.sidebar.selectbox("The Quality of The Material On The Exterior",options=cycle_4)

cycle_5 = [0,1,2,3]
s5 = st.sidebar.selectbox("Full Bathrooms Above Grade",options=cycle_5)

s6=st.sidebar.slider("Original Construction Date",min_value=1950, max_value=2010)
s7=st.sidebar.slider("Remodel Date",min_value=1950, max_value=2010)
s8=st.sidebar.slider("Above Grade (ground) Living Area Square Feet",min_value=334,max_value=2800)
s9=st.sidebar.slider("Lot Size in Square Feet",min_value=1481,max_value=17675)
s10=st.sidebar.slider("Garage Area",min_value=0,max_value=940)

#keeping inputs in a dic
input_dict = {'HeatingQC':[s1],'OverallQual':[s2],'KitchenQual':[s3],'ExterQual':[s4],'FullBath':[s5],'YearBuilt':[s6],'YearRemodAdd':[s7],
        'GrLivArea':[s8],'LotArea':[s9],'GarageArea':[s10]}
dictf = pd.DataFrame(input_dict)
df = df.append(dictf,ignore_index= True) 

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
if st.sidebar.button('Show House Price'):
    ypred = lr.predict(newdata)
    st.title("Value of your dream house: ")
    st.title(str(np.round(ypred[0]))+" $")



