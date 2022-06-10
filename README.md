# housepriceprediction <3
This project is a part of K136. Istanbul Data Science Bootcamp. 

house_price_1 is original dataset and evfiyati is modified version of house_price_1

HousePricesoriginal contains all processing that convert house_price_1 to evfiyati. There is also a modeling part.
HousePrices was created to be easier to work with. It uses evfiyati, which is a directly processed data.

HousePricePrediction_lit.py is streamlit code. 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/izelcelikkaya/housepriceprediction/main/HousePricePrediction_lit.py)

## Description

This repo has been developed for the Istanbul Data Science Bootcamp, organized in cooperation with İBB & Kodluyoruz. Prediction for house prices was developed using the Kaggle House Prices - Advanced Regression Techniques competition dataset.

## Data

The dataset is available at [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Goal

The goal of this project is to predict the price of a house in Ames using the features provided by the dataset.

## Features

The dataset contains the following features:

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
* **SalePrice**: Sale price

## Usage

```bash
# clone the repo
git clone https://github.com/uzunb/house-prices-prediction-LGBM.git

# change to the repo directory
cd house-prices-prediction-LGBM

# if virtualenv is not installed, install it
#pip install virtualenv

# create a virtualenv
virtualenv -p python3 venv

# activate virtualenv for LINUX or MACOS
source venv/bin/activate

# # activate virtualenv for WINDOWS
# venv\Scripts\activate.ps1
#     # throubleshooting for activation error in windows
#     Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# install dependencies
pip install -r requirements.txt

# run the script
streamlit run main.py
```

## Model Development

### Model

The model is based on a [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) algorithm.

### Training

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(max_depth=3, 
                    n_estimators = 100, 
                    learning_rate = 0.2,
                    min_child_samples = 10)
model.fit(x_train, y_train)
```

Grid Search Cross Validation is used for hyper parameters of the model.

```python
from sklearn.model_selection import GridSearchCV

params = [{"max_depth":[3, 5], 
            "n_estimators" : [50, 100], 
            "learning_rate" : [0.1, 0.2],
            "min_child_samples" : [20, 10]}]

gs_knn = GridSearchCV(model,
                      param_grid=params,
                      cv=5)

gs_knn.fit(x_train, y_train)
gs_knn.score(x_train, y_train)

pred_y_train = model.predict(x_train)
pred_y_test = model.predict(x_test)

r2_train = metrics.r2_score(y_train, pred_y_train)
r2_test = metrics.r2_score(y_test, pred_y_test)

msle_train =metrics.mean_squared_log_error(y_train, pred_y_train)
msle_test =metrics.mean_squared_log_error(y_test, pred_y_test)

print(f"Train r2 = {r2_train:.2f} \nTest r2 = {r2_test:.2f}")
print(f"Train msle = {msle_train:.2f} \nTest msle = {msle_test:.2f}")

print(gs_knn.best_params_)
```

### Evaluation

```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error

y_pred = model.predict(x_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Squared Log Error:', mean_squared_log_error(y_test, y_pred))
print('Explained Variance Score:', explained_variance_score(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))
```

## Deployment

Simple model distribution is made using Streamlit.

```python
import streamlit as st

st.title("House Prices Prediction")
st.write("This is a simple model for house prices prediction.")

st.sidebar.title("Model Parameters")

variables = droppedDf["Alley"].drop_duplicates().to_list()
inputDict["Alley"] = st.sidebar.selectbox("Alley", options=variables)

inputDict["LotFrontage"] = st.sidebar.slider("LotFrontage", ceil(droppedDf["LotFrontage"].min()), 
floor(droppedDf["LotFrontage"].max()), int(droppedDf["LotFrontage"].mean()))
```

## Results

The model is trained on the dataset and tested on the test dataset. The results are shown demo with Streamlit below:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/uzunb/house-prices-prediction-lgbm/main/1_%F0%9F%92%BB_Enter_Page.py)

## Contributions

* Batuhan UZUN - [Github](https://https://github.com/uzunb) - [LinkedIn](https://linkedin.com/in/uzunb)
* Dursun Tunahan BİLGİN - [Github](https://github.com/bilgind17) - [LinkedIn](https://www.linkedin.com/in/dtunahanbilgin/)
* Anıl DÖNMEZ - [Github](https://github.com/anildonmz) - [LinkedIn](https://www.linkedin.com/in/anilldonmez/)
* Hazal SEZGİN - [Github](https://github.com/hazalsezgin) - [LinkedIn](https://www.linkedin.com/in/hazal-sezgin-48a253170)
* Müşerref ÖZKAN - [Github](https://github.com/MuserrefOzkn) - [LinkedIn](https://www.linkedin.com/in/müşerrefözkan)
* Hanife YAMAN - [Github](https://github.com/hanifeyaman) - [LinkedIn](https://www.linkedin.com/in/hanife-yaman/)
* Üftade Bengi EROLÇAY - [Github](https://github.com/uftadeerolcay) - [LinkedIn](https://www.linkedin.com/in/uftade-bengi-erolcay)
* Yiğit YILMAZ - [Github](https://github.com/yilmazyigit) - [LinkedIn](https://www.linkedin.com/in/yigityilmaz4/)
* Selin ÇILDAM - [Github](https://github.com/selincildam) - [LinkedIn](https://www.linkedin.com/in/selincildam/)
