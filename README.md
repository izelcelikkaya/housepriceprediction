# housepriceprediction <3
This project is a part of K136. Istanbul Data Science Bootcamp. 
house_price_1 is original dataset and evfiyati is modified version of house_price_1
HousePricesoriginal contains all processing that convert house_price_1 to evfiyati. There is also a modeling part.
HousePrices was created to be easier to work with. It uses evfiyati, which is a directly processed data.
HousePricePrediction_lit.py is streamlit code. 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/izelcelikkaya/housepriceprediction/main/About_🚀.py)
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
git clone https://github.com/izelcelikkaya/housepriceprediction.git
# change to the repo directory
cd housepriceprediction
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
streamlit run HousePrices.ipynb
```
## Model Development
### Model
The model is based on a [XGBoost](https://xgboost.readthedocs.io/en/stable/) algorithm.
### Training
```python
from xgboost import XGBRegressor
import xgboost as xgb
model = XGBRegressor(n_estimators=1150, 
                    max_depth=5, eta=0.03, 
                    subsample=0.5, 
                    colsample_bytree=0.8)
model.fit(X_train,y_train)
```
### Evaluation
```python
from sklearn.metrics import mean_squared_error
from math import sqrt
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))
pred = model.predict(X_test)
r_squared = model.score(X_test, y_test)
print(sqrt(mean_squared_error(y_test, pred)))
print(mae(y_test,pred))
print(r_squared)
```
## Results
The model is trained on the dataset and tested on the test dataset. The results are shown demo with Streamlit below:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/izelcelikkaya/housepriceprediction/main/About_🚀.py)
## Contributions
* İzel Çelikkaya - [Github](https://github.com/izelcelikkaya) - [LinkedIn](https://www.linkedin.com/in/izelcelikkaya)
* Mehmet Özmen - [Github](https://github.com/mehmetzmn) - [LinkedIn](https://www.linkedin.com/in/mehmetozmen)
* Uğur Can Kıvanç - [Github](https://github.com/Exedeus21) - [LinkedIn](https://www.linkedin.com/in/ugur-can-kivanc)
* Fuat Akdemir - [Github](https://github.com/FuatAkdemir) - [LinkedIn](https://www.linkedin.com/in/fuatakdemir)
* Çiğdem Taş - [Github](https://github.com/chidemmm) - [LinkedIn](https://www.linkedin.com/in/tashchidem)
* Ercan Tuncay - [Github] - [LinkedIn](https://www.linkedin.com/in/ercantuncay/)
* Serenay Ardahanlı - [Github](https://github.com/Serenayarda) - [LinkedIn](https://www.linkedin.com/in/serenay-ardahanli)
* Ali Haydar Şenyurt - [Github](https://github.com/alisenyurt87) - [LinkedIn](https://www.linkedin.com/in/ali-haydar-senyurt)
* Aybüke Akçay - [Github](https://github.com/akcaybuke) - [LinkedIn](https://www.linkedin.com/in/aybuke-akcay)
* Ömer Batuhan Özbay - [Github](https://github.com/kakan18) - [LinkedIn](https://www.linkedin.com/in/omerbatuhanozbay)
* Melisa Gündüz - [Github](https://github.com/megunduz) - [LinkedIn](https://www.linkedin.com/in/melisagunduz)
