import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# House Price Prediction App
""")
st.write("---")

# Load the house price Dataset
boston = pd.read_csv('boston_full.csv')

# specify predictors/features
features = boston.columns[:-1].tolist()
X = pd.DataFrame(boston, columns=features)

# specify target column
Y = pd.DataFrame(boston, columns=['MEDV'])

# Sidebar
# Header of specify Input Paramaters
st.sidebar.header('Specify Input Paramaters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM: per capita crime rate by town', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN=st.sidebar.slider('ZN: proportion of residential land zoned for lots over 25,000 sq.ft.', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS=st.sidebar.slider('INDUS: proportion of non-retail business acres per town', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS=st.sidebar.slider('Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX=st.sidebar.slider('NOX: nitric oxides concentration (parts per 10 million)', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM=st.sidebar.slider('RM: average number of rooms per dwelling', X.RM.min(), X.RM.max(), X.RM.mean())
    AGE=st.sidebar.slider('AGE: proportion of owner-occupied units built prior to 1940', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS=st.sidebar.slider('DIS: weighted distances to ﬁve Boston employment centers', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD=st.sidebar.slider('RAD: index of accessibility to radial highways', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX=st.sidebar.slider('TAX: full-value property-tax rate per $10,000', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO=st.sidebar.slider('PTRATIO: pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    BLACK=st.sidebar.slider('BLACK: 1000(Bk - 0.63)^2 where Black is the proportion of blacks by town.', X.BLACK.min(), X.BLACK.max(), X.BLACK.mean())
    LSTAT=st.sidebar.slider('LSTAT: lower status of the population (percent).', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': CRIM,
        'ZN': ZN,
        'INDUS': INDUS,
        'CHAS': CHAS,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'RAD': RAD,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'BLACK': BLACK,
        'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Bring in a pre-built model for use to remove necissity to keep training
with open('price_prediction_model', 'rb') as f:
    model = pickle.load(f)

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV: Median value of owner-occupied homes in $1000s')
st.write(f'${prediction*1000}')
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
plt.close()
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
plt.close()