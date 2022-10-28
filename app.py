import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

st.set_page_config(
    page_title="SME Default Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_data():
    df = pd.read_csv('Technical_Task_Dataset.csv')
    return df

def load_model():
    model = pickle.load(open('classification.sav', 'rb'))
    return model

def load_rfe():
    rfe = pickle.load(open('rfe.sav', 'rb'))
    return rfe

@st.cache
def plot_histogram(data, x, height, width, margin, title_text=None):
    fig = px.histogram(data, x=x)
    fig.update_layout(bargap=0.05, 
                      height=height, 
                      width=width, 
                      title_text=title_text, 
                      margin=dict(t=margin, b=margin))
    return fig

@st.cache
def plot_heatmap(corr, height, margin, title_text=None):
    fig = go.Figure(go.Heatmap(z=np.array(corr), 
                               x=corr.columns, 
                               y=corr.columns,
                               colorscale=px.colors.diverging.RdBu, 
                               zmax=1, zmin=-1))       
    fig.update_layout(bargap=0.05, 
                      height=height, 
                      width=height+100, 
                      title_text=title_text, 
                      margin=dict(t=margin, b=margin))
    return fig

def preprocess_df(df):
    df2 = df[df.index.isin(df.query('1900 < `Account Year` < 2023').index)]
    df2=df2.convert_dtypes()
    df3 = df2.copy()
    
    for col in df3.columns:
        if df3[col].dtype == 'string':
            try:
                df3[col] = pd.to_datetime(df2[col])
            except ValueError:
                pass
            
    df4 = df3[df3['Latest Accounts Date'].dt.year==df3['Account Year']]
    
    df4['Years Since Incorporation'] = df4['Account Year']-df4['Date of Incorporation'].dt.year
    df41 = df4.copy()
    df41['Years Since Incorporation']= df41['Years Since Incorporation'].astype(int)
    
    df5 = df41.copy()
    df5['Trading Status'] = np.where(df5['Trading Status'] == 'Active', "Non-default", "Default")
    
    df5['Directors Remuneration'].fillna(df5['EBITDA + Directors Remuneration'] - df5['EBITDA'], inplace=True)
    df5['Total Assets'].fillna(df5['Total Current Assets'] + df5['Total Non Current Assets'], inplace=True)
    df5['Working Capital'].fillna(df5['Total Current Assets'] + df5['Total Current Liabilities'], inplace=True) 
    
    df6 = df5.drop(columns=['Account Year', 'Bank Overdraft', 'Bank Postcode', 'Capital Expenditure', 
                            'Date of Incorporation', 'Director Loans (current)', 'Director Loans (non-current)',
                            'EBIT', 'EBITDA + Directors Remuneration', 'Highest Paid Director ',
                            'Latest Accounts Date', 'Leasehold', 'Profit Before Tax + Directors Remuneration',
                            'Registered Number', 'Registered or Trading Postcode', 
                            'Total Non Current Liabilities (Incl Provisions)', 'UK SIC Code', 'Wages', 'Working Capital'])
    df7 = df6.dropna(how='any').reset_index(drop=True)
    return df5, df7

    
def prediction(model, c_bank, profit, r_earn, t_asset, t_equity):
    prediction = model.predict([[c_bank, profit, r_earn, t_asset, t_equity]])
    prediction_prob = model.predict_proba([[c_bank, profit, r_earn, t_asset, t_equity]])
     
    if prediction == 'Default':
        pred = 'default'
    else:
        pred = 'not default'
    pred_prob = prediction_prob[0][0]*100
    return pred, pred_prob


def main():
    """
    # SME Default Prediction App

    A machine learning app to predict the proability of a company's loan defaulting

    """    
      
    # ----------- Data -------------------
    
    df = load_data()
    model = load_model()
    rfe = load_rfe()
    
    
    # ----------- Sidebar ---------------
    
    condition = st.sidebar.selectbox("Select a page:",
                                     ("Introduction", "Data preprocessing", "The machine learning model", "Make a prediction"))
    
    # ------------- Introduction ------------------------
    
    if condition == 'Introduction':
        
        st.title("Loan Default Prediction App")
        st.write("A machine learning app to predict the proability of an SME's loan defaulting")
        
        st.subheader('About')

        st.write("""
        This app predicts the probability of a SME loan defaulting based on selected inputs. 
        The sidebar contains the following sections:
        - Data preprocessing
        - The machine learning model
        - Make a prediction
        """)
        
        st.subheader('Technical task questions')
        
        st.write("""
        1. In two or three paragraphs, please describe your methodological approach to the problem (e.g., how you framed the problem, any assumptions you made, why you chose certain techniques, etc). If applicable, please include any references to the literature that you used
            
            The first step I took was to conduct data preprocessing and model implementation in a Jupyter Notebook, which allowed for quick visualisations.
            Once these steps were complete, I transferred the relevant code to a py file and created the streamlit app.
            
            My initial focus was on the data available, investigating the variables and their relationship to each other.
            Many missing data points existed, which was handled by dropping NAs, although imputation methods could have been used.
            Using recursive feature elimination to reduce the feature space, five features were selected as input to the model.
            Once the data was preprocessed, I developed a predictive model for binary classification.
            The literature showed that neural networks have promise for this type of problem, however, for the sake of simplicity and time, I chose to implement a Random Forest, which has also shown promise.
            
            The resulting Random Forest model was trained on a random sample of 80% of the data and tested on the remaining 20%.
            Cross validation could be applied to reduce the chances of overfitting and improve the ability to predict on unseen data.
            Overall, the modelled had an accuracy of 84%, which could be improved with more input.
        2. In 3 bullet points, please explain what feedback you'd give to the engineering team responsible for the data API to help them improve any aspect of the tool that you think would benefit
            - Put checks in place to ensure data is entered correctly, e.g., postcodes appeared in the latest accounts dates
            - If missing data can be filled based on data from other variables, then this should be implemented, e.g., total assets = total current assets + total non-current assets, so if one of the three variables are missing, then these can be easily imputed.
            - Incorporate further information about the business, e.g., value of the loan applied for, credit history, free text information allowing for NLP.
        3. In 3 bullet points, please give guidance to the business regarding any suggestions you'd give them for using this model in production
            - Quantify the uncertainty of the predictions through confidence intervals.
            - Use other tools in conjunction with the prediction app, such as reviewing individual applications to assess outcome reliability.
            - Train the users to understand how the model works and what data is being used.
        4. What two things would you do to improve this test? One line for each.
            - Implement a variety of machine learning models, including XGBoost and neural networks.
            - Cross validation was not used in this implementation. This would allow for better predictions on unseen data by helping to avoid overfitting.
        """)        
        
    # ------------- Data preprocessing ------------------------
    
    elif condition == 'Data preprocessing':
        
        st.title('Data preprocessing')
        
        st.subheader('Raw data')
        
        st.dataframe(df)
               
        st.subheader('Preprocessing steps')
        
        st.write("""
        Data was preprocessed using the following steps:
        - Remove rows which appear to have misentered data (e.g., negative numbers in the Account Year)
        - Set the data types
        - Subset where Account Year is equal to the year in Latest Accounts Date
        - Calculate the Years Since Incorporation (Account Year - Date of Incorporation)
        - Recode Trading Status to Default (previously Active) or Non-default (previously Dissolved or Liquidated/Receivership)
        - Fill NAs for the following using the equation in brackets:
            - Directors Remuneration (EBITDA + Directors Remuneration - EBITDA)
            - Total Assets (Total Current Assets + Total Non Current Assets)
            - Working Capital (Total Current Assets + Total Current Liabilities)
        - Drop Registered Number as it is unique for each company
        - Examine the correlation matrix and remove strongly correlated variables
        - Examine remaining features and drop those with a large portion of missing data or which are not adding value
        - Drop rows with NAs
        """)
        
        df_no_drop, df_final = preprocess_df(df)

        st.subheader('Correlation matrix')
        
        height, width, margin = 450, 1500, 10
        fig = plot_heatmap(corr=df_no_drop.corr(), height=700, margin=margin)

        st.plotly_chart(fig)
        
        st.write("""
        The correlation matrix shows some variables have strong correlations. 
        The following variables were removed as they are strongly correlated with one or more variable (> 0.90):
        - EBIT
        - Highest Paid Director
        - Total Non Current Liabilities (Incl Provisions)
        - Wages
        - Working Capital
        
        The following variables were removed either due to missing data or are deemed to be less informative:
        - Bank Overdraft
        - Bank Postcode
        - Capital Expenditure
        - Director Loans (current)
        - Director Loans (non-current)
        - EBITDA + Directors Remuneration
        - Latest Accounts Date
        - Leasehold
        - Profit Before Tax + Directors Remuneration
        - Registered or Trading Postcode
        - UK SIC Code
        """)
        
        st.subheader('Preprocessed data')
        
        st.dataframe(df_final)
        
        st.subheader('Histogram of features')
        
        select_var = st.selectbox('Select a variable', [i for i in df_final.columns])

        fig = plot_histogram(data=df_final, x=select_var, height=height, width=width, margin=margin)

        st.plotly_chart(fig)

        
        
    # ------------- The machine learning model ------------------------
    
    elif condition == 'The machine learning model':
        
        st.subheader('Machine learning approach')
                
        df_no_drop, df_final = preprocess_df(df)
        
        xdf = df_final.drop(['Trading Status'], axis=1)
        ydf = df_final['Trading Status']
        
        X = xdf[xdf.columns[rfe.get_support(1)]]
        y = ydf.to_numpy()
        
        st.write("Recursive feature elimination was used to select the top 5 features, the final features included in the model are:")
        for i in range(xdf.shape[1]):
            if rfe.support_[i]==True:
                st.write("- ", xdf.columns[i])
        
        st.subheader('Random forest model')
        
        st.write("A random forest model was used based on the literautre and time constraints. The model was trained on 80% of the data and tested on 20%.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        
#         st.text('Confusion matrix:\n ' + confusion_matrix(y_test, y_pred, labels=model.classes_))
        
        st.text('Model classification report:\n ' + classification_report(y_test, y_pred, target_names=model.classes_))
        
        st.subheader('Model metrics')
        
        y_pred_binary = np.where(y_pred == 'Default', 0, 1)
        y_test_binary = np.where(y_test == 'Default', 0, 1)
        
        accu = round(accuracy_score(y_test_binary, y_pred_binary),2)
        prec = round(precision_score(y_test_binary, y_pred_binary),2)
        reca = round(recall_score(y_test_binary, y_pred_binary),2)
        fsco = round(f1_score(y_test_binary, y_pred_binary),2)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", accu)
        col2.metric("Precision", prec)
        col3.metric("Recall", reca)
        col4.metric("F1 Score", fsco)
                
    
    
    # ------------- Make a prediction ------------------------------
    
    elif condition == 'Make a prediction':
        
        st.subheader('Make a prediction')
        st.write("Enter the relevant fields below to predict the probability of a company's loan defaulty")
        
        # following lines create boxes in which user can enter data required to make prediction 
        c_bank = st.number_input('Cash at the bank',0)
        profit = st.number_input('Profit for the year')
        r_earn = st.number_input('Retained earnings')
        t_asset = st.number_input('Total assets',0)
        t_equity = st.number_input('Total equity',0)
        result =""

        # when 'Predict' is clicked, make the prediction and store it 
        if st.button("Predict"): 
            res, res_prob = prediction(model, c_bank, profit, r_earn, t_asset, t_equity) 
            st.success(f'This company loan will default with a probability of {res_prob}%')
            
            with st.expander("Model details"):
                st.write(f"The model used was Random Forest.") 
                     


if __name__ == '__main__':
    main()
