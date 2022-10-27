import pandas as pd
import numpy as np
import pickle
import streamlit as st



df = pd.read_csv('Technical_Task_Dataset.csv')
model = pickle.load(open('classification.sav', 'rb'))

    
def prediction(c_exp, c_bank, ebitda, e_remun, profit, r_earn, t_asset, t_equity):
    prediction = model.predict([[c_exp, c_bank, ebitda, e_remun, profit, r_earn, t_asset, t_equity]])
    prediction_prob = model.predict_proba([[c_exp, c_bank, ebitda, e_remun, profit, r_earn, t_asset, t_equity]])
     
    if prediction == 'Default':
        pred = 'default'
        pred_prob = prediction_prob[0][0]*100
    else:
        pred = 'not default'
        pred_prob = prediction_prob[0][1]*100
    
    return pred, pred_prob


def main():
    st.title("Loan Default Prediction App")
    st.write("A machine learning app to predict the proability of a company's loan defaulting")
      
    # following lines create boxes in which user can enter data required to make prediction 
    c_exp = st.number_input('Capital expenditure')
    c_bank = st.number_input('Cash at the bank',0)
    ebitda = st.number_input('EBITDA')
    e_remun = st.number_input('Employees remuneration',0)
    profit = st.number_input('Profit for the year')
    r_earn = st.number_input('Retained earnings')
    t_asset = st.number_input('Total assets',0)
    t_equity = st.number_input('Total equity',0)
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        res, res_prob = prediction(c_exp, c_bank, ebitda, e_remun, profit, r_earn, t_asset, t_equity) 
        st.success(f'This loan will {res} with a probability of {res_prob}%')


if __name__ == '__main__':
    main()
