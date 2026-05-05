import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="wide", page_icon="🔮")

@st.cache_resource
def load_model():
    return joblib.load('model/churn_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('data/telco_churn.csv')
    # TotalCharges ko numeric banao, empty ko 0 se fill karo
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

model = load_model()
df_orig = load_data()
X_orig = df_orig.drop(['customerID', 'Churn'], axis=1)

st.title("Customer Churn Predictor 🔮")
st.write("Customer ki details daalo aur turant prediction pao")

st.sidebar.header("Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', [0, 1])
    Partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    Dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.sidebar.slider('Tenure Months', 0, 72, 12)
    PhoneService = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    StreamingTV = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.sidebar.number_input('Monthly Charges', 18.0, 120.0, 70.0)
    TotalCharges = st.sidebar.number_input('Total Charges', 0.0, 9000.0, tenure * MonthlyCharges)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    return pd.DataFrame([data])

input_df = user_input_features()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Selected Customer Data')
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader('Prediction')
    if st.button('Predict Churn', type="primary", use_container_width=True):
        # Preprocessing: Training jaisa hi karo
        df_input = input_df.copy()
        
        # 1. TotalCharges numeric karo
        df_input['TotalCharges'] = pd.to_numeric(df_input['TotalCharges'], errors='coerce').fillna(0)
        
        # 2. Saare categorical columns ko encode karo - training jaisa
        cat_cols = df_input.select_dtypes(include='object').columns.tolist()
        
        for col in cat_cols:
            # Har column ke liye training data se mapping banao
            mapping = {val: idx for idx, val in enumerate(X_orig[col].unique())}
            df_input[col] = df_input[col].map(mapping)
        
        # 3. Column order same rakho
        df_input = df_input[X_orig.columns]
        
        # 4. Predict
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)
        
        churn_prob = probability[0][1] * 100
        
        if prediction[0] == 1:
            st.error(f'⚠️ High Churn Risk')
            st.metric(label="Churn Probability", value=f"{churn_prob:.1f}%")
            st.write('**Recommendation:** Retention call karo, discount offer do')
        else:
            st.success(f'✅ Low Churn Risk')
            st.metric(label="Stay Probability", value=f"{100-churn_prob:.1f}%")
            st.write('**Recommendation:** Loyal customer, upsell try karo')
    else:
        st.info('Details bharo aur Predict button dabao')

st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn | Accuracy: ~79%")