import numpy as np
import pandas as pd
from AutoClean import AutoClean
from sklearn.preprocessing import LabelEncoder, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import BorderlineSMOTE
import time
import streamlit as st
import pandas.api.types as pytype
st.set_page_config(layout='wide')
st.title('CSV Data Processing and Prediction')
tab1,tab2=st.tabs(['Processing','Prediction'])
with tab1:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        target_column = st.selectbox("Select the target column", df.columns)

        if st.button('Process File'):
            for column in df.columns:
                if any(keyword in column.lower() for keyword in ['name', 'id', 'ticket', 'date', 'unnamed']):
                    df.drop(column, axis=1, inplace=True)

            df = AutoClean(df, mode='manual', duplicates='auto', missing_num='knn', missing_categ='knn', outliers='winz')
            df = df.output
            df.drop_duplicates(inplace=True)

            df.dropna(axis=1, inplace=True)

            df_copy = df.copy()

            cat_feature = df.select_dtypes(include=['object']).columns
            encoder = {}
            
            for cat in cat_feature:
                le = LabelEncoder()
                df[cat] = le.fit_transform(df[cat])
                encoder[cat] = le

            x = df.drop(target_column, axis=1)
            y = df[target_column]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            value_counts = y.value_counts()
            threshold = 0.4
            if (value_counts[0] / value_counts.sum() < threshold) or (value_counts[1] / value_counts.sum() < threshold):
                smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)
                x_train, y_train = smote.fit_resample(x_train, y_train)

            pt = PowerTransformer(method='yeo-johnson')
            x_train_sk = pt.fit_transform(x_train)
            x_test_sk = pt.transform(x_test)

            scaler = RobustScaler()
            x_train_sc = scaler.fit_transform(x_train_sk)
            x_test_sc = scaler.transform(x_test_sk)

            model = LogisticRegression()
            model.fit(x_train_sc, y_train)

            st.session_state.df = df
            st.session_state.df_copy = df_copy
            st.session_state.encoder = encoder
            st.session_state.model = model
            st.session_state.pt = pt
            st.session_state.scaler = scaler
            st.session_state.cat_feature = cat_feature
            st.session_state.target_column = target_column
            st.success('Processing and training completed!')
with tab2:
    if 'df_copy' in st.session_state:
        input_features = st.session_state.df_copy.drop(st.session_state.target_column, axis=1).columns
        cat_features = st.session_state.df_copy.drop(st.session_state.target_column, axis=1).select_dtypes(include='object').columns
        inputs = {}
        for feature in input_features:
            if feature in cat_features:
                unique_values = st.session_state.df_copy[feature].unique().tolist()
                inputs[feature] = st.selectbox(feature, options=unique_values)
            else:
                if pytype.is_float_dtype(df[feature]):
                    inputs[feature] = st.number_input(feature,step=0.1,format='%.2f')
                else:
                    inputs[feature] = st.number_input(feature,step=1)

        if st.button('Logistic'):
            with st.spinner('Making prediction...'):
                time.sleep(0.8)
                features = []
                for feature in input_features:
                    value = inputs[feature]
                    if feature in cat_features:
                        value = st.session_state.encoder[feature].transform([value])[0]
                    features.append(value)

                features = np.array(features).reshape(1, -1)
                feature_scaled = st.session_state.pt.transform(features)
                feature_scaled = st.session_state.scaler.transform(feature_scaled)

                y_pred = st.session_state.model.predict(feature_scaled)
                
                if y_pred == 1:
                    st.success(st.session_state.target_column)
                else:
                    st.error(f'Not {st.session_state.target_column}')
