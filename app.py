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
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(layout='wide')
st.title('🤖CSV Data Processing and Prediction')
tab1,tab2,tab3=st.tabs(['⚙️Processing','📊Analysis & Visualizations','🤖Prediction'])
with tab1:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        df = pd.read_csv(uploaded_file)
        st.session_state.df_original = df
        st.write("Data Preview:")
        st.dataframe(df.head())

    if "uploaded_file" in st.session_state:
        df = st.session_state.df_original

        target_column = st.selectbox("Select the target column", df.columns)

        if st.button('Process File'):
            for column in df.columns:
                if any(keyword in column.lower() for keyword in ['name', 'ticket', 'id', 'date', 'unnamed']):
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
            x_train_df = pd.DataFrame(x_train_sc)
            x_test_df = pd.DataFrame(x_test_sc)

            model = LogisticRegression()
            model.fit(x_train_sc, y_train)

            st.session_state.df = df
            st.session_state.df_copy = df_copy
            st.session_state.target_column = target_column
            st.session_state.encoder = encoder
            st.session_state.pt = pt
            st.session_state.scaler = scaler
            st.session_state.model = model

            st.session_state.processed_csv = df.to_csv(index=False).encode('utf-8')
            st.session_state.train_csv = x_train_df.to_csv(index=False).encode('utf-8')
            st.session_state.test_csv = x_test_df.to_csv(index=False).encode('utf-8')

            st.success('Processing and training completed!')


        if "processed_csv" in st.session_state:
            st.download_button(
                label="Download Preprocessed Data",
                data=st.session_state.processed_csv,
                file_name='preprocessed_data.csv',
                mime='text/csv'
            )

            st.download_button(
                label="Download Train Data",
                data=st.session_state.train_csv,
                file_name='train_data.csv',
                mime='text/csv'
            )

            st.download_button(
                label="Download Test Data",
                data=st.session_state.test_csv,
                file_name='test_data.csv',
                mime='text/csv'
            )
            with open('model.pkl', 'wb') as model_file:
                pkl.dump(st.session_state.model, model_file)
            with open('model.pkl', 'rb') as model_file:
                st.download_button(
                    label="Download Model",
                    data=model_file,
                    file_name='model.pkl',
                    mime='application/octet-stream'
                )

with tab2:
    if 'df' in st.session_state: 
        df = st.session_state.df

        def dataset_summary(data):
            return {
                "Shape": data.shape,
                "Missing Values": data.isnull().sum().to_dict(),
                "Data Types": data.dtypes.astype(str).to_dict(),
            }

        def plot_categorical(data, column):
            value_counts = data[column].value_counts().reset_index()
            value_counts.columns = [column, "Count"] 
            fig = px.bar(
                value_counts,
                x=column,
                y="Count",
                labels={column: column, "Count": "Count"},
                title=f"Distribution of {column}",
            )
            return fig

        def plot_numerical(data, column):
            fig = px.histogram(
                data,
                x=column,
                nbins=30,
                marginal="box",
                title=f"Distribution of {column}",
                template="plotly_white",
            )
            return fig

        def plot_correlation(data):
            corr_matrix = data.corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="Viridis",
                    zmin=-1,
                    zmax=1,
                )
            )
            fig.update_layout(title="Correlation Heatmap")
            return fig

        st.write("Dataset Summary:")
        summary = dataset_summary(df)
        st.json(summary)

        st.write("Visualizations:")
        for column in df.columns:
            try:
                if df[column].dtype == "object" or df[column].nunique() < 20:
                    st.write(f"Categorical Column: {column}")
                    fig = plot_categorical(df, column)
                else:
                    st.write(f"Numerical Column: {column}")
                    fig = plot_numerical(df, column)
                st.plotly_chart(fig)
            except Exception as e:
                st.warning(f"Could not plot column `{column}`: {str(e)}")

        numerical_data = df.select_dtypes(include=["float64", "int64"])
        if not numerical_data.empty:
            st.write("Correlation Heatmap:")
            heatmap_fig = plot_correlation(numerical_data)
            st.plotly_chart(heatmap_fig)
        else:
            st.warning("No numerical data available for correlation heatmap.")
    else:
        st.warning("No data available. Please upload and process the data in the 'Processing' tab first.")
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []

with tab3:
    if 'df_copy' in st.session_state:
        input_features = st.session_state.df_copy.drop(st.session_state.target_column, axis=1).columns
        cat_features = st.session_state.df_copy.drop(st.session_state.target_column, axis=1).select_dtypes(include='object').columns
        
        inputs = {}
        
        for feature in input_features:
            if feature in cat_features:
                unique_values = st.session_state.df_copy[feature].unique().tolist()
                inputs[feature] = st.selectbox(feature, options=unique_values)
            else:
                if pd.api.types.is_float_dtype(st.session_state.df_copy[feature]):
                    inputs[feature] = st.number_input(feature, step=0.1, format='%.2f')
                else:
                    inputs[feature] = st.number_input(feature, step=1)
        
        if st.button('Predict'):
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
                
                input_with_prediction = inputs.copy()
                input_with_prediction['Prediction'] = y_pred[0]
                st.session_state.user_inputs.append(input_with_prediction)
        
        if st.session_state.user_inputs:
            user_inputs_df = pd.DataFrame(st.session_state.user_inputs)
            csv = user_inputs_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download New Data as CSV",
                data=csv,
                file_name='user_inputs.csv',
                mime='text/csv'
            )
