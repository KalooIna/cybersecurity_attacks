"""
Streamlit Web Application for Cyber Security Attack Type Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Cyber Security Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Cyber Security Attack Type Detection")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    """Load the trained model and preprocessor."""
    try:
        model = joblib.load('models/best_model.pkl')
        preprocessor = DataPreprocessor()
        preprocessor.load('models/preprocessor.pkl')
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ Models not found! Please run the training pipeline first.")
        st.info("Run: `python quick_analysis.py` to train the models.")
        return None, None

model, preprocessor = load_models()

if model is not None and preprocessor is not None:
    # Get class names
    class_names = preprocessor.target_encoder.classes_
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    input_method = st.sidebar.radio(
        "Select Input Method",
        ["📁 Upload CSV File", "✍️ Manual Input"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application predicts cybersecurity attack types based on network traffic data. "
        "Upload a CSV file or enter data manually to get predictions."
    )
    
    # Main content
    if input_method == "📁 Upload CSV File":
        st.header("Upload CSV File for Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with the same columns as the training data (excluding 'Attack Type')"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df_input = pd.read_csv(uploaded_file)
                
                st.success(f"✓ File uploaded successfully! ({df_input.shape[0]} rows, {df_input.shape[1]} columns)")
                
                # Display preview
                st.subheader("📋 Data Preview")
                st.dataframe(df_input.head(10), use_container_width=True)
                
                # Predict button
                if st.button("🔮 Predict Attack Types", type="primary"):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Preprocess
                            X = preprocessor.preprocess(df_input, fit=False)
                            
                            # Predict
                            predictions = model.predict(X)
                            probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                            
                            # Decode predictions
                            predicted_classes = preprocessor.target_encoder.inverse_transform(predictions)
                            
                            # Create results dataframe
                            results_df = df_input.copy()
                            results_df['Predicted Attack Type'] = predicted_classes
                            
                            if probabilities is not None:
                                # Add probability for predicted class
                                max_probs = np.max(probabilities, axis=1)
                                results_df['Prediction Confidence'] = max_probs
                                
                                # Add top 3 predictions with probabilities
                                top3_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
                                top3_predictions = []
                                top3_probs = []
                                
                                for i, indices in enumerate(top3_indices):
                                    preds = [class_names[idx] for idx in indices]
                                    probs = [probabilities[i][idx] for idx in indices]
                                    top3_predictions.append(", ".join([f"{p} ({pr:.2%})" for p, pr in zip(preds, probs)]))
                                    top3_probs.append(probs[0])
                                
                                results_df['Top 3 Predictions'] = top3_predictions
                            
                            # Display results
                            st.subheader("🎯 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Predictions", len(results_df))
                            with col2:
                                st.metric("Unique Attack Types", results_df['Predicted Attack Type'].nunique())
                            with col3:
                                most_common = results_df['Predicted Attack Type'].mode()[0]
                                st.metric("Most Common", most_common)
                            with col4:
                                if 'Prediction Confidence' in results_df.columns:
                                    avg_conf = results_df['Prediction Confidence'].mean()
                                    st.metric("Avg Confidence", f"{avg_conf:.2%}")
                            
                            # Distribution chart
                            st.subheader("📊 Attack Type Distribution")
                            attack_counts = results_df['Predicted Attack Type'].value_counts()
                            st.bar_chart(attack_counts)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="attack_predictions.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.info("Make sure your CSV file has the same columns as the training data (excluding 'Attack Type').")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    else:  # Manual Input
        st.header("Manual Data Input")
        st.markdown("Enter the values for each feature to get a prediction.")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            # Get feature columns (excluding target)
            feature_cols = preprocessor.feature_columns
            
            # Group features logically
            network_features = [col for col in feature_cols if any(x in col.lower() for x in ['ip', 'port', 'protocol', 'packet', 'traffic'])]
            security_features = [col for col in feature_cols if any(x in col.lower() for x in ['malware', 'anomaly', 'alert', 'severity', 'signature'])]
            system_features = [col for col in feature_cols if any(x in col.lower() for x in ['user', 'device', 'network', 'geo', 'proxy', 'firewall', 'ids', 'log'])]
            other_features = [col for col in feature_cols if col not in network_features + security_features + system_features]
            
            input_data = {}
            
            with col1:
                st.subheader("🌐 Network Features")
                for col in network_features[:len(network_features)//2 + 1]:
                    if preprocessor.categorical_columns and col in preprocessor.categorical_columns:
                        # For categorical, use text input
                        input_data[col] = st.text_input(col, value="")
                    else:
                        # For numerical, use number input
                        input_data[col] = st.number_input(col, value=0.0)
                
                st.subheader("🔒 Security Features")
                for col in security_features[:len(security_features)//2 + 1]:
                    if preprocessor.categorical_columns and col in preprocessor.categorical_columns:
                        input_data[col] = st.text_input(col, value="")
                    else:
                        input_data[col] = st.number_input(col, value=0.0)
            
            with col2:
                st.subheader("🌐 Network Features (cont.)")
                for col in network_features[len(network_features)//2 + 1:]:
                    if preprocessor.categorical_columns and col in preprocessor.categorical_columns:
                        input_data[col] = st.text_input(col, value="")
                    else:
                        input_data[col] = st.number_input(col, value=0.0)
                
                st.subheader("🔒 Security Features (cont.)")
                for col in security_features[len(security_features)//2 + 1:]:
                    if preprocessor.categorical_columns and col in preprocessor.categorical_columns:
                        input_data[col] = st.text_input(col, value="")
                    else:
                        input_data[col] = st.number_input(col, value=0.0)
            
            # System and other features
            if system_features or other_features:
                st.subheader("💻 System & Other Features")
                cols = st.columns(min(3, len(system_features + other_features)))
                all_other = system_features + other_features
                for idx, col in enumerate(all_other):
                    with cols[idx % len(cols)]:
                        if preprocessor.categorical_columns and col in preprocessor.categorical_columns:
                            input_data[col] = st.text_input(col, value="")
                        else:
                            input_data[col] = st.number_input(col, value=0.0)
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Attack Type", type="primary")
            
            if submitted:
                try:
                    # Create dataframe from input
                    df_input = pd.DataFrame([input_data])
                    
                    # Preprocess
                    X = preprocessor.preprocess(df_input, fit=False)
                    
                    # Predict
                    prediction = model.predict(X)[0]
                    predicted_class = preprocessor.target_encoder.inverse_transform([prediction])[0]
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)[0]
                        top3_indices = np.argsort(probabilities)[-3:][::-1]
                        top3_classes = [class_names[idx] for idx in top3_indices]
                        top3_probs = [probabilities[idx] for idx in top3_indices]
                    
                    # Display result
                    st.success("✅ Prediction Complete!")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.metric("🎯 Predicted Attack Type", predicted_class)
                        if hasattr(model, 'predict_proba'):
                            confidence = probabilities[prediction]
                            st.metric("📊 Confidence", f"{confidence:.2%}")
                    
                    with col2:
                        if hasattr(model, 'predict_proba'):
                            st.subheader("Top 3 Predictions")
                            for i, (cls, prob) in enumerate(zip(top3_classes, top3_probs), 1):
                                st.write(f"{i}. **{cls}**: {prob:.2%}")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.info("Please fill in all required fields with appropriate values.")

else:
    st.warning("⚠️ Please train the models first by running `python quick_analysis.py`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Cyber Security Attack Type Detection System | DSTI Project"
    "</div>",
    unsafe_allow_html=True
)

