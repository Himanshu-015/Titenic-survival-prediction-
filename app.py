# ==============================================================================
# TITANIC WEB APPLICATION (FINAL COMPLETED & WORKING CODE)
# ==============================================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------------------
# 🚨 [अनिवार्य सुधार] कस्टम OutlierCapper क्लास जोड़ना ताकि मॉडल क्रैश न हो
# ------------------------------------------------------------------------------
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - (self.factor * iqr)
            self.upper_bounds_[col] = q3 + (self.factor * iqr)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = np.where(X_df[col] < self.lower_bounds_[col], self.lower_bounds_[col], X_df[col])
            X_df[col] = np.where(X_df[col] > self.upper_bounds_[col], self.upper_bounds_[col], X_df[col])
        return X_df.values

# 1. वेब ऐप का टाइटल और लुक सेट करें
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction Web App")
st.write("अपनी नई एक्सेल फ़ाइल अपलोड करें और यात्रियों के बचने की संभावना का पता लगाएं।")

# 2. बैकग्राउंड में सेव किए गए मॉडल और प्रीप्रोसेसर को लोड करें
@st.cache_resource
def load_ml_assets():
    folder_path = r"C:\Users\om\Desktop\titenic model"
    preprocessor = joblib.load(f"{folder_path}\\preprocessor_pipeline.joblib")
    model = joblib.load(f"{folder_path}\\final_classification_model.joblib")
    return preprocessor, model

try:
    loaded_preprocessor, loaded_model = load_ml_assets()
    st.success("✅ मशीन लर्निंग मॉडल सफलतापूर्वक लोड हो चुका है!")
except Exception as e:
    st.error(f"❌ मॉडल फाइल्स नहीं मिलीं! असली वजह: {e}")

# 3. फ़ाइल अपलोडर विजेट (File Uploader Widget)
uploaded_file = st.file_uploader("अपनी एक्सेल (.xlsx) फाइल यहाँ अपलोड करें", type=["xlsx"])

if uploaded_file is not None:
    new_df = pd.read_excel(uploaded_file)
    st.write("### 📊 अपलोड किया गया डेटा:", new_df.head())
    
    # प्रेडिक्शन बटन
    if st.button("🔮 Survival प्रेडिक्ट करें"):
        with st.spinner("⏳ मॉडल प्रेडिक्शन कर रहा है..."):
            try:
                # [सुधार] अगर एक्सेल फाइल में ये कॉलम्स गायब हैं, तो इन्हें खाली (None) बनाकर जोड़ दें 
                # ताकि मॉडल को उसके पुराने सीखे हुए सारे कॉलम्स वापस मिल जाएं
                required_missing_cols = ['Ticket', 'PassengerId', 'Name', 'Cabin']
                for col in required_missing_cols:
                    if col not in new_df.columns:
                        new_df[col] = None
                
                # प्रेडिक्शन से पहले 'Survived' कॉलम (अगर गलती से आ गया हो) तो ही हटाएं
                new_df_cleaned = new_df.drop(columns=['Survived'], errors='ignore')
                
                # डेटा ट्रांसफ़ॉर्मेशन और प्रेडिक्शन
                processed_data = loaded_preprocessor.transform(new_df_cleaned)
                predictions = loaded_model.predict(processed_data)
                probabilities = loaded_model.predict_proba(processed_data)[:, 1]
                
                # नतीजों को जोड़ना
                new_df['Status'] = ["Survived (जीवित) 🎉" if x == 1 else "Not Survived (मृत) ❌" for x in predictions]
                new_df['Survival_Probability (%)'] = np.round(probabilities * 100, 2)
                
                st.balloons() # स्क्रीन पर गुब्बारे उड़ाने के लिए!
                st.write("### 🏆 प्रेडिक्शन के नतीजे:")
                
                # सिर्फ काम के कॉलम्स स्क्रीन पर दिखाना
                st.dataframe(new_df[['Pclass', 'Sex', 'Age', 'Status', 'Survival_Probability (%)']])
                
                # डाउनलोड करने का विकल्प
                csv = new_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 प्रेडिक्शन रिपोर्ट डाउनलोड करें (CSV)", data=csv, file_name="titanic_predictions.csv", mime="text/csv")

            except Exception as error:
                st.error(f"गड़बड़: फाइल के कॉलम्स आपके मॉडल से मैच नहीं हो रहे हैं। त्रुटि विवरण: {error}")
