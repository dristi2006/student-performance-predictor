import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎓 Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .error-box {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the student dataset"""
    try:
        # Try to load the dataset
        df = pd.read_csv('student_mat.csv')
        return df
    except FileNotFoundError:
        # If file not found, create sample data for demonstration
        st.warning("student_mat.csv not found. Using sample data for demonstration.")
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'age': np.random.randint(15, 22, n_samples),
            'sex': np.random.choice(['M', 'F'], n_samples),
            'studytime': np.random.randint(1, 5, n_samples),
            'failures': np.random.randint(0, 4, n_samples),
            'absences': np.random.randint(0, 30, n_samples),
            'G1': np.random.randint(0, 20, n_samples),
            'G2': np.random.randint(0, 20, n_samples),
            'G3': np.random.randint(0, 20, n_samples),
            'schoolsup': np.random.choice(['yes', 'no'], n_samples),
            'famsup': np.random.choice(['yes', 'no'], n_samples),
            'higher': np.random.choice(['yes', 'no'], n_samples),
            'internet': np.random.choice(['yes', 'no'], n_samples),
            'romantic': np.random.choice(['yes', 'no'], n_samples),
            'famrel': np.random.randint(1, 6, n_samples),
            'freetime': np.random.randint(1, 6, n_samples),
            'goout': np.random.randint(1, 6, n_samples),
            'health': np.random.randint(1, 6, n_samples),
        }
        
        df = pd.DataFrame(data)
        return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    df_processed = df.copy()
    
    # Create target variable (Pass/Fail based on G3 grade)
    df_processed['pass'] = (df_processed['G3'] >= 10).astype(int)
    
    # Handle categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        le_dict[col] = le
    
    return df_processed, le_dict

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return the best one"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    model_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Get the best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
    best_model = model_results[best_model_name]['model']
    
    return model_results, best_model, best_model_name

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Prediction", "📈 Model Performance"])
    
    # Load data
    df = load_data()
    df_processed, le_dict = preprocess_data(df)
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Student Performance Prediction System! 🎯
        
        This application uses machine learning to predict whether a student will pass or fail based on various academic and personal factors.
        
        ### 🚀 Features:
        - **Interactive Data Analysis**: Explore student data with beautiful visualizations
        - **Multiple ML Models**: Compare performance across different algorithms
        - **Real-time Predictions**: Make predictions for new students
        - **Comprehensive Metrics**: Detailed model performance analysis
        
        ### 📋 How to Use:
        1. **Data Analysis**: Explore the dataset and understand patterns
        2. **Model Training**: Train and compare different ML models
        3. **Prediction**: Make predictions for new students
        4. **Performance**: Analyze model accuracy and metrics
        
        ### 📊 Dataset Overview:
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{len(df)}</h3><p>Total Students</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{len(df.columns)}</h3><p>Features</p></div>', unsafe_allow_html=True)
        with col3:
            pass_rate = (df_processed['pass'].sum() / len(df_processed)) * 100
            st.markdown(f'<div class="metric-card"><h3>{pass_rate:.1f}%</h3><p>Pass Rate</p></div>', unsafe_allow_html=True)
        with col4:
            avg_grade = df['G3'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_grade:.1f}</h3><p>Avg Final Grade</p></div>', unsafe_allow_html=True)
        
        st.markdown("### 📈 Quick Data Preview")
        st.dataframe(df.head(), use_container_width=True)
    
    elif page == "📊 Data Analysis":
        st.header("📊 Exploratory Data Analysis")
        
        # Grade distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_grades = px.histogram(df, x='G3', bins=20, title='Final Grade Distribution',
                                    color_discrete_sequence=['#667eea'])
            fig_grades.update_layout(height=400)
            st.plotly_chart(fig_grades, use_container_width=True)
        
        with col2:
            pass_fail = df_processed['pass'].value_counts()
            fig_pass = px.pie(values=pass_fail.values, names=['Fail', 'Pass'], 
                             title='Pass/Fail Distribution',
                             color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
            fig_pass.update_layout(height=400)
            st.plotly_chart(fig_pass, use_container_width=True)
        
        # Age and study time analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age = px.box(df, x='pass' if 'pass' in df.columns else None, y='age', 
                            title='Age Distribution by Pass/Fail',
                            color='pass' if 'pass' in df.columns else None,
                            color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
            if 'pass' not in df.columns:
                fig_age = px.box(df, y='age', title='Age Distribution')
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            fig_study = px.histogram(df, x='studytime', title='Study Time Distribution',
                                   color_discrete_sequence=['#a8e6cf'])
            fig_study.update_layout(height=400)
            st.plotly_chart(fig_study, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔍 Feature Correlations")
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        corr_matrix = df_processed[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            title='Feature Correlation Matrix',
                            color_continuous_scale='RdYlBu_r')
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training & Comparison")
        
        # Prepare features
        feature_cols = [col for col in df_processed.columns if col not in ['G3', 'pass']]
        X = df_processed[feature_cols]
        y = df_processed['pass']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... Please wait!"):
                model_results, best_model, best_model_name = train_models(X_train, X_test, y_train, y_test)
                
                # Save the best model
                joblib.dump(best_model, 'best_model.pkl')
                joblib.dump(feature_cols, 'feature_cols.pkl')
                joblib.dump(le_dict, 'label_encoders.pkl')
                
                st.session_state['model_results'] = model_results
                st.session_state['best_model'] = best_model
                st.session_state['best_model_name'] = best_model_name
                st.session_state['feature_cols'] = feature_cols
                st.session_state['le_dict'] = le_dict
            
            st.markdown(f'<div class="success-box"><h3>✅ Training Complete!</h3><p>Best Model: {best_model_name}</p></div>', 
                       unsafe_allow_html=True)
        
        # Display results if available
        if 'model_results' in st.session_state:
            st.subheader("📊 Model Comparison")
            
            # Create comparison chart
            model_names = list(st.session_state['model_results'].keys())
            accuracies = [st.session_state['model_results'][name]['accuracy'] for name in model_names]
            
            fig_comparison = px.bar(x=model_names, y=accuracies, 
                                  title='Model Accuracy Comparison',
                                  color=accuracies,
                                  color_continuous_scale='viridis')
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Display metrics
            col1, col2 = st.columns(2)
            for i, (name, results) in enumerate(st.session_state['model_results'].items()):
                col = col1 if i % 2 == 0 else col2
                with col:
                    accuracy_color = "success-box" if results['accuracy'] > 0.8 else "warning-box" if results['accuracy'] > 0.7 else "error-box"
                    st.markdown(f'<div class="{accuracy_color}"><h4>{name}</h4><p>Accuracy: {results["accuracy"]:.3f}</p></div>', 
                               unsafe_allow_html=True)
    
    elif page == "🔮 Prediction":
        st.header("🔮 Make Predictions")
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Model Training' section!")
            return
        
        st.markdown("### 📝 Enter Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 15, 22, 17)
            sex = st.selectbox("Sex", ["M", "F"])
            studytime = st.slider("Study Time (hours per week)", 1, 4, 2)
            failures = st.slider("Number of Past Failures", 0, 3, 0)
            absences = st.slider("Number of Absences", 0, 30, 5)
        
        with col2:
            g1 = st.slider("First Period Grade (G1)", 0, 20, 10)
            g2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
            schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
            famsup = st.selectbox("Family Educational Support", ["yes", "no"])
            higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        
        col3, col4 = st.columns(2)
        with col3:
            internet = st.selectbox("Internet Access", ["yes", "no"])
            romantic = st.selectbox("In Romantic Relationship", ["yes", "no"])
            famrel = st.slider("Family Relationship Quality", 1, 5, 4)
        
        with col4:
            freetime = st.slider("Free Time", 1, 5, 3)
            goout = st.slider("Going Out with Friends", 1, 5, 3)
            health = st.slider("Health Status", 1, 5, 3)
        
        if st.button("🎯 Predict Performance", type="primary"):
            # Prepare input data
            input_data = {
                'age': age,
                'sex': sex,
                'studytime': studytime,
                'failures': failures,
                'absences': absences,
                'G1': g1,
                'G2': g2,
                'schoolsup': schoolsup,
                'famsup': famsup,
                'higher': higher,
                'internet': internet,
                'romantic': romantic,
                'famrel': famrel,
                'freetime': freetime,
                'goout': goout,
                'health': health
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in input_df.select_dtypes(include=['object']).columns:
                if col in st.session_state['le_dict']:
                    le = st.session_state['le_dict'][col]
                    input_df[col] = le.transform(input_df[col])
            
            # Make prediction
            prediction = st.session_state['best_model'].predict(input_df[st.session_state['feature_cols']])[0]
            probability = st.session_state['best_model'].predict_proba(input_df[st.session_state['feature_cols']])[0]
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown('<div class="success-box"><h3>🎉 PASS</h3><p>Student is likely to pass!</p></div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box"><h3>❌ FAIL</h3><p>Student is at risk of failing!</p></div>', 
                               unsafe_allow_html=True)
            
            with col2:
                fig_prob = px.bar(x=['Fail', 'Pass'], y=probability, 
                                 title='Prediction Probability',
                                 color=probability,
                                 color_continuous_scale=['red', 'green'])
                fig_prob.update_layout(height=300)
                st.plotly_chart(fig_prob, use_container_width=True)
    
    elif page == "📈 Model Performance":
        st.header("📈 Model Performance Analysis")
        
        if 'model_results' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Model Training' section!")
            return
        
        # Prepare test data
        feature_cols = [col for col in df_processed.columns if col not in ['G3', 'pass']]
        X = df_processed[feature_cols]
        y = df_processed['pass']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Best model performance
        best_model_name = st.session_state['best_model_name']
        st.markdown(f"### 🏆 Best Model: {best_model_name}")
        
        y_pred = st.session_state['model_results'][best_model_name]['predictions']
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(cm, 
                          text_auto=True, 
                          aspect="auto",
                          title='Confusion Matrix',
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Fail', 'Pass'],
                          y=['Fail', 'Pass'])
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification Report
        st.subheader("📊 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{report["accuracy"]:.3f}</h3><p>Overall Accuracy</p></div>', 
                       unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h3>{report["macro avg"]["precision"]:.3f}</h3><p>Precision</p></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{report["macro avg"]["recall"]:.3f}</h3><p>Recall</p></div>', 
                       unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h3>{report["macro avg"]["f1-score"]:.3f}</h3><p>F1-Score</p></div>', 
                       unsafe_allow_html=True)
        
        # Feature Importance (if available)
        if hasattr(st.session_state['best_model'], 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            feature_importance = pd.DataFrame({
                'feature': st.session_state['feature_cols'],
                'importance': st.session_state['best_model'].feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(feature_importance, 
                                   x='importance', 
                                   y='feature',
                                   orientation='h',
                                   title='Feature Importance',
                                   color='importance',
                                   color_continuous_scale='viridis')
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)

if __name__ == "__main__":
    main()