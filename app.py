import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

# ----------------------------
# Load dataset function
# ----------------------------
@st.cache_data
def load_data(file_path="student_mat.csv"):
    df = pd.read_csv(file_path, sep=";")
    df.columns = df.columns.str.strip().str.replace('"', '')
    return df

# ----------------------------
# Load default or uploaded CSV
# ----------------------------
st.sidebar.header("📂 Dataset Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=";")
    data.columns = data.columns.str.strip().str.replace('"', '')
else:
    data = load_data()

if "G3" not in data.columns:
    st.error("❌ Dataset must contain a 'G3' column for final grade.")
    st.stop()

# ----------------------------
# Add Pass/Fail target
# ----------------------------
data["pass"] = data["G3"].apply(lambda x: 1 if x >= 10 else 0)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filter Data")
sex_filter = st.sidebar.multiselect("Select Gender", options=data["sex"].unique(), default=data["sex"].unique())
school_filter = st.sidebar.multiselect("Select School", options=data["school"].unique(), default=data["school"].unique())

filtered_data = data[(data["sex"].isin(sex_filter)) & (data["school"].isin(school_filter))]

# ----------------------------
# Show Dataset
# ----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(filtered_data.head())

# ----------------------------
# Plotly Visualization
# ----------------------------
st.subheader("📈 Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_data, x="G3", color="sex", barmode="overlay", title="Grade Distribution by Gender")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(filtered_data, x="G1", y="G3", color="pass", title="G1 vs G3 with Pass/Fail")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Train Model
# ----------------------------
X = filtered_data.drop(columns=["pass", "G3"])
X = pd.get_dummies(X, drop_first=True)  # Encode categorical vars
y = filtered_data["pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.sidebar.write(f"✅ Model Accuracy: {acc:.2f}")
st.sidebar.write("📄 Classification Report:")
st.sidebar.text(classification_report(y_test, y_pred))

# ----------------------------
# Manual Prediction Form
# ----------------------------
st.subheader("📝 Predict Pass/Fail from User Input")

with st.form("student_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=15, max_value=22, value=17)
        absences = st.number_input("Absences", min_value=0, max_value=50, value=2)
        G1 = st.number_input("G1 Grade", min_value=0, max_value=20, value=10)
    with col2:
        G2 = st.number_input("G2 Grade", min_value=0, max_value=20, value=10)
        health = st.slider("Health (1-5)", 1, 5, 3)
        sex = st.selectbox("Gender", options=["F", "M"])

    submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        # Prepare one-row dataframe for prediction
        input_df = pd.DataFrame({
            "age": [age],
            "absences": [absences],
            "G1": [G1],
            "G2": [G2],
            "health": [health],
            "sex": [sex],
            # Fill with defaults for other categorical vars
            "school": [data["school"].mode()[0]],
            "address": [data["address"].mode()[0]],
            "famsize": [data["famsize"].mode()[0]],
            "Pstatus": [data["Pstatus"].mode()[0]],
            "Medu": [data["Medu"].mode()[0]],
            "Fedu": [data["Fedu"].mode()[0]],
            "Mjob": [data["Mjob"].mode()[0]],
            "Fjob": [data["Fjob"].mode()[0]],
            "reason": [data["reason"].mode()[0]],
            "guardian": [data["guardian"].mode()[0]],
            "traveltime": [data["traveltime"].mode()[0]],
            "studytime": [data["studytime"].mode()[0]],
            "failures": [data["failures"].mode()[0]],
            "schoolsup": [data["schoolsup"].mode()[0]],
            "famsup": [data["famsup"].mode()[0]],
            "paid": [data["paid"].mode()[0]],
            "activities": [data["activities"].mode()[0]],
            "nursery": [data["nursery"].mode()[0]],
            "higher": [data["higher"].mode()[0]],
            "internet": [data["internet"].mode()[0]],
            "romantic": [data["romantic"].mode()[0]],
            "famrel": [data["famrel"].mode()[0]],
            "freetime": [data["freetime"].mode()[0]],
            "goout": [data["goout"].mode()[0]],
            "Dalc": [data["Dalc"].mode()[0]],
            "Walc": [data["Walc"].mode()[0]]
        })

        # Match training columns
        input_df = pd.get_dummies(input_df, drop_first=True)
        missing_cols = set(X.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X.columns]

        prediction = model.predict(input_df)[0]
        result = "✅ Pass" if prediction == 1 else "❌ Fail"
        st.success(f"Prediction: {result}")
