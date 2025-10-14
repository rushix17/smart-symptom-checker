import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import ast

# --- Dataset ---
data = {
    'Fever':         [1,1,0,1,1,0,0,1,1,0],
    'Cough':         [1,0,0,1,1,1,0,1,0,1],
    'Headache':      [1,1,0,0,1,1,0,1,0,0],
    'Nausea':        [0,1,1,1,0,1,0,0,0,1],
    'Fatigue':       [1,1,0,1,1,1,0,1,0,1],
    'SoreThroat':    [1,0,0,1,1,0,0,1,1,0],
    'MusclePain':    [1,1,0,1,0,1,0,0,1,0],
    'Diarrhea':      [0,0,1,0,0,1,1,0,0,1],
    'ShortnessBreath':[0,1,0,1,1,0,0,1,0,0],
    'Disease': [
        'Flu', 'Dengue', 'Food Poisoning', 'COVID-19', 'Common Cold',
        'Malaria', 'Migraine', 'Asthma', 'Allergy', 'Stomach Infection'
    ]
}

df = pd.DataFrame(data)
feature_columns = [col for col in df.columns if col != 'Disease']

# Train model
X = df[feature_columns]
y = df['Disease']
model = DecisionTreeClassifier()
model.fit(X, y)

# SQLite DB setup
conn = sqlite3.connect('symptom_checker.db', check_same_thread=False)
c = conn.cursor()

# Add new columns if missing (safe schema update)
try:
    c.execute("ALTER TABLE predictions ADD COLUMN patient_name TEXT")
except sqlite3.OperationalError:
    pass

try:
    c.execute("ALTER TABLE predictions ADD COLUMN age INTEGER")
except sqlite3.OperationalError:
    pass

try:
    c.execute("ALTER TABLE predictions ADD COLUMN contact_number TEXT")
except sqlite3.OperationalError:
    pass

# Create table if not exists
c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    patient_name TEXT,
    age INTEGER,
    contact_number TEXT,
    symptoms TEXT,
    predicted_disease TEXT,
    probabilities TEXT
)
''')
conn.commit()

# Symptom emoji mapping
symptom_emojis = {
    'Fever': '🤒',
    'Cough': '🤧',
    'Headache': '🤕',
    'Nausea': '🤢',
    'Fatigue': '😴',
    'SoreThroat': '😷',
    'MusclePain': '💪',
    'Diarrhea': '🚽',
    'ShortnessBreath': '😤',
}

# Streamlit UI
st.set_page_config(page_title="Smart Symptom Checker", page_icon="🩺", layout="centered")

st.title("🩺 Smart Symptom Checker")
st.write("Enter your personal details and select symptoms to get disease predictions.")

# Sidebar for patient details and symptoms
st.sidebar.header("Patient Details")
patient_name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
contact_number = st.sidebar.text_input("Contact Number")

st.sidebar.header("Select Your Symptoms")
symptoms_input = {}
for symptom in feature_columns:
    emoji = symptom_emojis.get(symptom, '')
    symptoms_input[symptom] = st.sidebar.checkbox(f"{emoji} {symptom}")

user_df = pd.DataFrame([{k: int(v) for k, v in symptoms_input.items()}])

if st.button("Check Disease Prediction"):

    # Validate required patient info
    if not patient_name.strip():
        st.error("Please enter the patient name.")
    elif not contact_number.strip():
        st.error("Please enter the contact number.")
    else:
        with st.spinner("Analyzing your symptoms..."):
            probs = model.predict_proba(user_df)[0]
            disease_probs = dict(zip(model.classes_, probs))
            sorted_diseases = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)

        with st.expander("Your Symptoms Summary", expanded=True):
            for sym, val in symptoms_input.items():
                status = "✅ Yes" if val else "❌ No"
                st.write(f"**{sym}:** {status}")

        with st.expander("Top 3 Predicted Diseases", expanded=True):
            for i, (disease, prob) in enumerate(sorted_diseases[:3], start=1):
                st.write(f"{i}. ✅ **{disease}** (Probability: {prob*100:.1f}%)")

        emergency_diseases = ['COVID-19', 'Dengue', 'Malaria', 'Asthma']
        top_disease = sorted_diseases[0][0]
        if top_disease in emergency_diseases:
            st.error("🚨 Suggestion: Please visit an Emergency or Specialist immediately!")
        else:
            st.info("🩻 Suggestion: You may consider visiting a General Physician.")

        st.markdown("### Disease Probability Chart")
        fig, ax = plt.subplots(figsize=(8, 5))
        diseases = [d for d, p in sorted_diseases]
        probabilities = [p*100 for d, p in sorted_diseases]
        ax.barh(diseases, probabilities, color='#0a9396')
        ax.set_xlabel("Probability (%)")
        ax.set_title("Disease Probabilities Based on Your Symptoms")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Save prediction in DB
        c.execute('''
            INSERT INTO predictions (timestamp, patient_name, age, contact_number, symptoms, predicted_disease, probabilities)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_name.strip(),
            int(age),
            contact_number.strip(),
            str(symptoms_input),
            top_disease,
            str(disease_probs)
        ))
        conn.commit()

else:
    st.info("Fill in your details, select symptoms from the sidebar, and click the button above.")

# Show recent predictions with Delete buttons
with st.expander("Show recent predictions"):

    c.execute('''
        SELECT id, timestamp, patient_name, age, contact_number, symptoms, predicted_disease 
        FROM predictions ORDER BY id DESC LIMIT 10
    ''')
    rows = c.fetchall()

    records = []
    for row in rows:
        id_, timestamp, patient_name, age, contact_number, symptoms_str, predicted = row
        try:
            symptoms_dict = ast.literal_eval(symptoms_str)
        except:
            symptoms_dict = {}

        symptom_status = {sym: ("Yes" if symptoms_dict.get(sym, False) else "No") for sym in feature_columns}

        record = {
            "ID": id_,
            "Timestamp": timestamp,
            "Patient Name": patient_name,
            "Age": age,
            "Contact Number": contact_number,
            **symptom_status,
            "Predicted Disease": predicted
        }
        records.append(record)

    if records:
        df_recent = pd.DataFrame(records)

        # Custom table with Delete button per row
        st.write("### Recent Predictions:")
        for idx, row in df_recent.iterrows():
            cols = st.columns([1.5, 2, 1, 1.5, 2, 2, 3])  # Adjust columns width as needed

            # Display basic info
            cols[0].write(row['Timestamp'])
            cols[1].write(row['Patient Name'])
            cols[2].write(row['Age'])
            cols[3].write(row['Contact Number'])
            cols[4].write(row['Predicted Disease'])

            # Display symptom summary concisely
            symptom_summary = ", ".join([f"{sym}: {row[sym]}" for sym in feature_columns if row[sym] == "Yes"])
            cols[5].write(symptom_summary if symptom_summary else "No symptoms")

            # Delete button
            if cols[6].button("Delete", key=f"delete_{row['ID']}"):
                c.execute("DELETE FROM predictions WHERE id=?", (row['ID'],))
                conn.commit()
                st.experimental_rerun()

    else:
        st.write("No recent predictions found.")

# New section: Disease Distribution Pie Chart
with st.expander("Disease Distribution Analysis"):

    c.execute("SELECT predicted_disease, COUNT(*) FROM predictions GROUP BY predicted_disease")
    data = c.fetchall()

    if data:
        diseases, counts = zip(*data)
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=diseases, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        ax2.axis('equal')  # Equal aspect ratio ensures pie is circular.
        st.pyplot(fig2)
    else:
        st.write("No data available to show disease distribution.")
