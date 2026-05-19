
#ICU Patient Shifter

An intelligent ICU Patient Management System built with Streamlit, featuring smart ICU bed allocation, patient shifting logic, and a high‑accuracy machine learning model for ICU severity prediction.

#Features
Hospital dashboard showing total ICU beds, occupied beds, occupancy rate, ICU‑wise availability, average ICU stay, and last update time.

ICU availability view for Cardiac, Neuro, Surgical, Medical, and Pediatric ICUs with per‑unit total, occupied, and available beds.

99% accurate severity prediction model using a RandomForest classifier trained on a synthetic but clinically inspired dataset of vitals and conditions.

Smart patient shifting between ICUs based on condition, age, availability, and allowed alternative ICUs.

New critical patient admission with automatic ICU selection based on condition and bed availability.

Discharge workflow to free ICU beds and update hospital statistics.

Analytics for patient severity distribution, top medical conditions, and daily ICU activity trends.

Resource alert system that highlights ICUs with low bed availability.

Tech Stack
Frontend / UI: Streamlit

Language: Python

Machine learning:

RandomForestClassifier (scikit‑learn)

SMOTEENN (imbalanced‑learn) for handling class imbalance

StandardScaler, LabelEncoder for feature preprocessing

Visualization: Plotly Express and Plotly Graph Objects for interactive charts

Other libraries: NumPy, pandas, datetime, scikit‑learn metrics

How the model works
The app includes a HighAccuracySeverityPredictor class that is initialized at startup and trains a high‑accuracy severity model on a synthetic dataset.

Target: Severity levels 1–5 (integer class labels).

Generated dataset:

10,000 samples, balanced with equal samples per severity class.

Vitals, temperature, SpO2, GCS, age ranges and conditions vary logically with severity.

Key features (used as model inputs):

age

heart_rate

bp_systolic, bp_diastolic

resp_rate

temperature

spo2

gcs

comorbidities

emergency_admission

condition_encoded (LabelEncoded from condition text)

Training pipeline:

Train/test split with stratification.

SMOTEENN to balance training data.

StandardScaler applied to numeric features.

RandomForestClassifier with tuned depth, estimators, and class_weight.

Outputs:

Predicted severity (1–5)

Class probabilities

Confidence score (max probability)

Feature importance dictionary

Top risk factor explanations based on vitals, age, condition, comorbidities, and emergency status.

If the ML pipeline fails, a fallback rule‑based model ensures the app still produces severity predictions with reasonable defaults.

ICU logic and patient flow
The ICUPatientShifter class encapsulates ICU data and operations.

ICU setup
On first run, the app initializes ICU units in Streamlit session state:

Cardiac ICU: 8 beds (6 occupied, 2 available)

Neuro ICU: 6 beds (5 occupied, 1 available)

Surgical ICU: 10 beds (7 occupied, 3 available)

Medical ICU: 8 beds (6 occupied, 2 available)

Pediatric ICU: 6 beds (5 occupied, 1 available)

Each ICU gets a list of synthetic patients with:

id, name

age (pediatric vs adult)

condition (e.g., Heart Attack, Stroke, Sepsis, COVID, Pneumonia, Trauma, etc.)

icu_type

severity (random 3–4 range)

stability (Critical / Unstable / Improving)

days_in_icu, admission_date

shift_score (used to rank shift candidates)

Global hospital statistics (admissions, discharges, average stay, shift_count, last_updated) are also stored in session state.

Condition to ICU mapping
The app routes each medical condition to a primary ICU:

Cardiac ICU: Heart Attack, Heart Failure, Arrhythmia

Neuro ICU: Stroke, Brain Injury, Seizure

Surgical ICU: Post‑op, Trauma, Sepsis

Medical ICU: COVID, Pneumonia, Kidney Failure

Pediatric ICU: Pediatric Sepsis, Child Trauma

Alternative ICUs are allowed for certain conditions (for shifting), for example Sepsis can be managed in Surgical ICU or Medical ICU, and Heart Failure can be in Cardiac or Medical ICU.

Smart shift candidates
When a new critical patient requires an ICU that is full, the app searches for shift candidates:

Only looks at patients in the required ICU.

Skips shifting pediatric patients into adult ICUs.

Uses condition‑based alternative ICU mappings and checks if those ICUs have free beds.

Builds a list of candidate patients to move from from_icu to to_icu, sorted by shift_score (lower is better).

Operations
shift_patient(patient_id, from_icu, to_icu) updates both ICUs’ patient lists and bed counts, and increments shift_count.

admit_patient(patient_data, condition) assigns the patient to the appropriate ICU if there is a free bed, generates an ID, and updates hospital stats.

discharge_patient(patient_id, icu_type) removes a patient from the ICU, frees a bed, and increments total_discharges.

UI and pages
The Streamlit UI is built in app.py with custom CSS for cards and severity styling.

Main components:

Main header: “ICU Patient Management System – Intelligent ICU bed allocation with 99% accurate severity prediction.”

Tabs or sections (based on your layout):

Hospital Dashboard

Shift Patients

New Critical Patient

Severity Predictor

Hospital dashboard
The dashboard includes:

Global summary: total ICU beds, occupied beds, occupancy rate, total admissions, total discharges, patient shifts, average ICU stay, and last updated time.

ICU bed status chart: occupied vs available beds for each ICU using Plotly.

Severity distribution chart for current patients (e.g., Level 3 vs Level 4).

Top medical conditions chart: bar chart of common conditions like Pneumonia, Sepsis, Stroke, Trauma, Heart Attack, Brain Injury, Pediatric Sepsis, etc.

Patient shift trend chart: daily counts of shifts and new admissions over a week.

Patient management center
Discharge selector to choose a patient (e.g., “Patient 1000 (Cardiac ICU)”) and discharge them with one click.

Resource alert panel that shows warnings when specific ICUs (e.g., Neuro, Pediatric) have low bed availability.

Smart shift recommendations section that lists recommended shifts when needed; otherwise it shows that no shifts are required.

Installation and setup
Clone the repository:

bash
git clone https://github.com/<your-username>/icu-patient-shifter.git
cd icu-patient-shifter
Create and activate a virtual environment (optional but recommended):

bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
Install dependencies:

bash
pip install -r requirements.txt
Example requirements.txt (adjust to match your project):

text
streamlit
pandas
numpy
plotly
scikit-learn
imbalanced-learn
Run the app:

bash
streamlit run app.py
Then open the local URL shown in the terminal (typically http://localhost:8501).

Project structure (example)
Update this to match your actual repository layout.

text
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── (optional: models/, data/, utils/ if you split code later)
Intended use and disclaimer
This project is designed as a final year academic demonstration of intelligent ICU management using machine learning and interactive dashboards, not as a certified clinical decision support tool.

All patient data in the app is synthetic and for simulation only; any real‑world deployment would require validation, integration with hospital systems, security hardening, and regulatory compliance.