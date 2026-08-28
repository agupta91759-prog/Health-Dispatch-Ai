# Health-Dispatch-Ai

### AI-Powered Patient No-Show Risk Prediction & Mobile Healthcare Dispatch Decision Support

Health Dispatch AI is a machine learning application designed to help mobile healthcare and home-visit operations identify appointments with a high risk of patient no-shows.

The system combines **machine learning, explainable AI, batch prediction, and financial decision logic** to help dispatchers prioritize which appointments may require additional intervention.

---

## 🎯 Problem Statement

Patient no-shows can create significant operational costs for mobile healthcare providers.

A missed appointment can result in:

- Wasted technician travel time
- Fuel and transportation costs
- Lost appointment capacity
- Reduced workforce utilization
- Additional administrative effort

The objective of this project is to predict the probability of a patient no-show before dispatch and translate that prediction into an actionable operational recommendation.

---

## 💡 Solution

The application uses a **Random Forest Classifier** to estimate the probability that a patient will miss their appointment.

The predicted probability is then combined with an estimated dispatch cost to calculate the expected financial loss.

### Expected Loss

text
Expected Loss = No-Show Probability × Cost of Wasted Dispatch


For example:

text
No-Show Probability = 50%
Dispatch Cost       = ₹800

Expected Loss = 0.50 × ₹800
              = ₹400


This allows the system to consider both **ML risk and business impact** instead of relying only on a probability score.



# 🧠 Machine Learning Pipeline

text
Appointment Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Random Forest Classifier
        ↓
No-Show Probability
        ↓
Expected Financial Loss
        ↓
Operational Recommendation


## 🔍 Features

The model uses appointment and patient attributes such as:

* Age
* Lead time between scheduling and appointment
* Technician drive time
* Scholarship / financial aid
* Hypertension
* Diabetes
* Alcoholism
* Handicap level
* SMS reminder status

### Feature Engineering

The project derives:

**Lead_Time_Days**

from the difference between the scheduled date and appointment date.

A simulated:

**Drive_Time_Mins**

feature is also used to represent technician travel time in the mobile healthcare scenario.



# 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** from Scikit-learn.

The model configuration includes:

* n_estimators = 100
* max_depth = 10
* class_weight = balanced
* random_state = 42

The dataset is divided into:

* **80% training data**
* **20% testing data**

The model produces probability estimates using the classifier's probability prediction capability.

The trained model is serialized using **Joblib** and stored as:

text
model.pkl


This allows the application to load the trained model without retraining it every time the dashboard starts.

---

# 🔬 Explainable AI with SHAP

Machine learning predictions can be difficult to interpret.

To make individual predictions easier to understand, the application integrates **SHAP (SHapley Additive exPlanations)**.

SHAP helps identify which features contributed to the predicted no-show risk.

For example, features such as:

* Longer lead time
* Longer technician travel time
* Previous reminder status

can influence the model's prediction.

This provides a more transparent view of why a particular appointment received a high or low risk score.

---

# 💰 Business Decision Layer

The project goes beyond simply predicting:

> "Will this patient show up?"

It also asks:

> "Is it economically worthwhile to intervene?"

The system estimates financial exposure using:

text
Expected Loss
=
Predicted No-Show Probability
×
Estimated Dispatch Cost


The expected loss is then compared against the estimated cost of additional intervention.

This creates operational recommendations such as:

### 🟢 AUTO

Normal automated reminder is sufficient.

### 🟡 SEND SMS

Additional confirmation/reminder is recommended.

### 🔴 CALL REQ

The expected loss is high enough to justify manual phone intervention.

---

# 📅 Weekly Dispatch Command Center

The application includes a batch-processing workflow for a weekly appointment roster.

The dashboard can:

1. Generate a simulated appointment roster
2. Score multiple appointments
3. Calculate no-show probabilities
4. Estimate expected financial loss
5. Prioritize higher-risk appointments
6. Generate recommended dispatcher actions

This demonstrates how an ML model can be integrated into an operational workflow rather than being used only for individual predictions.

---

# 🖥️ Streamlit Dashboard

The project is deployed as an interactive **Streamlit** application.

The dashboard provides two primary workflows:

### 1. Single Patient Analysis

A dispatcher can enter patient and appointment information and receive:

* No-show probability
* Expected financial loss
* Recommended action
* SHAP-based explanation

### 2. Weekly Dispatch Analysis

The system performs batch inference across multiple appointments and helps prioritize high-risk cases.

---

# 🛠️ Technology Stack

| Technology    | Purpose                   |
| ------------- | ------------------------- |
| Python        | Core programming language |
| Pandas        | Data processing           |
| NumPy         | Numerical operations      |
| Scikit-learn  | Machine learning          |
| Random Forest | Classification            |
| Joblib        | Model serialization       |
| SHAP          | Explainable AI            |
| Matplotlib    | Visualization             |
| Streamlit     | Interactive dashboard     |

---

# 📂 Project Structure

text
Health-Dispatch-Ai/
│
├── app.py
├── train_model.py
├── dataset.csv
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore


### File Description

| File               | Description                                     |
| ------------------ | ----------------------------------------------- |
| `app.py`           | Streamlit dashboard and prediction logic        |
| `train_model.py`   | Data preparation, model training and evaluation |
| `dataset.csv`      | Appointment dataset                             |
| `model.pkl`        | Serialized trained Random Forest model          |
| `requirements.txt` | Python dependencies                             |
| `README.md`        | Project documentation                           |
| `.gitignore`       | Files excluded from version control             |


# 🚀 Run Locally

## 1. Clone the repository

bash
git clone https://github.com/agupta91759-prog/Health-Dispatch-Ai.git
cd Health-Dispatch-Ai


## 2. Install dependencies

bash
pip install -r requirements.txt


## 3. Run the Streamlit application

bash
streamlit run app.py


The dashboard should open in your browser.

---

# 🧪 Retrain the Model

To train the model again:

bash
python train_model.py


The training pipeline:

1. Loads the dataset
2. Performs data preprocessing
3. Creates engineered features
4. Splits the data into training and testing sets
5. Trains the Random Forest classifier
6. Evaluates the model
7. Saves the trained model as `model.pkl`

---

# ⚠️ Limitations

This project is a prototype and educational AI/ML decision-support system.

The current implementation includes several demonstration assumptions:

* Technician drive time is simulated.
* The weekly dispatch roster uses simulated appointment data.
* Financial costs are demonstration assumptions.
* The SMS workflow is simulated and does not represent actual message delivery.
* The system should not be used for real clinical decision-making without appropriate validation, security, privacy controls, and regulatory review.

---

# 🔮 Future Improvements

Potential production improvements include:

* Real appointment CSV upload
* Real-time appointment database integration
* Real routing and travel-time APIs
* Actual SMS integration
* Model performance monitoring
* Automated model retraining
* ROC-AUC / Precision / Recall monitoring
* Data and model drift detection
* Cloud deployment
* Authentication and role-based access
* Production database integration
* Model versioning and experiment tracking

---

# 👩‍💻 Author

**Ananya Gupta**

AI/ML portfolio project demonstrating:

**Machine Learning + Explainable AI + Batch Inference + Business Decision Support + Streamlit Deployment**




