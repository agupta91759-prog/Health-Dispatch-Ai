import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. Loading dataset.csv...")
df = pd.read_csv('dataset.csv')

print("2. Cleaning data and engineering features...")
# FIX: Use dt.normalize() to safely handle the dates in newer Pandas versions
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], utc=True).dt.normalize()
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], utc=True).dt.normalize()

# Calculate Lead Time (days between scheduling and the actual appointment)
df['Lead_Time_Days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['Lead_Time_Days'] = df['Lead_Time_Days'].apply(lambda x: max(x, 0)) # Remove negative days

# Create synthetic Drive Time to simulate a mobile health technician driving to a home
np.random.seed(42)
df['Drive_Time_Mins'] = np.random.randint(10, 61, size=len(df))

# Convert Target column 'No-show' into 1 (Yes, missed) and 0 (No, showed up)
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

# Select our training features
features = ['Age', 'Lead_Time_Days', 'Drive_Time_Mins', 'Scholarship', 
            'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
X = df[features]
y = df['No-show']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training the Random Forest AI model (this may take 10-20 seconds)...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\n4. Saving the model...")
joblib.dump(model, 'model.pkl')
print("✅ Success! 'model.pkl' has been saved to your folder.")