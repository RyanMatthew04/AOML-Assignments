import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "star_classification.csv"  
df = pd.read_csv(file_path)

# Clean dataset
columns_to_drop = ["obj_ID", "alpha", "delta", "run_ID", "rerun_ID", "cam_col", 
                   "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"]
df_cleaned = df.drop(columns=columns_to_drop)

# Feature types
categorical_cols = []  
numerical_cols = ["u", "g", "r", "i", "z", "redshift"]

# Label encoding for target class
label_encoder = LabelEncoder()
df_cleaned["class"] = label_encoder.fit_transform(df_cleaned["class"])

# One-hot encoding for categorical columns if any
if categorical_cols:
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded_cats = one_hot_encoder.fit_transform(df_cleaned[categorical_cols])
    df_encoded = pd.DataFrame(encoded_cats, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
    df_cleaned = df_cleaned.drop(columns=categorical_cols).reset_index(drop=True)
    df_cleaned = pd.concat([df_cleaned, df_encoded], axis=1)
else:
    one_hot_encoder = None

# Feature scaling
scaler = StandardScaler()
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

# Define features and target
X = df_cleaned.drop(columns=["class"])
y = df_cleaned["class"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()


# with open("scaler.pkl", "wb") as scaler_file:
#     pickle.dump(scaler, scaler_file)

# with open("label_encoder.pkl", "wb") as encoder_file:
#     pickle.dump(label_encoder, encoder_file)

# if one_hot_encoder:
#     with open("one_hot_encoder.pkl", "wb") as ohe_file:
#         pickle.dump(one_hot_encoder, ohe_file)

# with open("model.pkl", "wb") as model_file:
#     pickle.dump(clf, model_file)

# print("Model, scaler, and encoders saved successfully!")
