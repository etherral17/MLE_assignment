import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

#  Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

#  Data Preprocessing (Remove Unwanted Columns)
def preprocess_data(df, target_column="vomitoxin_ppb", id_column="hsi_id"):
    X = df.drop(columns=[target_column, id_column])
    y = df[target_column]
    return X, y

#  Train Initial XGBoost Model
def train_xgboost(X_train, y_train, params=None):
    if params is None:
        params = {"n_estimators": 300, "learning_rate": 0.05,"max_depth" : 6,"subsample": 0.8,"colsample_bytree": 0.8, "random_state": 42}
    
    # model = xgb.XGBRegressor(
    # objective="reg:squarederror",
    # n_estimators=500,  # Number of trees
    # learning_rate=0.05,  # Step size shrinkage
    # max_depth=6,  # Maximum depth of a tree
    # subsample=0.8,  # Fraction of samples used for training
    # colsample_bytree=0.8,  # Fraction of features used for training
    # random_state=42)


    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

#  Get Feature Importances from XGBoost
def get_xgb_important_features(model, X_train, threshold=0.01):
    feature_importances = model.feature_importances_
    important_features = X_train.columns[feature_importances > threshold]
    return list(important_features)

#  Perform SHAP Analysis
def get_shap_important_features(model, X_train, threshold=0.01):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap_importances = np.abs(shap_values.values).mean(axis=0)
    important_features = X_train.columns[shap_importances > threshold]
    return list(important_features)

#  Select Final Important Features
def select_final_features(xgb_features, shap_features):
    return list(set(xgb_features) & set(shap_features))

#  Retrain Model on Important Features
def retrain_model(X_train, y_train, X_test, final_features, params=None):
    X_train_reduced = X_train[final_features]
    X_test_reduced = X_test[final_features]

    model = train_xgboost(X_train_reduced, y_train, params)
    return model, X_test_reduced

#  Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse,r2

#  Save the Trained Model
def save_model(model, file_name="xgb_model.pkl"):
    with open(file_name, "wb") as file:
        pickle.dump(model, file)
    print(f"✅ Model saved as {file_name}")

#   MAIN FUNCTION TO RUN EVERYTHING 
def main():
    # Load and preprocess data
    df = load_data(r"C:\Users\ether\OneDrive\Documents\assignment_imago\MLE-Assignment.csv")
    X, y = preprocess_data(df)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Initial Model
    xgb_model = train_xgboost(X_train, y_train)

    # Get Important Features
    xgb_features = get_xgb_important_features(xgb_model, X_train)
    shap_features = get_shap_important_features(xgb_model, X_train)

    # Select Final Features
    final_features = select_final_features(xgb_features, shap_features)
    print(f"Selected Features: {final_features}")

    # Retrain Model on Important Features
    reduced_model, X_test_reduced = retrain_model(X_train, y_train, X_test, final_features)

    # Evaluate Model
    mse,r2 = evaluate_model(reduced_model, X_test_reduced, y_test)
    print(f" Mean Squared Error (Reduced Model): {mse:.4f} and R2 score : {r2:.4f}")

    # Save Model
    save_model(reduced_model)

if __name__ == "__main__":
    main()
