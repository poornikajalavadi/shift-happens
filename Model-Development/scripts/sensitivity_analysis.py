import pandas as pd
import shap
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt

# 1. Define Architecture Paths
BASE_DIR = Path(__file__).resolve().parent
FINAL_DATA_PATH = BASE_DIR / ".."/ ".."/ "Data-Pipeline"/"data"/"processed" / "application_train_merged.pkl"


def load_and_split_data(filepath: Path):
    """
    Loads the final merged dataset and splits it into features (X) and target (y).
    Automatically converts string/object columns to categorical for LightGBM.
    """
    print(f"Loading final dataset from: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find {filepath}")
        
    df = pd.read_pickle(filepath)
    print(f"Loaded successfully. Shape: {df.shape}")
    
    join_key = 'SK_ID_CURR'
    target_col = 'TARGET'
    
    # Separate Features (X) and Target (y)
    y = df[target_col]
    X = df.drop(columns=[target_col, join_key], errors='ignore')
    
    # --- Convert 'object' types and handle NaNs ---
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"Converting {len(object_cols)} object columns to category dtype...")
        for col in object_cols:
            # 1. Fill NaN values with the string 'Missing'
            X[col] = X[col].fillna('Missing')
            # 2. Convert to category dtype for LightGBM
            X[col] = X[col].astype('category')
    # -------------------------------------------------------
    # -----------------------------------------------------
    
    print(f"Feature matrix (X) shape: {X.shape}")
    return X, y

def run_shap_analysis(model, X):
    """
    Calculates SHAP values and generates the global feature importance plot.
    """
    print("\nCalculating SHAP values... (This might take a moment on large datasets)")
    
    # TreeExplainer is highly optimized for LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Generate Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    
    plt.title("SHAP Feature Importance (LightGBM)")
    plt.tight_layout()
    plt.show()
    
    return explainer, shap_values

def plot_feature_dependence(shap_values, X, feature_name, interaction=None):
    """
    Generates a SHAP dependence plot to see the exact impact of specific values/categories.
    """
    print(f"Generating dependence plot for {feature_name}...")
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Handle LightGBM binary classification output (sometimes it returns a list of arrays for class 0 and class 1)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # Generate the plot
    shap.dependence_plot(
        feature_name, 
        vals, 
        X, 
        interaction_index=interaction,
        show=False
    )
    
    plt.title(f"SHAP Dependence: How {feature_name} impacts Default Risk", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Step 1: Load data and build X, y directly from the merged file
    X, y = load_and_split_data(FINAL_DATA_PATH)
    
    # Step 2: Train LightGBM Model
    print("\nTraining LightGBM model...")
    # Using LGBMClassifier for predicting the Default Risk (TARGET)
    model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    
    # Step 3: Execute SHAP Analysis
    # Keeping this as a test if X is massive (>100k rows), we want to pass a sample to SHAP for speed:
    # explainer, shap_vals = run_shap_analysis(model, X.sample(10000, random_state=42))
    explainer, shap_values = run_shap_analysis(model, X)

    # Look at Occupation Type without any color interactions
    plot_feature_dependence(shap_values, X, 'OCCUPATION_TYPE', interaction=None)

    # Look at Age (DAYS_BIRTH) to see the exact curve of age vs risk
    plot_feature_dependence(shap_values, X, 'DAYS_BIRTH', interaction=None)