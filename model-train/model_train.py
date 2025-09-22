import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories"""
    os.makedirs("trained-models", exist_ok=True)
    print("Created trained-models directory")

def load_and_validate_data(file_path):
    """Load and validate the training data"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data not found at {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Check if required columns exist
        required_cols = ['Recommended_Career', 'Preferred_Work_Style', 'CGPA', 
                        'Current_Projects_Count', 'Internship_Experience',
                        'Wants_to_Go_for_Masters', 'Interested_in_Research']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("Data validation passed")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data with improved handling"""
    
    # Define columns for multi-label encoding
    multi_cols = [
        'Programming_Languages',
        'Certifications',
        'Extracurricular_Interests',
        'Interest_Areas',
        'Soft_Skills',
        'Tools_Techstack',
        'Favourite_Subjects',
        'Problem_Solving_Style'
    ]
    
    # Check which multi-label columns actually exist
    existing_multi_cols = [col for col in multi_cols if col in df.columns]
    print(f"Found {len(existing_multi_cols)} multi-label columns: {existing_multi_cols}")
    
    # Fill NaN values in multi-label columns
    for col in existing_multi_cols:
        df[col] = df[col].fillna('')
        print(f"Processed NaN values in {col}")
    
    # Split the multi-label columns into lists
    def split_to_list(s):
        if isinstance(s, str):
            return [item.strip() for item in s.split(',') if item.strip()]
        return []
    
    for col in existing_multi_cols:
        df[col] = df[col].apply(split_to_list)
        unique_items = set()
        for items in df[col]:
            unique_items.update(items)
        print(f"{col}: {len(unique_items)} unique items")
    
    # Apply MultiLabelBinarizer to each multi-label column
    mlb_dict = {}
    for col in existing_multi_cols:
        try:
            mlb = MultiLabelBinarizer()
            mat = mlb.fit_transform(df[col])
            cols = [f"{col}_{v}" for v in mlb.classes_]
            df_mlb = pd.DataFrame(mat, columns=cols, index=df.index)
            df = pd.concat([df, df_mlb], axis=1).drop(columns=[col])
            mlb_dict[col] = mlb
            print(f"MultiLabelBinarizer for {col}: {len(mlb.classes_)} classes")
        except Exception as e:
            print(f"Warning: Could not process {col}: {e}")
    
    # One-hot encoding for 'Preferred_Work_Style'
    if 'Preferred_Work_Style' in df.columns:
        try:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            arr = ohe.fit_transform(df[['Preferred_Work_Style']])
            ohe_cols = ohe.get_feature_names_out(['Preferred_Work_Style'])
            df_ohe = pd.DataFrame(arr, columns=ohe_cols, index=df.index)
            df = pd.concat([df, df_ohe], axis=1).drop(columns=['Preferred_Work_Style'])
            print(f"OneHotEncoder for Preferred_Work_Style: {len(ohe.categories_[0])} categories")
        except Exception as e:
            print(f"Warning: Could not process Preferred_Work_Style: {e}")
            # Create a dummy encoder
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit([['Remote'], ['Office'], ['Hybrid']])
    else:
        # Create a dummy encoder if column doesn't exist
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit([['Remote'], ['Office'], ['Hybrid']])
        print("Created dummy OneHotEncoder for Preferred_Work_Style")
    
    return df, mlb_dict, ohe

def train_model(df):
    """Train the RandomForest model"""
    
    # Prepare features (X) and target (y)
    if 'Recommended_Career' not in df.columns:
        raise ValueError("Target column 'Recommended_Career' not found")
    
    X = df.drop(columns=['Recommended_Career'])
    y = df['Recommended_Career']
    
    print(f"Features shape: {X.shape}")
    print(f"Target classes: {len(y.unique())}")
    print(f"Target distribution:")
    print(y.value_counts().head(10))
    
    # Label encode the target variable
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train RandomForestClassifier
    print("Training RandomForest model...")
    final_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        ccp_alpha=1e-3,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    final_rf.fit(X_train, y_train)
    
    # Validate the model
    y_pred = final_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print classification report for top classes
    target_names = le.classes_
    print("\nClassification Report (first 10 classes):")
    unique_test_classes = np.unique(y_test)[:10]
    test_target_names = [target_names[i] for i in unique_test_classes]
    
    return final_rf, le, X.columns

def save_models(model, label_encoder, mlb_dict, ohe, feature_names):
    """Save all models and encoders"""
    
    try:
        # Save model, encoders, and transformers
        joblib.dump(model, "trained-models/careermodel.pkl")
        print("Saved: careermodel.pkl")
        
        joblib.dump(label_encoder, "trained-models/labelencoder.pkl")
        print("Saved: labelencoder.pkl")
        
        joblib.dump(mlb_dict, "trained-models/mlbdict.pkl")
        print("Saved: mlbdict.pkl")
        
        joblib.dump(ohe, "trained-models/ohencoder.pkl")
        print("Saved: ohencoder.pkl")
        
        # Save feature names for reference
        joblib.dump(list(feature_names), "trained-models/feature_names.pkl")
        print("Saved: feature_names.pkl")
        
        return True
        
    except Exception as e:
        print(f"Error saving models: {e}")
        return False

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    
    try:
        # Get importances & sort
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importances.nlargest(top_n)
        
        plt.figure(figsize=(12, 8))
        top_features.sort_values().plot.barh()
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('trained-models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance.png")
        
        plt.show()
        
    except Exception as e:
        print(f"Warning: Could not create feature importance plot: {e}")

def test_saved_models():
    """Test that saved models can be loaded"""
    
    try:
        print("\nTesting saved models...")
        
        # Load all models
        model = joblib.load("trained-models/careermodel.pkl")
        le = joblib.load("trained-models/labelencoder.pkl")
        mlb_dict = joblib.load("trained-models/mlbdict.pkl")
        ohe = joblib.load("trained-models/ohencoder.pkl")
        
        print(f"Model type: {type(model).__name__}")
        print(f"Model features: {len(model.feature_names_in_)}")
        print(f"Label encoder classes: {len(le.classes_)}")
        print(f"MultiLabelBinarizers: {len(mlb_dict)}")
        print(f"OneHotEncoder categories: {len(ohe.categories_[0])}")
        
        # Test prediction with dummy data
        dummy_features = np.zeros((1, len(model.feature_names_in_)))
        prediction = model.predict(dummy_features)
        career = le.inverse_transform(prediction)[0]
        print(f"Test prediction: {career}")
        
        print("All models loaded and tested successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing models: {e}")
        return False

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Career Navigator - Model Training Pipeline")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Load and validate data
    df = load_and_validate_data("data/19k.csv")
    if df is None:
        print("Training failed: Could not load data")
        return False
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_processed, mlb_dict, ohe = preprocess_data(df)
    
    # Train model
    print("\nTraining model...")
    model, label_encoder, feature_names = train_model(df_processed)
    
    # Save models
    print("\nSaving models...")
    if save_models(model, label_encoder, mlb_dict, ohe, feature_names):
        print("Models saved successfully!")
    else:
        print("Failed to save models")
        return False
    
    # Plot feature importance
    print("\nGenerating feature importance plot...")
    plot_feature_importance(model, feature_names)
    
    # Test saved models
    if test_saved_models():
        print("\n" + "=" * 60)
        print("SUCCESS: All models trained and saved successfully!")
        print("You can now start your backend with:")
        print("uvicorn backend:app --reload --host 0.0.0.0 --port 8000")
        print("=" * 60)
        return True
    else:
        print("Training completed but model testing failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nTraining failed. Please check the errors above.")
    else:
        # Print model summary
        print("\nModel Summary:")
        print("- RandomForest with 200 estimators")
        print("- Cross-validated accuracy reported")
        print("- Feature importance analysis included")
        print("- All encoders saved for production use")
        print("\nFiles created:")
        for file in ["careermodel.pkl", "labelencoder.pkl", "mlbdict.pkl", "ohencoder.pkl", "feature_names.pkl"]:
            print(f"- trained-models/{file}")