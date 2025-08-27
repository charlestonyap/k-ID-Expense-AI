# cd "C:\Users\charl\Documents\K-ID\Corporate Credit AI"
# python -m venv .venv
# .venv\Scripts\activate.bat
# pip install pandas numpy streamlit fuzzywuzzy python-Levenshtein shap spacy sentence-transformers imbalanced-learn scikit-learn xgboost plotly
# streamlit run train_model.py

import pandas as pd
import os
from Personal_Expense_AI import PersonalExpenseDetector
import warnings
import logging

# Suppress Streamlit ScriptRunContext warnings
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

def train_and_save_model():
    """
    One-time training script to create and save the ML model
    Run this once to generate the trained model files
    """
    print("=" * 60)
    print("🚀 PERSONAL EXPENSE AI MODEL TRAINING")
    print("=" * 60)
    print("Starting model training pipeline...")
    
    # Check if dataset exists
    dataset_path = "credit_card_transactions.csv"
    print(f"\n📁 Checking for dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the credit card fraud dataset is available")
        return
    else:
        print("✅ Dataset file found!")
    
    try:
        # Create detector and train models
        print("\n🏗️  Initializing PersonalExpenseDetector...")
        detector = PersonalExpenseDetector(auto_train=False)
        print("✅ Detector initialized successfully!")
        
        print("\n📊 Loading and validating dataset...")
        # Quick dataset validation
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded: {len(df):,} records")
        
        required_cols = ['amt', 'category', 'trans_date_trans_time', 'is_fraud']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return
        print("✅ All required columns present!")
        print(f"   - Fraud cases: {df['is_fraud'].sum():,} ({(df['is_fraud'].sum()/len(df)*100):.2f}%)")
        legitimate_count = (df['is_fraud'] == 0).sum()
        legitimate_pct = (legitimate_count / len(df)) * 100
        print(f"   - Legitimate cases: {legitimate_count:,} ({legitimate_pct:.2f}%)")
        
        print("\n🔄 Starting pre-training of all models...")
        print("   This may take several hours depending on your system...")
        
        success = detector.pretrain_all_models(
            credit_card_data_path=dataset_path,
            save_path="trained_models/",
            test_size=0.2
        )
        
        # Check NLP status
        if detector.is_embeddings_loaded:
            print(f"\n✅ NLP models loaded successfully: {detector.embedding_type}")
        else:
            print("\n⚠️ NLP models failed to load, using TF-IDF fallback")
        
        if detector.is_ml_trained:
            print("\n" + "=" * 60)
            print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            print("\n📊 TRAINING SUMMARY:")
            print("-" * 30)
            print(f"🏆 Best Model: {detector.best_model_name}")
            print(f"📈 Cross-Validation F1 Score: {detector.training_stats.get('best_cv_score', 0):.4f}")
            print(f"🎯 Test Set F1 Score: {detector.training_stats.get('best_test_f1', 0):.4f}")
            print(f"📋 Total Records Used: {detector.training_stats.get('total_records', 0):,}")
            print(f"🔄 Train/Test Split: {detector.training_stats.get('train_records', 0):,} / {detector.training_stats.get('test_records', 0):,}")
            
            # Show all models trained
            print(f"\n🔧 Models Trained and Evaluated:")
            for model_name, result in detector.cv_results.items():
                if 'error' not in result:
                    print(f"   ✅ {model_name}: Test F1 = {result['test_f1']:.4f}")
                else:
                    print(f"   ❌ {model_name}: Failed - {result['error']}")
            
            # NLP summary
            if detector.is_embeddings_loaded:
                print(f"\n📊 NLP Embedding Type: {detector.embedding_type}")
                print(f"📏 Embedding Dimension: {detector.embedding_dim}")
            else:
                print("\n📊 NLP: Using TF-IDF fallback")
            
            # Save the trained model
            print(f"\n💾 Saving trained model...")
            try:
                detector.save_models("trained_models/")
                print("✅ Model saved to 'trained_models/' directory")
                
                # List saved files
                saved_files = os.listdir("trained_models/")
                print("📁 Saved files:")
                for file in saved_files:
                    print(f"   - {file}")
                
                print("\n" + "=" * 60)
                print("🎊 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                print("✅ You can now run the main application to use the trained model.")
                print("🚀 The AI detection system is ready for production use!")
                
            except Exception as e:
                print(f"❌ Failed to save model: {e}")
        else:
            print("\n" + "=" * 60)
            print("❌ MODEL TRAINING FAILED")
            print("=" * 60)
            print("🔍 Check the training logs above for specific errors")
            
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        print("\n🔧 TROUBLESHOOTING CHECKLIST:")
        print("1. ✓ Dataset file exists and has the correct format")
        print("2. ✓ Required columns: 'amt', 'category', 'trans_date_trans_time', 'is_fraud'")
        print("3. ✓ All required Python packages are installed")
        print("4. ✓ Sufficient system memory available")
        print("5. ✓ No other processes using the dataset file")
        print("\n💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    train_and_save_model()