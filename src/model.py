from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class GamePredictor:
    def __init__(self, model_path='model.pkl'):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_path = model_path
        self.feature_names = []

    def train(self, X, y):
        """Trains the model and saves it."""
        self.feature_names = X.columns.tolist()
        
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        self.save()
        return accuracy

    def predict_proba(self, features_df):
        """Predicts probability of a win."""
        # Ensure features match
        missing_cols = set(self.feature_names) - set(features_df.columns)
        for c in missing_cols:
            features_df[c] = 0 # Default missing features to 0
            
        # Select only model columns in order
        features_df = features_df[self.feature_names]
        
        return self.model.predict_proba(features_df)[0][1] # Probability of class 1 (Win)

    def save(self):
        joblib.dump({'model': self.model, 'features': self.feature_names}, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.feature_names = data['features']
            logger.info("Model loaded successfully.")
            return True
        return False
