import argparse
import sys
import os
import pandas as pd
import logging
import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper import CBBScraper
from processor import DataProcessor
from model import GamePredictor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_current_season():
    now = datetime.datetime.now()
    # If currently later than October, use next year (e.g., Nov 2023 -> 2024 season)
    if now.month >= 11:
        return now.year + 1
    return now.year

def main():
    parser = argparse.ArgumentParser(description="March Madness Predictor")
    parser.add_argument("--team_a", type=str, help="First Team Name (e.g. 'Duke')", required=True)
    parser.add_argument("--team_b", type=str, help="Second Team Name (e.g. 'North Carolina')", required=True)
    parser.add_argument("--home_court", action="store_true", help="Set if Team A is the home team")
    parser.add_argument("--neutral", action="store_true", help="Set if game is neutral site (overrides home_court)")
    
    args = parser.parse_args()
    
    current_season = get_current_season()
    train_season = current_season - 1
    logger.info(f"Training Season: {train_season}, Prediction Season: {current_season}")
    
    scraper = CBBScraper()
    processor = DataProcessor()
    predictor = GamePredictor(model_path=os.path.join(CACHE_DIR, 'cbb_model.pkl'))
    
    # Check if model exists, if not ensure we have data and train
    if not predictor.load():
        logger.info("Model not found or update needed. Checking for cached data...")
        
        # Training Data
        stats_path_train = os.path.join(CACHE_DIR, f'stats_{train_season}.csv')
        games_path_train = os.path.join(CACHE_DIR, f'games_{train_season}.csv')
        
        if os.path.exists(stats_path_train) and os.path.exists(games_path_train):
            stats_df_train = pd.read_csv(stats_path_train)
            games_df_train = pd.read_csv(games_path_train)
        else:
            logger.info("Scraping fresh training data...")
            stats_df_train = scraper.get_team_stats(train_season)
            games_df_train = scraper.get_game_results(train_season, stats_df_train)
            
            if stats_df_train is not None and games_df_train is not None:
                stats_df_train.to_csv(stats_path_train, index=False)
                games_df_train.to_csv(games_path_train, index=False)
            else:
                logger.error("Failed to scrape training data.")
                return

        logger.info("Training model...")
        X_y = processor.prepare_training_data(games_df_train, stats_df_train)
        if X_y.empty:
             logger.error("No training data generated.")
             return
             
        # Columns that are not 'Won' are features
        features = [c for c in X_y.columns if c != 'Won']
        predictor.train(X_y[features], X_y['Won'])

    # Load current stats for prediction
    stats_path_pred = os.path.join(CACHE_DIR, f'stats_{current_season}.csv')
    if os.path.exists(stats_path_pred):
        stats_df_pred = pd.read_csv(stats_path_pred)
    else:
        logger.info(f"Fetching statistics for current season ({current_season})...")
        stats_df_pred = scraper.get_team_stats(current_season)
        if stats_df_pred is not None:
            stats_df_pred.to_csv(stats_path_pred, index=False)
        else:
            logger.error(f"Failed to fetch stats for {current_season}")
            return



    # Normalize input names
    team_a = args.team_a.strip()
    team_b = args.team_b.strip()
    
    # Fuzzy match or direct lookup validation (Basic implementation)
    # create map
    stats_df_pred['School_Clean'] = stats_df_pred['School'].apply(lambda x: x.replace(' NCAA', '').strip())
    
    # Simple lookup
    def find_team_stats(df, name):
        matches = df[df['School_Clean'].str.contains(name, case=False, regex=False)]
        if matches.empty:
            return None
        if len(matches) > 1:
            # Try exact match
            exact = df[df['School_Clean'].str.lower() == name.lower()]
            if not exact.empty:
                return exact.iloc[0]
            logger.warning(f"Multiple matches for {name}: {matches['School_Clean'].tolist()}. Using first.")
        return matches.iloc[0]

    stats_a = find_team_stats(stats_df_pred, team_a)
    stats_b = find_team_stats(stats_df_pred, team_b)
    
    if stats_a is None:
        logger.error(f"Could not find stats for {team_a}")
        return
    if stats_b is None:
        logger.error(f"Could not find stats for {team_b}")
        return
        
    logger.info(f"Matchup: {stats_a['School_Clean']} vs {stats_b['School_Clean']}")
    
    is_home = args.home_court and not args.neutral
    
    # Create feature vector
    # We need to construct it exactly like training data features
    # But wait, processor.extract_features handles the logic
    # We DO need to pass it `is_home_a`
    
    features_dict = processor.extract_features(stats_a, stats_b, is_home)
    features_df = pd.DataFrame([features_dict])
    
    prob = predictor.predict_proba(features_df)
    
    print(f"\nprediction_success: true")
    print(f"Matchup: {stats_a['School_Clean']} vs {stats_b['School_Clean']}")
    print(f"Location: {'Home' if is_home else 'Neutral/Away'} for {stats_a['School_Clean']}")
    print(f"Probability of {stats_a['School_Clean']} winning: {prob:.2%}")

if __name__ == "__main__":
    main()
