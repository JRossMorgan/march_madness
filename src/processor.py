import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        pass

    def prepare_training_data(self, games_df, stats_df):
        """
        Merges game results with team stats to create a training dataset.
        Returns X (features) and y (target).
        """
        # Clean specific scraped columns from games_df
        # Expected cols: Date, Visitor, PTS, Home, PTS.1
        
        # Rename columns to be standard
        # Note: Depending on the pandas version and read_html, duplicate cols might be PTS, PTS.1
        # Let's verify columns exist or find them by index if names are messy
        
        # Identify columns by position if names are variable
        # Typically: Date(0), Time(1), Visitor(2), PTS(3), Home(4), PTS(5) ... check indices
        # But safest is to trust the scraper's dataframe structure or clean it there.
        # Let's assume the scraper returns a raw dataframe and we fix it here or in scraper.
        # I'll update this to be robust.
        
        df = games_df.copy()
        
        # Standardize column names if they are auto-generated
        # We need 'Visitor', 'Visitor_Pts', 'Home', 'Home_Pts'
        
        # Let's try to infer based on column content or position if names are generic
        # But for now, let's assume we can rely on string matching 'PTS'
        
        cols = df.columns.tolist()
        # Find indices of 'PTS'
        pts_indices = [i for i, c in enumerate(cols) if 'PTS' in str(c)]
        if len(pts_indices) >= 2:
            df.rename(columns={cols[pts_indices[0]]: 'Visitor_Pts', cols[pts_indices[1]]: 'Home_Pts'}, inplace=True)
        else:
            # Fallback based on typical position
            # [Date, Visitor, Visitor_Pts, Home, Home_Pts] -> often indices 1, 2, 3, 4, 5
            # We'll just log warning for now
            logger.warning("Could not automatically identify PTS columns. Checking for manual names.")

        # Ensure numeric
        df['Visitor_Pts'] = pd.to_numeric(df['Visitor_Pts'], errors='coerce')
        df['Home_Pts'] = pd.to_numeric(df['Home_Pts'], errors='coerce')
        
        # Drop rows with missing scores (future games)
        df = df.dropna(subset=['Visitor_Pts', 'Home_Pts'])
        
        # Create Target: 1 if Home Team Wins, 0 if Visitor Wins
        # Note: The request asks for "Team A", "Team B", "Home Court Boolean".
        # We can train the model relative to "Team A".
        # Let's structure the training data such that:
        # We process each game twice? Or just once with a "IsHome" feature?
        # If we feed (Home, Visitor, IsHome=1) -> Target=Win/Loss
        # We could also feed (Visitor, Home, IsHome=0) -> Target=Win/Loss
        # This doubles data and ensures model learns both sides.
        
        processed_games = []
        
        # We need to map team names in games_df to stats_df 'School'
        # Basic normalization wrapper
        stats_df['School_Clean'] = stats_df['School'].apply(lambda x: x.replace(' NCAA', '').strip())
        team_stats_map = stats_df.set_index('School_Clean').to_dict('index')
        
        for _, row in df.iterrows():
            visitor = row['Visitor'].replace(' NCAA', '').strip()
            home = row['Home'].replace(' NCAA', '').strip()
            
            if visitor not in team_stats_map or home not in team_stats_map:
                continue
                
            visitor_stats = team_stats_map[visitor]
            home_stats = team_stats_map[home]
            
            # Perspective: Home Team
            home_row = {
                'Team_AdjEM': home_stats.get('SRS', 0), # SRS is a good proxy for efficiency
                'Team_OffEff': home_stats.get('ORtg', home_stats.get('Pts', 0)), # Fallback
                'Opp_AdjEM': visitor_stats.get('SRS', 0),
                'Opp_OffEff': visitor_stats.get('ORtg', visitor_stats.get('Pts', 0)),
                'Is_Home': 1,
                'Won': 1 if row['Home_Pts'] > row['Visitor_Pts'] else 0
            }
            # Add differentials
            for k in home_stats:
                if isinstance(home_stats[k], (int, float)) and k in visitor_stats:
                    home_row[f'{k}_Diff'] = home_stats[k] - visitor_stats[k]
            
            processed_games.append(home_row)
            
            # Perspective: Visitor Team
            visitor_row = {
                'Team_AdjEM': visitor_stats.get('SRS', 0),
                'Team_OffEff': visitor_stats.get('ORtg', visitor_stats.get('Pts', 0)),
                'Opp_AdjEM': home_stats.get('SRS', 0),
                'Opp_OffEff': home_stats.get('ORtg', home_stats.get('Pts', 0)),
                'Is_Home': 0,
                'Won': 1 if row['Visitor_Pts'] > row['Home_Pts'] else 0
            }
             # Add differentials
            for k in visitor_stats:
                 if isinstance(visitor_stats[k], (int, float)) and k in home_stats:
                    visitor_row[f'{k}_Diff'] = visitor_stats[k] - home_stats[k]

            processed_games.append(visitor_row)
            
        return pd.DataFrame(processed_games)

    def extract_features(self, team_a_stats, team_b_stats, is_home_a):
        """Prepares a single row for prediction."""
        # Need to ensure we use the same columns as training.
        # This will be handled by the DataFrame structure, but we need to guarantee column order or usage.
        # For now, return a dict that can be converted to DF.
        
        row = {
            'Is_Home': 1 if is_home_a else 0,
            # Add other base features matchin training
             'Team_AdjEM': team_a_stats.get('SRS', 0),
             'Opp_AdjEM': team_b_stats.get('SRS', 0),
        }
        
        # Calculate differentials
        for k in team_a_stats.index:
             if isinstance(team_a_stats[k], (int, float)) and k in team_b_stats:
                row[f'{k}_Diff'] = team_a_stats[k] - team_b_stats[k]
                
        return row
