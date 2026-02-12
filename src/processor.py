import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        pass

        pass

    def _create_team_game_logs(self, games_df):
        """
        Transforms games_df into a double-stacked team-centric log with rolling stats.
        Returns a DataFrame with ['Team', 'Date', 'Last5_Pts', 'Last5_Opp_Pts', 'Last5_Margin']
        """
        # 1. Create Team-Centric Rows
        team_logs = []
        
        # Ensure Date is datetime
        games_df['Date'] = pd.to_datetime(games_df['Date'])
        
        for _, row in games_df.iterrows():
            # Visitor Perspective
            team_logs.append({
                'Team': row['Visitor'].replace(' NCAA', '').strip(),
                'Date': row['Date'],
                'Pts': row['Visitor_Pts'],
                'Opp_Pts': row['Home_Pts'],
                'Margin': row['Visitor_Pts'] - row['Home_Pts']
            })
            # Home Perspective
            team_logs.append({
                'Team': row['Home'].replace(' NCAA', '').strip(),
                'Date': row['Date'],
                'Pts': row['Home_Pts'],
                'Opp_Pts': row['Visitor_Pts'],
                'Margin': row['Home_Pts'] - row['Visitor_Pts']
            })
            
        df_logs = pd.DataFrame(team_logs)
        
        # 2. Sort and Calculate Rolling Stats
        df_logs = df_logs.drop_duplicates(subset=['Team', 'Date'])
        df_logs = df_logs.sort_values(['Team', 'Date'])
        
        # Group by Team and calculate rolling mean
        # shift(1) allows us to use PAST games to predict CURRENT game
        # min_periods=1 allows stats early in season
        window = 5
        
        grouped = df_logs.groupby('Team')
        df_logs['Last5_Pts'] = grouped['Pts'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        df_logs['Last5_Opp_Pts'] = grouped['Opp_Pts'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        df_logs['Last5_Margin'] = grouped['Margin'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        
        return df_logs.set_index(['Team', 'Date'])

    def prepare_training_data(self, games_df, stats_df):
        """
        Merges game results with team stats to create a training dataset.
        Returns X (features) and y (target).
        """
        df = games_df.copy()
        
        # Standardize column names if needed (similar to existing logic)
        cols = df.columns.tolist()
        pts_indices = [i for i, c in enumerate(cols) if 'PTS' in str(c)]
        if len(pts_indices) >= 2:
            df.rename(columns={cols[pts_indices[0]]: 'Visitor_Pts', cols[pts_indices[1]]: 'Home_Pts'}, inplace=True)
        
        df['Visitor_Pts'] = pd.to_numeric(df['Visitor_Pts'], errors='coerce')
        df['Home_Pts'] = pd.to_numeric(df['Home_Pts'], errors='coerce')
        df = df.dropna(subset=['Visitor_Pts', 'Home_Pts'])
        
        # Generate Rolling Stats
        rolling_stats = self._create_team_game_logs(df)
        
        processed_games = []
        
        stats_df['School_Clean'] = stats_df['School'].apply(lambda x: x.replace(' NCAA', '').strip())
        team_stats_map = stats_df.set_index('School_Clean').to_dict('index')
        
        # Ensure DataFrame Date is datetime for merging
        df['Date'] = pd.to_datetime(df['Date'])
        
        for _, row in df.iterrows():
            visitor = row['Visitor'].replace(' NCAA', '').strip()
            home = row['Home'].replace(' NCAA', '').strip()
            date = row['Date']
            
            if visitor not in team_stats_map or home not in team_stats_map:
                continue
                
            visitor_stats = team_stats_map[visitor]
            home_stats = team_stats_map[home]
            
            # Get Rolling Stats
            # Use .get or loc handling for missing index
            def get_rolling_margin(team, date_val):
                try:
                    # If index is unique, loc returns Series (row). If not, DataFrame.
                    # We deduplicated, so it should be Series.
                    roll = rolling_stats.loc[(team, date_val)]
                    val = roll['Last5_Margin']
                    if pd.isna(val): return 0
                    return val
                except KeyError:
                    return 0
            
            vis_last5_margin = get_rolling_margin(visitor, date)
            home_last5_margin = get_rolling_margin(home, date)
            
            # Perspective: Home Team
            home_row = {
                'Team_AdjEM': home_stats.get('SRS', 0),
                'Team_OffEff': home_stats.get('ORtg', home_stats.get('Pts', 0)),
                'Opp_AdjEM': visitor_stats.get('SRS', 0),
                'Opp_OffEff': visitor_stats.get('ORtg', visitor_stats.get('Pts', 0)),
                'Is_Home': 1,
                'Won': 1 if row['Home_Pts'] > row['Visitor_Pts'] else 0,
                'Team_Last5_Margin': home_last5_margin,
                'Opp_Last5_Margin': vis_last5_margin
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
                'Won': 1 if row['Visitor_Pts'] > row['Home_Pts'] else 0,
                'Team_Last5_Margin': vis_last5_margin,
                'Opp_Last5_Margin': home_last5_margin
            }
             # Add differentials
            for k in visitor_stats:
                 if isinstance(visitor_stats[k], (int, float)) and k in home_stats:
                    visitor_row[f'{k}_Diff'] = visitor_stats[k] - home_stats[k]

            processed_games.append(visitor_row)
            
        return pd.DataFrame(processed_games)

    def extract_features(self, team_a_stats, team_b_stats, is_home_a, team_a_log=None, team_b_log=None):
        """
        Prepares a single row for prediction.
        team_a_log: DataFrame of team A's recent games (for rolling calc)
        team_b_log: DataFrame of team B's recent games
        """
        # Calculate Rolling Stats if logs provided
        team_a_rolling_margin = 0
        team_b_rolling_margin = 0
        
        window = 5
        
        if team_a_log is not None and not team_a_log.empty:
            # Assume log is sorted or sort it
            # We want the stats LEADING UP TO this game.
            # So we take the last 5 games from the log.
            # Convert date just in case
            if 'Date' in team_a_log.columns:
                 team_a_log['Date'] = pd.to_datetime(team_a_log['Date'])
                 team_a_log = team_a_log.sort_values('Date')
            
            # Calculate margin for each game
            # We need to know if they were Home or Visitor in that log to calculate margin
            # Normalized log (like in _create_team_game_logs) is best.
            # But the log passed here might be raw schedule.
            # Let's process it quickly.
            
            margins = []
            for _, row in team_a_log.tail(window).iterrows():
                 # Infer perspective
                 # If we have 'Margin' column pre-calculated, great.
                 if 'Margin' in row:
                     margins.append(row['Margin'])
                 else:
                     # Calculate
                     # Check if team A is Home or Visitor in this row
                     # Rough heuristic: if 'Home' == Team A Name... but names vary.
                     # Simpler: We assume the log passed in IS specific to Team A, 
                     # but typically schedule scraper returns standard [Visitor, Home] cols.
                     pass 
            
            # Actually, main.py should pass pre-calculated rolling value or a cleaned log?
            # Let's calculate it here properly using internal helper if possible, 
            # Or just expect the simple "Last 5" dataframe which has 'Margin'.
            if 'Margin' in team_a_log.columns:
                 margins = team_a_log['Margin'].tail(window)
                 if len(margins) > 0:
                     team_a_rolling_margin = margins.mean()

        if team_b_log is not None and not team_b_log.empty:
             if 'Date' in team_b_log.columns:
                 team_b_log['Date'] = pd.to_datetime(team_b_log['Date'])
                 team_b_log = team_b_log.sort_values('Date')
                 
             if 'Margin' in team_b_log.columns:
                 margins = team_b_log['Margin'].tail(window)
                 if len(margins) > 0:
                     team_b_rolling_margin = margins.mean()
        
        row = {
            'Is_Home': 1 if is_home_a else 0,
            'Team_AdjEM': team_a_stats.get('SRS', 0),
            'Opp_AdjEM': team_b_stats.get('SRS', 0),
            'Team_Last5_Margin': team_a_rolling_margin,
            'Opp_Last5_Margin': team_b_rolling_margin
        }
        
        # Calculate differentials
        for k in team_a_stats.index:
             if isinstance(team_a_stats[k], (int, float)) and k in team_b_stats:
                row[f'{k}_Diff'] = team_a_stats[k] - team_b_stats[k]
                
        return row
