import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CBBScraper:
    def __init__(self):
        self.base_url = "https://www.sports-reference.com/cbb/seasons"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _make_request(self, url):
        """Helper to make requests with rate limiting."""
        try:
            logger.info(f"Fetching {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            time.sleep(3)  # Respect rate limits: 1 request every 3 seconds
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def get_team_stats(self, season):
        """Scrapes advanced team stats for a given season."""
        url = f"{self.base_url}/{season}-school-stats.html"
        content = self._make_request(url)
        
        if not content:
            return None

        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', {'id': 'basic_school_stats'})
        
        if not table:
            logger.error("Stats table not found")
            return None

        # Process the table into a DataFrame
        # Note: Sports-Reference puts headers every 20 rows, we need to filter them out
        try:
            df = pd.read_html(StringIO(str(table)))[0]
        except Exception as e:
            logger.error(f"Error parsing HTML table: {type(e).__name__}")
            return None
        
        # Clean column names
        # The table has a multi-level index, we want to flatten it or just take the relevant level
        # For 'basic_school_stats', the top level headers are just grouping (Overall, Conf, etc.)
        # The second level is the actual metric.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        # Deduplicate columns (keep first, usually Overall)
        df = df.loc[:, ~df.columns.duplicated()]

        # Filter out repeater headers and empty rows
        df = df[df['School'] != 'School']
        df = df[df['School'] != 'Overall'] # Remove league averages if present
        df = df.dropna(subset=['School'])

        # Rename columns for clarity if needed, or keep as is.
        # Key stats often: 'SRS', 'SOS', 'W', 'L', 'Pts', 'Opp', 'FG', 'FG%', '3P', '3P%', etc.
        # Ensure we keep 'School' as the identifier.
        
        # Convert numeric columns
        cols_to_numeric = ['W', 'L', 'SRS', 'SOS', 'Pts', 'Opp', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract slugs from the table links
        slugs = []
        rows = table.find_all('tr')
        for row in rows:
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class'):
                continue
            
            school_cell = row.find('td', {'data-stat': 'school_name'})
            if school_cell:
                link = school_cell.find('a')
                if link:
                    # href format: /cbb/schools/duke/men/2025.html
                    # slug: duke
                    href = link['href']
                    parts = href.split('/')
                    if len(parts) > 3:
                        slugs.append(parts[3])
                    else:
                        slugs.append(None)
                else:
                    slugs.append(None)
            else:
                # Might be a divider row or logic mismatch
                pass
        
        # Filter df to match slugs alignment? 
        # pd.read_html removes rows? It handles thead/tbody.
        # But 'Rk' rows are in the body sometimes.
        # 'df' already filtered "School != School".
        # Let's align by iterating or just use the extracted list if lengths match.
        # Safer: Extract slug + school name from BS4 and map to DF.
        
        # Simpler approach: Iterate rows, build a dict {School_Name: Slug}, map to DF.
        slug_map = {}
        for row in rows:
             school_cell = row.find('td', {'data-stat': 'school_name'})
             if school_cell and school_cell.find('a'):
                 name = school_cell.get_text().strip()
                 href = school_cell.find('a')['href']
                 # /cbb/schools/SLUG/men/YEAR.html
                 try:
                     slug = href.split('/')[3]
                     # Name might need cleaning (remove NCAA suffix handled in df, but here it has text)
                     slug_map[name] = slug
                 except:
                     pass

        # Normalize df 'School' to match key
        # Removing ' NCAA'
        df['School_Raw'] = df['School'].str.replace(' NCAA', '', regex=False).str.strip()
        
        # Apply map. Note: extracting text from 'a' usually matches df content but let's be careful.
        # Actually, df['School'] comes from text content.
        # BS4 text content should match.
        
        # Let's try to map.
        # We need a robust matching because of specific characters like '&' or '.'
        # We can construct the map with "name that matches DF".
        # In BS4: name = school_cell.get_text() -> "Duke" or "Duke NCAA"
        # In DF: "Duke" (after simple clean)
        
        def get_slug(name):
            # name input is from DF (cleaned)
            # check direct match
            if name in slug_map: return slug_map[name]
            # check with NCAA suffix
            if name + " NCAA" in slug_map: return slug_map[name + " NCAA"]
            return None

        df['Slug'] = df['School_Raw'].apply(get_slug)
        df.drop(columns=['School_Raw'], inplace=True)

        # Fetch Ratings to get Conference
        ratings_url = f"{self.base_url}/{season}-ratings.html"
        ratings_content = self._make_request(ratings_url)
        if ratings_content:
            try:
                soup_r = BeautifulSoup(ratings_content, 'html.parser')
                table_r = soup_r.find('table', {'id': 'ratings'})
                if table_r:
                    df_r = pd.read_html(StringIO(str(table_r)))[0]
                    
                    if isinstance(df_r.columns, pd.MultiIndex):
                        df_r.columns = df_r.columns.get_level_values(1)
                    
                    df_r = df_r[df_r['School'] != 'School']
                    df_r = df_r.dropna(subset=['School'])
                    
                    # Extract Slugs from ratings to join on
                    r_slug_map = {}
                    for row in table_r.find_all('tr'):
                        inv_cell = row.find('td', {'data-stat': 'school_name'})
                        if inv_cell and inv_cell.find('a'):
                           name = inv_cell.get_text().strip()
                           href = inv_cell.find('a')['href']
                           try:
                               slug = href.split('/')[3]
                               r_slug_map[name] = slug
                           except:
                               pass
                    
                    df_r['School_Raw'] = df_r['School'].str.replace(' NCAA', '', regex=False).str.strip()
                    df_r['Slug'] = df_r['School_Raw'].apply(lambda x: r_slug_map.get(x) or r_slug_map.get(x + " NCAA"))
                    
                    if 'Conf' in df_r.columns:
                        logger.info(f"Ratings DataFrame has Conf column. Sample: {df_r['Conf'].head()}")
                        
                    if 'Conf' in df_r.columns and 'Slug' in df_r.columns:
                        conf_map = df_r.set_index('Slug')['Conf'].to_dict()
                        df['Conf'] = df['Slug'].map(conf_map)
                        logger.info(f"Merged Conf column. Non-null count: {df['Conf'].count()}")
                    else:
                         logger.error(f"Ratings DataFrame missing columns. Conf: {'Conf' in df_r.columns}, Slug: {'Slug' in df_r.columns}")
                else:
                    logger.error("Ratings table (id='ratings') NOT found in content")
            except Exception as e:
                logger.error(f"Error processing ratings for conference info: {e}")
        else:
             logger.error("Ratings content is empty or None")
        
        logger.info(f"Team Stats Columns: {df.columns}")
        return df

    def get_game_results(self, season, stats_df=None, limit=30, conference=None):
        """
        Scrapes game results by iterating through top teams' schedules.
        stats_df: DataFrame returned by get_team_stats (must contain 'Slug').
        :param conference: Optional conference to filter by (e.g. "ACC", "Big 10")
        """
        if stats_df is None:
             logger.error("stats_df is None")
             return None
        logger.info(f"Game Results received stats_df columns: {stats_df.columns}")
        if 'Slug' not in stats_df.columns:
            logger.error("stats_df with Slugs required for partial scraping.")
            return None
        
        # Filter by Conference if provided
        filtered_df = stats_df.copy()
        if conference:
             # Fuzzy match or direct match
             # 'Conf' column expected from get_team_stats
             if 'Conf' in filtered_df.columns:
                 # Normalize to uppercase for comparison
                 # Check strict match first?
                 # Sports-Ref uses "ACC", "Big Ten", "SEC", etc.
                 target_conf = conference.strip().lower()
                 filtered_df = filtered_df[filtered_df['Conf'].str.lower() == target_conf]
                 logger.info(f"Filtered to {len(filtered_df)} teams in {conference}")
                 
                 # If we are filtering by conference, we probably want ALL teams in that conference, not just top 30
                 # So we ignore 'limit' implicitly by taking the whole filtered_df?
                 # Or we can still respect limit if it's very large. 
                 # Let's set limit to len(filtered_df) if conference is set.
                 limit = len(filtered_df)
             else:
                 logger.warning("Conference column not found in stats_df. Ignoring conference filter.")

        # Sort by SRS (Simple Rating System) to get top teams, or just take top of list (usually sorted by school name?)
        # Sports-Reference table is typically alphabetical.
        # Verify SRS exists
        if 'SRS' in filtered_df.columns:
            top_teams = filtered_df.sort_values('SRS', ascending=False).head(limit)
        else:
            top_teams = filtered_df.head(limit)
            
        # logger.info(f"Top teams to scrape (limit={limit}):\n{top_teams[['School', 'Conf', 'Slug']].head(20)}")
        
        all_games = []
        
        logger.info(f"Scraping schedules for top {len(top_teams)} teams...")
        
        for _, team in top_teams.iterrows():
            slug = team['Slug']
            school = team['School']
            if not slug:
                continue
                
            # /cbb/schools/duke/men/2025-schedule.html
            url = f"https://www.sports-reference.com/cbb/schools/{slug}/men/{season}-schedule.html"
            content = self._make_request(url)
            
            if not content:
                continue
                
            try:
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', {'id': 'schedule'})
                if not table:
                    continue
                    
                df_schedule = pd.read_html(StringIO(str(table)))[0]
                
                # Clean up
                df_schedule = df_schedule[df_schedule['Date'] != 'Date']
                
                # Cols: G, Date, Time, Type, Opponent, Conf, Tm, Opp, OT, W/L, Streak, Arena
                # We need: Date, Visitor, Home, Visitor_Pts, Home_Pts
                
                # Map columns based on 'Type' (vs / @)
                # 'Type' column is often empty for home, 'N' for neutral, '@' for away
                # But sometimes it's explicitly 'Location'?
                # Let's check typical columns. 
                # [Date, Time, Type, Opponent, Result, Tm, Opp, ...]
                # Type: 'nan' (Home), '@' (Away), 'N' (Neutral)
                
                # We need to standardize to: Date, Visitor, Visitor_Pts, Home, Home_Pts
                
                # Renaming
                # 'Tm' -> School Points
                # 'Opp' -> Opponent Points
                # 'Opponent' -> Opponent Name
                
                schedule_cleaned = []
                for _, row in df_schedule.iterrows():
                    loc = row.get('Type')
                    opponent = row.get('Opponent')
                    tm_pts = row.get('Tm')
                    opp_pts = row.get('Opp')
                    date = row.get('Date')
                    
                    if pd.isna(tm_pts) or pd.isna(opp_pts):
                        continue
                        
                    # Normalize Location
                    # if loc == '@', School is Visitor
                    # if loc == 'N', Neutral (treat as Home/Visitor arbitrary, or use distinct flag in processor)
                    # if loc is nan, School is Home
                    
                    # Also remove rankings from Opponent name (e.g. "(10) Duke")
                    opponent = str(opponent).split(') ')[-1] if ')' in str(opponent) else str(opponent)
                    
                    if loc == '@':
                        visitor = school
                        visitor_pts = tm_pts
                        home = opponent
                        home_pts = opp_pts
                    elif loc == 'N':
                         # Neutral
                         visitor = school
                         visitor_pts = tm_pts
                         home = opponent
                         home_pts = opp_pts
                    else:
                        # Home
                        visitor = opponent
                        visitor_pts = opp_pts
                        home = school
                        home_pts = tm_pts
                        
                    schedule_cleaned.append({
                        'Date': date,
                        'Visitor': visitor,
                        'Visitor_Pts': visitor_pts,
                        'Home': home,
                        'Home_Pts': home_pts
                    })
                    
                all_games.extend(schedule_cleaned)
                
            except Exception as e:
                logger.error(f"Error processing schedule for {school}: {e}")
                continue

        return pd.DataFrame(all_games)
