
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.scraper import CBBScraper

class TestConferenceFiltering(unittest.TestCase):
    def setUp(self):
        self.scraper = CBBScraper()

    @patch('src.scraper.CBBScraper._make_request')
    def test_conference_filtering(self, mock_request):
        # Mock HTML for school-stats
        # Simple table with Duke and Texas Tech
        stats_html = """
        <html>
        <table id="basic_school_stats">
        <thead>
            <tr><th>Rk</th><th>School</th><th>SRS</th><th>W</th><th>L</th></tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td data-stat="school_name"><a href="/cbb/schools/duke/men/2025.html">Duke</a></td>
                <td>20.5</td><td>25</td><td>5</td>
            </tr>
            <tr>
                <td>2</td>
                <td data-stat="school_name"><a href="/cbb/schools/texas-tech/men/2025.html">Texas Tech</a></td>
                <td>18.5</td><td>22</td><td>8</td>
            </tr>
        </tbody>
        </table>
        </html>
        """
        
        # Mock HTML for ratings (includes Conf)
        ratings_html = """
        <html>
        <table id="ratings">
        <thead>
            <tr><th>Rk</th><th>School</th><th>Conf</th><th>SRS</th></tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td data-stat="school_name"><a href="/cbb/schools/duke/men/2025.html">Duke</a></td>
                <td>ACC</td>
                <td>20.5</td>
            </tr>
            <tr>
                <td>2</td>
                <td data-stat="school_name"><a href="/cbb/schools/texas-tech/men/2025.html">Texas Tech</a></td>
                <td>Big 12</td>
                <td>18.5</td>
            </tr>
        </tbody>
        </table>
        </html>
        """

        # Side effect: return stats_html for school-stats, ratings_html for ratings
        def side_effect(url):
            if "school-stats" in url:
                return stats_html.encode('utf-8')
            if "ratings" in url:
                return ratings_html.encode('utf-8')
            return None
        
        mock_request.side_effect = side_effect
        
        # 1. Get Team Stats
        print("Fetching team stats...")
        df = self.scraper.get_team_stats(2025)
        
        print("Team Stats Columns:", df.columns)
        print(df[['School', 'Conf', 'Slug']])
        
        self.assertIn('Conf', df.columns)
        self.assertEqual(df.loc[df['School'] == 'Duke', 'Conf'].values[0], 'ACC')
        self.assertEqual(df.loc[df['School'] == 'Texas Tech', 'Conf'].values[0], 'Big 12')
        
        # 2. Test get_game_results filtering
        # Mock schedule fetch (return empty or minimal)
        # We assume if filter works, loop only triggers for Duke
        
        # Mock _make_request again to track calls?
        # Actually current mock is fine, we just want to see top_teams
        
        # We can't easily inspect internal 'top_teams' unless we modify scraper or mock 'pd.read_html' inside get_game_results
        # But we can call get_game_results and check the calls to _make_request for schedules
        
        mock_request.reset_mock()
        mock_request.side_effect = side_effect # Restore
        
        print("\nCalling get_game_results with conference='ACC'...")
        self.scraper.get_game_results(2025, df, conference='ACC')
        
        # Check calls
        calls = mock_request.call_args_list
        fetch_urls = [c[0][0] for c in calls]
        print("Fetched URLs:", fetch_urls)
        
        duke_schedule = '/cbb/schools/duke/men/2025-schedule.html' in str(fetch_urls)
        tt_schedule = '/cbb/schools/texas-tech/men/2025-schedule.html' in str(fetch_urls)
        
        self.assertTrue(duke_schedule, "Should fetch Duke schedule")
        self.assertFalse(tt_schedule, "Should NOT fetch Texas Tech schedule")

if __name__ == '__main__':
    unittest.main()
