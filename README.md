# March Madness Predictor

A Machine Learning application that saves you from bracket-busting heartbreak (maybe). This tool scrapes historical college basketball data from [Sports-Reference](https://www.sports-reference.com/cbb/), trains a Random Forest classifier, and predicts the probability of a win for any given matchup.

## Features

- **Automated Scraping**: Fetches team statistics and game results directly from the web.
- **Robust Training**: Uses data from the previous season to train the model and current season data for predictions.
- **Home Court Advantage**: Accounts for game location (Home/Away/Neutral) in its predictions.
- **Caching**: Saves scraped data locally to `cache/` to respect rate limits and speed up subsequent runs.

## Setup

### Prerequisites
- Python 3.8+
- `pip`

### Installation

1.  **Clone or Download** the repository.
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

**Note**: Always ensure your virtual environment is activated (`source venv/bin/activate`) before running the script.

### Basic Prediction
To predict a game where the first team is playing at home:

```bash
python src/main.py --team_a "Duke" --team_b "North Carolina" --home_court
```

### Neutral Site
For tournament games (March Madness), usually played on neutral courts:

```bash
python src/main.py --team_a "Gonzaga" --team_b "Kansas" --neutral
```

### Arguments
- `--team_a`: Name of the first team (e.g., "Duke", "UConn").
- `--team_b`: Name of the second team.
- `--home_court`: Flag to indicate `team_a` is the home team.
- `--neutral`: Flag to indicate a neutral venue (overrides `--home_court`).

## How It Works

1.  **Scraping**: The app checks the `cache/` directory. If data is missing or outdated, it scrapes team stats (Efficiency, Shooting, etc.) and schedule results from Sports-Reference.
2.  **Processing**: It calculates the *difference* between the two teams' stats for every historical matchup to create a feature vector.
3.  **Training**: A Random Forest Classifier is trained on these historical differentials.
4.  **Prediction**: The model evaluates the current matchup's stat differentials to output a win probability for `team_a`.

## Disclaimer
This tool is for educational purposes only. Do not use for gambling. The randomness of March Madness is undefeated.
