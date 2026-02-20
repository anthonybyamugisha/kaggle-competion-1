"""
March Machine Learning Mania 2026 - Winning Strategy Implementation
This script implements core predictive features for NCAA tournament prediction
without external ML dependencies
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NCAAPredictor:
    def __init__(self, data_path="Data/"):
        self.data_path = data_path
        self.team_stats = {}
        self.seeds = {}
        self.rankings = {}
        self.h2h_records = {}
        
    def load_and_process_data(self):
        """Load and process all required data files"""
        print("Loading competition data...")
        
        # Load core data files
        self.teams = pd.read_csv(f"{self.data_path}MTeams.csv")
        self.seasons = pd.read_csv(f"{self.data_path}MSeasons.csv")
        self.regular_results = pd.read_csv(f"{self.data_path}MRegularSeasonCompactResults.csv")
        self.tourney_results = pd.read_csv(f"{self.data_path}MNCAATourneyCompactResults.csv")
        self.seeds_data = pd.read_csv(f"{self.data_path}MNCAATourneySeeds.csv")
        self.massey = pd.read_csv(f"{self.data_path}MMasseyOrdinals.csv")
        self.sample_submission = pd.read_csv(f"{self.data_path}SampleSubmissionStage1.csv")
        
        print(f"Loaded {len(self.teams)} teams")
        print(f"Regular season games: {len(self.regular_results)}")
        print(f"Tournament games: {len(self.tourney_results)}")
        print(f"Massey rankings: {len(self.massey)}")
        
        # Process data into efficient structures
        self._process_teams()
        self._process_seeds()
        self._process_rankings()
        self._calculate_team_stats()
        self._calculate_h2h_records()
        
        print("Data processing complete!")
    
    def _process_teams(self):
        """Create team lookup dictionaries"""
        self.team_lookup = dict(zip(self.teams['TeamID'], self.teams['TeamName']))
        print(f"Team lookup created with {len(self.team_lookup)} teams")
    
    def _process_seeds(self):
        """Process tournament seeds by season"""
        # Extract seed numbers
        self.seeds_data['SeedNum'] = self.seeds_data['Seed'].str.extract(r'(\d+)').astype(int)
        
        # Create season-team-seed mapping
        for _, row in self.seeds_data.iterrows():
            season = row['Season']
            team_id = row['TeamID']
            seed = row['SeedNum']
            
            if season not in self.seeds:
                self.seeds[season] = {}
            self.seeds[season][team_id] = seed
        
        print(f"Seeds processed for {len(self.seeds)} seasons")
    
    def _process_rankings(self):
        """Process Massey ordinal rankings"""
        # Focus on key ranking systems and Selection Sunday rankings
        key_systems = ['POM', 'SAG', 'COL', 'DOL', 'MOR', 'SEL']  # Major ranking systems
        
        # Get rankings from day 128 (Selection Sunday)
        selection_sunday_rankings = self.massey[self.massey['RankingDayNum'] == 128].copy()
        
        # Filter to key systems
        selection_sunday_rankings = selection_sunday_rankings[
            selection_sunday_rankings['SystemName'].isin(key_systems)
        ]
        
        # Create team-season-ranking mapping
        for _, row in selection_sunday_rankings.iterrows():
            season = row['Season']
            team_id = row['TeamID']
            system = row['SystemName']
            rank = row['OrdinalRank']
            
            if season not in self.rankings:
                self.rankings[season] = {}
            if team_id not in self.rankings[season]:
                self.rankings[season][team_id] = {}
            self.rankings[season][team_id][system] = rank
        
        print(f"Rankings processed for {len(self.rankings)} seasons")
    
    def _calculate_team_stats(self):
        """Calculate comprehensive team statistics"""
        print("Calculating team statistics...")
        
        # Group by season and team
        for season in self.regular_results['Season'].unique():
            season_games = self.regular_results[self.regular_results['Season'] == season]
            
            # Calculate winning team stats
            w_team_stats = season_games.groupby('WTeamID').agg({
                'WScore': ['mean', 'std', 'count'],
                'LScore': 'mean'
            }).reset_index()
            w_team_stats.columns = ['TeamID', 'Points_Scored_Mean', 'Points_Scored_Std', 'Games_Won', 'Points_Allowed_When_Won']
            
            # Calculate losing team stats
            l_team_stats = season_games.groupby('LTeamID').agg({
                'LScore': ['mean', 'std', 'count'],
                'WScore': 'mean'
            }).reset_index()
            l_team_stats.columns = ['TeamID', 'Points_Scored_Mean_Loss', 'Points_Scored_Std_Loss', 'Games_Lost', 'Points_Allowed_When_Lost']
            
            # Merge and calculate overall stats
            team_stats = pd.merge(w_team_stats, l_team_stats, on='TeamID', how='outer')
            team_stats = team_stats.fillna(0)
            
            # Calculate derived metrics
            team_stats['Total_Games'] = team_stats['Games_Won'] + team_stats['Games_Lost']
            team_stats['Win_Pct'] = team_stats['Games_Won'] / team_stats['Total_Games']
            team_stats['Avg_Points_Scored'] = (
                (team_stats['Points_Scored_Mean'] * team_stats['Games_Won'] + 
                 team_stats['Points_Scored_Mean_Loss'] * team_stats['Games_Lost']) / 
                team_stats['Total_Games']
            )
            team_stats['Avg_Points_Allowed'] = (
                (team_stats['Points_Allowed_When_Won'] * team_stats['Games_Won'] + 
                 team_stats['Points_Allowed_When_Lost'] * team_stats['Games_Lost']) / 
                team_stats['Total_Games']
            )
            team_stats['Point_Differential'] = team_stats['Avg_Points_Scored'] - team_stats['Avg_Points_Allowed']
            
            # Store for this season
            self.team_stats[season] = team_stats.set_index('TeamID')
        
        print(f"Team statistics calculated for {len(self.team_stats)} seasons")
    
    def _calculate_h2h_records(self):
        """Calculate historical head-to-head records"""
        print("Calculating head-to-head records...")
        
        # Create matchup records
        for season in self.regular_results['Season'].unique():
            season_games = self.regular_results[self.regular_results['Season'] == season]
            self.h2h_records[season] = defaultdict(lambda: {'games': 0, 'wins': 0})
            
            for _, game in season_games.iterrows():
                team1, team2 = sorted([game['WTeamID'], game['LTeamID']])
                winner = game['WTeamID']
                
                key = (team1, team2)
                self.h2h_records[season][key]['games'] += 1
                if winner == team1:
                    self.h2h_records[season][key]['wins'] += 1
        
        print(f"H2H records calculated for {len(self.h2h_records)} seasons")
    
    def get_team_features(self, season, team_id):
        """Get comprehensive features for a team in a season"""
        features = {}
        
        # Basic team stats
        if season in self.team_stats and team_id in self.team_stats[season].index:
            team_data = self.team_stats[season].loc[team_id]
            features.update({
                'win_pct': team_data['Win_Pct'],
                'avg_points_scored': team_data['Avg_Points_Scored'],
                'avg_points_allowed': team_data['Avg_Points_Allowed'],
                'point_differential': team_data['Point_Differential'],
                'total_games': team_data['Total_Games']
            })
        else:
            # Default values for missing data
            features.update({
                'win_pct': 0.5,
                'avg_points_scored': 70,
                'avg_points_allowed': 70,
                'point_differential': 0,
                'total_games': 10
            })
        
        # Seed information
        if season in self.seeds and team_id in self.seeds[season]:
            features['seed'] = self.seeds[season][team_id]
        else:
            features['seed'] = 16  # Default high seed
        
        # Ranking information (average of available systems)
        if season in self.rankings and team_id in self.rankings[season]:
            rankings = list(self.rankings[season][team_id].values())
            features['avg_ranking'] = np.mean(rankings) if rankings else 100
            features['min_ranking'] = np.min(rankings) if rankings else 100
            features['max_ranking'] = np.max(rankings) if rankings else 100
        else:
            features['avg_ranking'] = 100
            features['min_ranking'] = 100
            features['max_ranking'] = 100
        
        return features
    
    def get_h2h_features(self, season, team1_id, team2_id):
        """Get head-to-head features for a matchup"""
        key = tuple(sorted([team1_id, team2_id]))
        
        if season in self.h2h_records and key in self.h2h_records[season]:
            record = self.h2h_records[season][key]
            h2h_games = record['games']
            if team1_id < team2_id:
                h2h_wins = record['wins']
            else:
                h2h_wins = h2h_games - record['wins']
            
            h2h_win_pct = h2h_wins / h2h_games if h2h_games > 0 else 0.5
        else:
            h2h_games = 0
            h2h_win_pct = 0.5
        
        return {
            'h2h_games': h2h_games,
            'h2h_win_pct': h2h_win_pct
        }
    
    def predict_matchup(self, season, team1_id, team2_id):
        """
        Predict probability that team1 beats team2
        Using weighted ensemble of features
        """
        # Get features for both teams
        team1_features = self.get_team_features(season, team1_id)
        team2_features = self.get_team_features(season, team2_id)
        h2h_features = self.get_h2h_features(season, team1_id, team2_id)
        
        # Feature weights (optimized through experimentation)
        weights = {
            'win_pct': 0.25,
            'point_differential': 0.20,
            'seed': 0.15,
            'avg_ranking': 0.15,
            'h2h_win_pct': 0.25
        }
        
        # Calculate weighted scores
        team1_score = (
            weights['win_pct'] * team1_features['win_pct'] +
            weights['point_differential'] * max(0, min(1, 0.5 + team1_features['point_differential'] / 50)) +
            weights['seed'] * (1 - (team1_features['seed'] - 1) / 15) +
            weights['avg_ranking'] * (1 - min(1, team1_features['avg_ranking'] / 200)) +
            weights['h2h_win_pct'] * h2h_features['h2h_win_pct']
        )
        
        team2_score = (
            weights['win_pct'] * team2_features['win_pct'] +
            weights['point_differential'] * max(0, min(1, 0.5 + team2_features['point_differential'] / 50)) +
            weights['seed'] * (1 - (team2_features['seed'] - 1) / 15) +
            weights['avg_ranking'] * (1 - min(1, team2_features['avg_ranking'] / 200)) +
            weights['h2h_win_pct'] * (1 - h2h_features['h2h_win_pct'])
        )
        
        # Convert to probability (logistic-like transformation)
        diff = team1_score - team2_score
        probability = 1 / (1 + np.exp(-diff * 2))  # Sigmoid function with scaling
        
        # Ensure reasonable bounds
        probability = max(0.01, min(0.99, probability))
        
        return probability
    
    def create_submission(self, submission_file="submission.csv"):
        """Create Stage 1 submission file"""
        print("Creating submission file...")
        
        # Parse sample submission IDs
        submission_data = []
        for _, row in self.sample_submission.iterrows():
            id_parts = row['ID'].split('_')
            season = int(id_parts[0])
            team1_id = int(id_parts[1])
            team2_id = int(id_parts[2])
            
            # Always ensure team1_id < team2_id as per competition format
            if team1_id > team2_id:
                team1_id, team2_id = team2_id, team1_id
            
            probability = self.predict_matchup(season, team1_id, team2_id)
            submission_data.append({
                'ID': row['ID'],
                'Pred': probability
            })
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(submission_file, index=False)
        
        print(f"Submission created with {len(submission_df)} predictions")
        print(f"Average prediction: {submission_df['Pred'].mean():.3f}")
        print(f"Prediction std: {submission_df['Pred'].std():.3f}")
        print(f"Saved to: {submission_file}")
        
        return submission_df

def main():
    """Main execution function"""
    print("=" * 50)
    print("MARCH MACHINE LEARNING MANIA 2026 - WINNING STRATEGY")
    print("=" * 50)
    
    # Initialize predictor
    predictor = NCAAPredictor()
    
    # Load and process all data
    predictor.load_and_process_data()
    
    # Create submission
    submission = predictor.create_submission()
    
    # Show sample predictions
    print("\nSample Predictions:")
    print(submission.head(10))
    
    # Show statistics by seed
    print("\nPrediction Analysis by Seed Differential:")
    sample_ids = submission['ID'].sample(n=1000, random_state=42)
    seed_diffs = []
    probs = []
    
    for idx, row in submission[submission['ID'].isin(sample_ids)].iterrows():
        parts = row['ID'].split('_')
        season, team1, team2 = int(parts[0]), int(parts[1]), int(parts[2])
        
        if season in predictor.seeds and team1 in predictor.seeds[season] and team2 in predictor.seeds[season]:
            seed_diff = abs(predictor.seeds[season][team1] - predictor.seeds[season][team2])
            seed_diffs.append(seed_diff)
            probs.append(row['Pred'])
    
    if seed_diffs:
        seed_df = pd.DataFrame({'Seed_Diff': seed_diffs, 'Probability': probs})
        print(seed_df.groupby('Seed_Diff')['Probability'].agg(['mean', 'std', 'count']).head(10))
    
    print("\n✓ Prediction pipeline completed successfully!")
    print("✓ Ready for competition submission")

if __name__ == "__main__":
    main()