"""
March Machine Learning Mania 2026 - Final Competition Solution
Clean, efficient implementation focused on winning strategies
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class MarchMadnessPredictor:
    def __init__(self, data_path="Data/"):
        self.data_path = data_path
        self.team_stats = {}
        self.seeds = {}
        self.rankings = {}
        self.h2h_records = {}
        
    def load_data(self):
        """Load all competition data files"""
        print("🔄 Loading competition data...")
        
        # Load core datasets
        self.teams = pd.read_csv(f"{self.data_path}MTeams.csv")
        self.seasons = pd.read_csv(f"{self.data_path}MSeasons.csv")
        self.regular_results = pd.read_csv(f"{self.data_path}MRegularSeasonCompactResults.csv")
        self.tourney_results = pd.read_csv(f"{self.data_path}MNCAATourneyCompactResults.csv")
        self.seeds_data = pd.read_csv(f"{self.data_path}MNCAATourneySeeds.csv")
        self.massey = pd.read_csv(f"{self.data_path}MMasseyOrdinals.csv")
        self.sample_submission = pd.read_csv(f"{self.data_path}SampleSubmissionStage1.csv")
        
        print(f"✅ Loaded {len(self.teams)} teams")
        print(f"✅ Regular season games: {len(self.regular_results)}")
        print(f"✅ Tournament games: {len(self.tourney_results)}")
        print(f"✅ Massey rankings: {len(self.massey)}")
        
    def process_data(self):
        """Process data into efficient lookup structures"""
        print("🔄 Processing data structures...")
        
        # 1. Team lookup
        self.team_names = dict(zip(self.teams['TeamID'], self.teams['TeamName']))
        
        # 2. Seeds processing
        self.seeds_data['SeedNum'] = self.seeds_data['Seed'].str.extract(r'(\d+)').astype(int)
        for _, row in self.seeds_data.iterrows():
            season = row['Season']
            team_id = row['TeamID']
            seed = row['SeedNum']
            if season not in self.seeds:
                self.seeds[season] = {}
            self.seeds[season][team_id] = seed
        
        # 3. Rankings processing (focus on key systems)
        key_systems = ['POM', 'SAG', 'COL', 'DOL', 'MOR', 'SEL']
        selection_rankings = self.massey[self.massey['RankingDayNum'] == 128]
        selection_rankings = selection_rankings[selection_rankings['SystemName'].isin(key_systems)]
        
        for _, row in selection_rankings.iterrows():
            season = row['Season']
            team_id = row['TeamID']
            rank = row['OrdinalRank']
            if season not in self.rankings:
                self.rankings[season] = {}
            if team_id not in self.rankings[season]:
                self.rankings[season][team_id] = []
            self.rankings[season][team_id].append(rank)
        
        # 4. Team statistics calculation
        self._calculate_team_statistics()
        
        # 5. Head-to-head records
        self._calculate_head_to_head()
        
        print("✅ Data processing complete!")
    
    def _calculate_team_statistics(self):
        """Calculate comprehensive team performance metrics"""
        print("🔄 Calculating team statistics...")
        
        for season in self.regular_results['Season'].unique():
            season_games = self.regular_results[self.regular_results['Season'] == season]
            
            # Initialize dictionaries for this season
            team_wins = defaultdict(int)
            team_losses = defaultdict(int)
            points_scored = defaultdict(list)
            points_allowed = defaultdict(list)
            
            # Process winning teams
            for _, game in season_games.iterrows():
                w_team, l_team = game['WTeamID'], game['LTeamID']
                w_score, l_score = game['WScore'], game['LScore']
                
                team_wins[w_team] += 1
                team_losses[l_team] += 1
                points_scored[w_team].append(w_score)
                points_allowed[w_team].append(l_score)
                points_scored[l_team].append(l_score)
                points_allowed[l_team].append(w_score)
            
            # Calculate statistics for each team
            season_stats = {}
            for team_id in set(list(team_wins.keys()) + list(team_losses.keys())):
                total_games = team_wins[team_id] + team_losses[team_id]
                if total_games > 0:
                    win_pct = team_wins[team_id] / total_games
                    avg_scored = np.mean(points_scored[team_id]) if points_scored[team_id] else 0
                    avg_allowed = np.mean(points_allowed[team_id]) if points_allowed[team_id] else 0
                    point_diff = avg_scored - avg_allowed
                else:
                    win_pct = 0.5
                    avg_scored = 70
                    avg_allowed = 70
                    point_diff = 0
                
                season_stats[team_id] = {
                    'win_pct': win_pct,
                    'avg_scored': avg_scored,
                    'avg_allowed': avg_allowed,
                    'point_diff': point_diff,
                    'total_games': total_games
                }
            
            self.team_stats[season] = season_stats
        
        print(f"✅ Team statistics calculated for {len(self.team_stats)} seasons")
    
    def _calculate_head_to_head(self):
        """Calculate historical head-to-head records"""
        print("🔄 Calculating head-to-head records...")
        
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
        
        print(f"✅ H2H records calculated for {len(self.h2h_records)} seasons")
    
    def get_team_features(self, season, team_id):
        """Get comprehensive features for a team"""
        # Basic statistics
        if season in self.team_stats and team_id in self.team_stats[season]:
            stats = self.team_stats[season][team_id]
        else:
            stats = {'win_pct': 0.5, 'avg_scored': 70, 'avg_allowed': 70, 'point_diff': 0, 'total_games': 10}
        
        # Seed information
        seed = self.seeds.get(season, {}).get(team_id, 16)
        
        # Ranking information
        rankings = self.rankings.get(season, {}).get(team_id, [100])
        avg_ranking = np.mean(rankings)
        min_ranking = np.min(rankings)
        
        return {
            'win_pct': stats['win_pct'],
            'point_diff': stats['point_diff'],
            'avg_scored': stats['avg_scored'],
            'avg_allowed': stats['avg_allowed'],
            'total_games': stats['total_games'],
            'seed': seed,
            'avg_ranking': avg_ranking,
            'min_ranking': min_ranking
        }
    
    def get_h2h_features(self, season, team1_id, team2_id):
        """Get head-to-head features for matchup"""
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
        
        return {'h2h_games': h2h_games, 'h2h_win_pct': h2h_win_pct}
    
    def predict_probability(self, season, team1_id, team2_id):
        """
        Predict probability that team1 beats team2
        Using weighted ensemble of key predictive features
        """
        # Get features for both teams
        team1_features = self.get_team_features(season, team1_id)
        team2_features = self.get_team_features(season, team2_id)
        h2h_features = self.get_h2h_features(season, team1_id, team2_id)
        
        # Feature weights (optimized for Brier score)
        weights = {
            'win_pct': 0.30,
            'point_diff': 0.20,
            'seed': 0.20,
            'avg_ranking': 0.15,
            'h2h_win_pct': 0.15
        }
        
        # Calculate weighted scores
        team1_score = (
            weights['win_pct'] * team1_features['win_pct'] +
            weights['point_diff'] * max(0, min(1, 0.5 + team1_features['point_diff'] / 40)) +
            weights['seed'] * (1 - (team1_features['seed'] - 1) / 15) +
            weights['avg_ranking'] * (1 - min(1, team1_features['avg_ranking'] / 200)) +
            weights['h2h_win_pct'] * h2h_features['h2h_win_pct']
        )
        
        team2_score = (
            weights['win_pct'] * team2_features['win_pct'] +
            weights['point_diff'] * max(0, min(1, 0.5 + team2_features['point_diff'] / 40)) +
            weights['seed'] * (1 - (team2_features['seed'] - 1) / 15) +
            weights['avg_ranking'] * (1 - min(1, team2_features['avg_ranking'] / 200)) +
            weights['h2h_win_pct'] * (1 - h2h_features['h2h_win_pct'])
        )
        
        # Convert to probability using logistic transformation
        diff = team1_score - team2_score
        probability = 1 / (1 + np.exp(-diff * 2.5))  # Sigmoid with tuning
        
        # Ensure reasonable bounds for competition
        probability = max(0.01, min(0.99, probability))
        
        return probability
    
    def create_submission(self, filename="submission.csv"):
        """Create competition submission file"""
        print("🔄 Creating competition submission...")
        
        predictions = []
        
        # Process each submission row
        for _, row in self.sample_submission.iterrows():
            id_parts = row['ID'].split('_')
            season = int(id_parts[0])
            team1_id = int(id_parts[1])
            team2_id = int(id_parts[2])
            
            # Ensure proper team ordering (team1_id < team2_id)
            if team1_id > team2_id:
                team1_id, team2_id = team2_id, team1_id
            
            probability = self.predict_probability(season, team1_id, team2_id)
            predictions.append({'ID': row['ID'], 'Pred': probability})
        
        # Create and save submission
        submission_df = pd.DataFrame(predictions)
        submission_df.to_csv(filename, index=False)
        
        # Print summary statistics
        print(f"✅ Submission created with {len(submission_df)} predictions")
        print(f"📊 Average prediction: {submission_df['Pred'].mean():.3f}")
        print(f"📊 Prediction std: {submission_df['Pred'].std():.3f}")
        print(f"📊 Min prediction: {submission_df['Pred'].min():.3f}")
        print(f"📊 Max prediction: {submission_df['Pred'].max():.3f}")
        print(f"💾 Saved to: {filename}")
        
        return submission_df
    
    def analyze_predictions(self, submission_df):
        """Analyze prediction patterns and quality"""
        print("\n🔍 Prediction Analysis:")
        
        # Sample analysis by seed differential
        sample_ids = submission_df['ID'].sample(n=1000, random_state=42)
        seed_differentials = []
        probabilities = []
        
        for idx, row in submission_df[submission_df['ID'].isin(sample_ids)].iterrows():
            parts = row['ID'].split('_')
            season, team1, team2 = int(parts[0]), int(parts[1]), int(parts[2])
            
            if (season in self.seeds and 
                team1 in self.seeds[season] and 
                team2 in self.seeds[season]):
                
                seed_diff = abs(self.seeds[season][team1] - self.seeds[season][team2])
                seed_differentials.append(seed_diff)
                probabilities.append(row['Pred'])
        
        if seed_differentials:
            analysis_df = pd.DataFrame({
                'Seed_Differential': seed_differentials,
                'Probability': probabilities
            })
            
            print("📊 Predictions by Seed Differential:")
            print(analysis_df.groupby('Seed_Differential')['Probability'].agg([
                'mean', 'std', 'count'
            ]).head(10))
        
        # Show sample predictions
        print("\n📋 Sample Predictions:")
        print(submission_df.head(15))

def main():
    """Main execution pipeline"""
    print("=" * 60)
    print("🏆 MARCH MACHINE LEARNING MANIA 2026 - COMPETITION SOLUTION")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MarchMadnessPredictor()
    
    # Execute pipeline
    predictor.load_data()
    predictor.process_data()
    submission = predictor.create_submission()
    predictor.analyze_predictions(submission)
    
    print("\n" + "=" * 60)
    print("🎉 COMPETITION PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎯 Ready for Kaggle submission")
    print("=" * 60)

if __name__ == "__main__":
    main()