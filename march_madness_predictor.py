import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

class MarchMadnessPredictor:
    def __init__(self, data_path="Data/"):
        self.data_path = data_path
        self.models = {}
        self.feature_names = []
        
    def load_data(self):
        """Load all necessary data files"""
        print("Loading data...")
        self.teams = pd.read_csv(f"{self.data_path}MTeams.csv")
        self.seasons = pd.read_csv(f"{self.data_path}MSeasons.csv")
        self.regular_results = pd.read_csv(f"{self.data_path}MRegularSeasonCompactResults.csv")
        self.tourney_results = pd.read_csv(f"{self.data_path}MNCAATourneyCompactResults.csv")
        self.seeds = pd.read_csv(f"{self.data_path}MNCAATourneySeeds.csv")
        self.massey = pd.read_csv(f"{self.data_path}MMasseyOrdinals.csv")
        self.sample_submission = pd.read_csv(f"{self.data_path}SampleSubmissionStage1.csv")
        print("Data loaded successfully!")
        
    def create_features(self):
        """Create predictive features from historical data"""
        print("Creating features...")
        
        # 1. Team-level statistics
        team_stats = self._calculate_team_stats()
        
        # 2. Seed information
        seed_features = self._process_seeds()
        
        # 3. Massey rankings
        ranking_features = self._process_rankings()
        
        # 4. Head-to-head historical records
        h2h_features = self._calculate_h2h_records()
        
        # 5. Recent form (last 10 games)
        form_features = self._calculate_recent_form()
        
        # Combine all features
        self.features = {
            'team_stats': team_stats,
            'seeds': seed_features,
            'rankings': ranking_features,
            'h2h': h2h_features,
            'form': form_features
        }
        
        print("Features created!")
        return self.features
    
    def _calculate_team_stats(self):
        """Calculate comprehensive team statistics"""
        # Winning team stats
        w_stats = self.regular_results.groupby(['Season', 'WTeamID']).agg({
            'WScore': ['mean', 'std', 'count'],
            'LScore': 'mean'  # Opponent score when they won
        }).reset_index()
        w_stats.columns = ['Season', 'TeamID', 'WPoints_Mean', 'WPoints_Std', 'WGames', 'WOppPoints_Mean']
        
        # Losing team stats
        l_stats = self.regular_results.groupby(['Season', 'LTeamID']).agg({
            'LScore': ['mean', 'std', 'count'],
            'WScore': 'mean'  # Opponent score when they lost
        }).reset_index()
        l_stats.columns = ['Season', 'TeamID', 'LPoints_Mean', 'LPoints_Std', 'LGames', 'LOppPoints_Mean']
        
        # Merge and calculate overall stats
        team_stats = pd.merge(w_stats, l_stats, on=['Season', 'TeamID'], how='outer')
        team_stats = team_stats.fillna(0)
        
        # Calculate derived metrics
        team_stats['TotalGames'] = team_stats['WGames'] + team_stats['LGames']
        team_stats['WinPct'] = team_stats['WGames'] / team_stats['TotalGames']
        team_stats['Points_Mean'] = (team_stats['WPoints_Mean'] * team_stats['WGames'] + 
                                   team_stats['LPoints_Mean'] * team_stats['LGames']) / team_stats['TotalGames']
        team_stats['OppPoints_Mean'] = (team_stats['WOppPoints_Mean'] * team_stats['WGames'] + 
                                      team_stats['LOppPoints_Mean'] * team_stats['LGames']) / team_stats['TotalGames']
        team_stats['PointDifferential'] = team_stats['Points_Mean'] - team_stats['OppPoints_Mean']
        team_stats['Points_Std'] = np.sqrt((team_stats['WPoints_Std']**2 * team_stats['WGames'] + 
                                          team_stats['LPoints_Std']**2 * team_stats['LGames']) / team_stats['TotalGames'])
        
        return team_stats
    
    def _process_seeds(self):
        """Process tournament seed information"""
        # Extract seed number from seed string (e.g., 'W01' -> 1)
        self.seeds['SeedNum'] = self.seeds['Seed'].str.extract(r'(\d+)').astype(int)
        return self.seeds
    
    def _process_rankings(self):
        """Process Massey ordinal rankings"""
        # Get latest rankings for each season (RankingDayNum = 128 is typically Selection Sunday)
        latest_rankings = self.massey[self.massey['RankingDayNum'] == 128].copy()
        latest_rankings = latest_rankings.groupby(['Season', 'TeamID', 'SystemName'])['OrdinalRank'].mean().reset_index()
        
        # Pivot to get system rankings as columns
        ranking_pivot = latest_rankings.pivot_table(
            index=['Season', 'TeamID'], 
            columns='SystemName', 
            values='OrdinalRank'
        ).reset_index()
        
        # Fill missing rankings with median for that season
        ranking_pivot = ranking_pivot.fillna(ranking_pivot.groupby('Season').transform('median'))
        
        return ranking_pivot
    
    def _calculate_h2h_records(self):
        """Calculate historical head-to-head records between teams"""
        # Get all matchups
        matchups = []
        for season in self.regular_results['Season'].unique():
            season_games = self.regular_results[self.regular_results['Season'] == season]
            for _, game in season_games.iterrows():
                team1, team2 = sorted([game['WTeamID'], game['LTeamID']])
                winner = game['WTeamID']
                matchups.append({
                    'Season': season,
                    'Team1': team1,
                    'Team2': team2,
                    'Winner': winner
                })
        
        matchups_df = pd.DataFrame(matchups)
        
        # Calculate head-to-head record
        h2h_stats = matchups_df.groupby(['Season', 'Team1', 'Team2']).agg({
            'Winner': ['count', lambda x: sum(x == x.name[0])]  # Total games, games won by Team1
        }).reset_index()
        
        h2h_stats.columns = ['Season', 'Team1', 'Team2', 'H2H_Games', 'H2H_Wins_Team1']
        h2h_stats['H2H_WinPct_Team1'] = h2h_stats['H2H_Wins_Team1'] / h2h_stats['H2H_Games']
        
        return h2h_stats
    
    def _calculate_recent_form(self):
        """Calculate recent form (last 10 games)"""
        # Sort by date and calculate rolling statistics
        form_data = []
        
        for season in self.regular_results['Season'].unique():
            season_games = self.regular_results[self.regular_results['Season'] == season].copy()
            season_games = season_games.sort_values('DayNum')
            
            # Process each team's recent games
            for team_id in pd.concat([season_games['WTeamID'], season_games['LTeamID']]).unique():
                team_games = season_games[
                    (season_games['WTeamID'] == team_id) | (season_games['LTeamID'] == team_id)
                ].copy()
                team_games = team_games.sort_values('DayNum')
                
                if len(team_games) >= 3:  # Need at least 3 games
                    last_10 = team_games.tail(10)
                    wins = sum((last_10['WTeamID'] == team_id))
                    win_pct = wins / len(last_10)
                    points_scored = []
                    points_allowed = []
                    
                    for _, game in last_10.iterrows():
                        if game['WTeamID'] == team_id:
                            points_scored.append(game['WScore'])
                            points_allowed.append(game['LScore'])
                        else:
                            points_scored.append(game['LScore'])
                            points_allowed.append(game['WScore'])
                    
                    form_data.append({
                        'Season': season,
                        'TeamID': team_id,
                        'Recent_WinPct': win_pct,
                        'Recent_Points_Mean': np.mean(points_scored),
                        'Recent_Points_Allowed_Mean': np.mean(points_allowed),
                        'Games_Used': len(last_10)
                    })
        
        return pd.DataFrame(form_data)
    
    def prepare_training_data(self):
        """Prepare training data for model fitting"""
        print("Preparing training data...")
        
        # Combine regular season and tournament results
        all_games = pd.concat([
            self.regular_results[['Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore']],
            self.tourney_results[['Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore']]
        ])
        
        # Create training examples (both directions for each game)
        training_data = []
        
        for _, game in all_games.iterrows():
            # Original direction (winner first)
            training_data.append({
                'Season': game['Season'],
                'Team1': game['WTeamID'],
                'Team2': game['LTeamID'],
                'Team1_Score': game['WScore'],
                'Team2_Score': game['LScore'],
                'Target': 1  # Team1 won
            })
            
            # Reverse direction (loser first)
            training_data.append({
                'Season': game['Season'],
                'Team1': game['LTeamID'],
                'Team2': game['WTeamID'],
                'Team1_Score': game['LScore'],
                'Team2_Score': game['WScore'],
                'Target': 0  # Team1 lost
            })
        
        train_df = pd.DataFrame(training_data)
        
        # Merge features
        train_df = self._merge_features(train_df)
        
        print(f"Training data prepared with {len(train_df)} samples")
        return train_df
    
    def _merge_features(self, df):
        """Merge all features into training dataframe"""
        # Merge team stats for both teams
        df = df.merge(self.features['team_stats'], 
                     left_on=['Season', 'Team1'], 
                     right_on=['Season', 'TeamID'], 
                     how='left', suffixes=('', '_Team1'))
        df = df.merge(self.features['team_stats'], 
                     left_on=['Season', 'Team2'], 
                     right_on=['Season', 'TeamID'], 
                     how='left', suffixes=('', '_Team2'))
        
        # Merge seeds
        df = df.merge(self.features['seeds'], 
                     left_on=['Season', 'Team1'], 
                     right_on=['Season', 'TeamID'], 
                     how='left')
        df = df.rename(columns={'SeedNum': 'Seed_Team1'})
        df = df.merge(self.features['seeds'], 
                     left_on=['Season', 'Team2'], 
                     right_on=['Season', 'TeamID'], 
                     how='left')
        df = df.rename(columns={'SeedNum': 'Seed_Team2'})
        
        # Merge rankings (using average of available systems)
        ranking_cols = [col for col in self.features['rankings'].columns 
                       if col not in ['Season', 'TeamID']]
        if ranking_cols:
            avg_ranking = self.features['rankings'][['Season', 'TeamID'] + ranking_cols].copy()
            avg_ranking['AvgRanking'] = avg_ranking[ranking_cols].mean(axis=1)
            
            df = df.merge(avg_ranking[['Season', 'TeamID', 'AvgRanking']], 
                         left_on=['Season', 'Team1'], 
                         right_on=['Season', 'TeamID'], 
                         how='left')
            df = df.rename(columns={'AvgRanking': 'Ranking_Team1'})
            df = df.merge(avg_ranking[['Season', 'TeamID', 'AvgRanking']], 
                         left_on=['Season', 'Team2'], 
                         right_on=['Season', 'TeamID'], 
                         how='left')
            df = df.rename(columns={'AvgRanking': 'Ranking_Team2'})
        
        # Merge head-to-head records
        df = df.merge(self.features['h2h'], 
                     left_on=['Season', 'Team1', 'Team2'], 
                     right_on=['Season', 'Team1', 'Team2'], 
                     how='left')
        # Add reverse H2H
        df_reverse = df.merge(self.features['h2h'], 
                            left_on=['Season', 'Team2', 'Team1'], 
                            right_on=['Season', 'Team1', 'Team2'], 
                            how='left', suffixes=('', '_reverse'))
        df_reverse['H2H_WinPct_Team1'] = df_reverse['H2H_WinPct_Team1'].fillna(
            1 - df_reverse['H2H_WinPct_Team1_reverse'])
        df_reverse['H2H_Games'] = df_reverse['H2H_Games'].fillna(df_reverse['H2H_Games_reverse'])
        
        # Merge recent form
        df_reverse = df_reverse.merge(self.features['form'], 
                                    left_on=['Season', 'Team1'], 
                                    right_on=['Season', 'TeamID'], 
                                    how='left')
        df_reverse = df_reverse.merge(self.features['form'], 
                                    left_on=['Season', 'Team2'], 
                                    right_on=['Season', 'TeamID'], 
                                    how='left', suffixes=('', '_Team2'))
        
        # Select final features
        feature_cols = [
            'WinPct', 'Points_Mean', 'OppPoints_Mean', 'PointDifferential', 'Points_Std',
            'WinPct_Team2', 'Points_Mean_Team2', 'OppPoints_Mean_Team2', 
            'PointDifferential_Team2', 'Points_Std_Team2',
            'Seed_Team1', 'Seed_Team2', 'Ranking_Team1', 'Ranking_Team2',
            'H2H_WinPct_Team1', 'H2H_Games', 'Recent_WinPct', 'Recent_Points_Mean',
            'Recent_Points_Allowed_Mean', 'Recent_WinPct_Team2', 
            'Recent_Points_Mean_Team2', 'Recent_Points_Allowed_Mean_Team2'
        ]
        
        # Fill missing values
        df_final = df_reverse[feature_cols + ['Target']].copy()
        df_final = df_final.fillna({
            'Seed_Team1': 16, 'Seed_Team2': 16,  # Default high seed
            'Ranking_Team1': 100, 'Ranking_Team2': 100,  # Default low ranking
            'H2H_WinPct_Team1': 0.5,  # No history assumption
            'H2H_Games': 0
        })
        df_final = df_final.fillna(df_final.median())  # Fill remaining with median
        
        self.feature_names = feature_cols
        return df_final
    
    def train_model(self, train_data):
        """Train ensemble model"""
        print("Training model...")
        
        X = train_data[self.feature_names]
        y = train_data['Target']
        
        # Use ensemble of models
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X, y)
            self.models[name] = model
        
        print("Model training completed!")
    
    def predict(self, test_data):
        """Make predictions using ensemble"""
        X_test = test_data[self.feature_names]
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict_proba(X_test)[:, 1]  # Probability of class 1
            predictions[name] = pred
        
        # Ensemble prediction (average)
        final_pred = np.mean(list(predictions.values()), axis=0)
        return final_pred
    
    def create_submission(self, submission_file="submission.csv"):
        """Create submission file for Stage 1"""
        print("Creating submission...")
        
        # Parse sample submission IDs
        self.sample_submission[['Season', 'Team1', 'Team2']] = (
            self.sample_submission['ID'].str.split('_', expand=True)
        )
        self.sample_submission['Season'] = self.sample_submission['Season'].astype(int)
        self.sample_submission['Team1'] = self.sample_submission['Team1'].astype(int)
        self.sample_submission['Team2'] = self.sample_submission['Team2'].astype(int)
        
        # Prepare test data
        test_data = self.sample_submission[self.feature_names + ['Season', 'Team1', 'Team2']].copy()
        
        # Merge features (same as training)
        test_data = self._merge_test_features(test_data)
        
        # Make predictions
        predictions = self.predict(test_data)
        
        # Create submission
        submission = self.sample_submission.copy()
        submission['Pred'] = predictions
        
        # Save submission
        submission[['ID', 'Pred']].to_csv(submission_file, index=False)
        print(f"Submission saved to {submission_file}")
        
        return submission
    
    def _merge_test_features(self, df):
        """Merge features for test data (same logic as training)"""
        # This would use the same merge logic as _merge_features
        # For now, we'll implement a simplified version
        df = df.fillna({
            'Seed_Team1': 16, 'Seed_Team2': 16,
            'Ranking_Team1': 100, 'Ranking_Team2': 100,
            'H2H_WinPct_Team1': 0.5,
            'H2H_Games': 0
        })
        df = df.fillna(df.median())
        return df

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = MarchMadnessPredictor()
    
    # Load data
    predictor.load_data()
    
    # Create features
    features = predictor.create_features()
    
    # Prepare training data
    train_data = predictor.prepare_training_data()
    
    # Train model
    predictor.train_model(train_data)
    
    # Create submission
    submission = predictor.create_submission()
    
    print("Prediction pipeline completed!")