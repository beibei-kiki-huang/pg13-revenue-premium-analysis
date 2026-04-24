import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ==========================================
# 1. Data Preparation & Engineering
# ==========================================


# Load the dataset
df = pd.read_csv('top-500-movies.csv')


# Feature Engineering: ROI (Return on Investment)
# ROI = (Revenue - Cost) / Cost
df['roi'] = (df['worldwide_gross'] - df['production_cost']) / df['production_cost']


# Feature Engineering: Seasonality
# Extract month from release date to create strategic release windows
df['release_date'] = pd.to_datetime(df['release_date'])
df['month'] = df['release_date'].dt.month


def get_season(month):
    """
    Categorize release month into strategic windows:
    - Summer: May, June, July, August (Blockbuster season)
    - Holiday: November, December (Oscar/Holiday season)
    - OffPeak: All other months (Baseline)
    """
    if month in [5, 6, 7, 8]:
        return 'Summer'
    elif month in [11, 12]:
        return 'Holiday'
    else:
        return 'OffPeak'


df['season'] = df['month'].apply(get_season)


# Data Cleaning
# Filter for relevant MPAA ratings
target_ratings = ['G', 'PG', 'PG-13', 'R']
df_clean = df[df['mpaa'].isin(target_ratings)].copy()


# Drop rows with missing values in the variables we will use
# Note: We include 'runtime' as a control variable
cols_to_check = ['worldwide_gross', 'roi', 'mpaa', 'production_cost', 'genre', 'season', 'year', 'runtime']
df_clean = df_clean.dropna(subset=cols_to_check)


print(f"Final Sample Size for Analysis: {len(df_clean)}")


# ==========================================
# 2. Descriptive Statistics
# ==========================================


# Generate Scatter Plot with Trend Lines (PG-13 vs R)
# This visual demonstrates the scaling effect of budget on revenue by rating
df_viz = df_clean[df_clean['mpaa'].isin(['PG-13', 'R'])]


plt.figure(figsize=(10, 6))
sns.lmplot(x='production_cost', y='worldwide_gross', hue='mpaa', data=df_viz,
           height=6, aspect=1.5, ci=None, palette={'PG-13': 'blue', 'R': 'orange'},
           scatter_kws={'alpha':0.5})
plt.title('Production Cost vs. Worldwide Gross: PG-13 vs. R')
plt.xlabel('Production Cost ($)')
plt.ylabel('Worldwide Gross ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('budget_vs_gross_trend.png')
print("Visualization saved as 'budget_vs_gross_trend.png'")




#Correlation Matrix


# 1. Create Dummy Variables manually for the matrix
# We do this to specifically check our key variables: PG-13 and Summer
df_clean['is_PG13'] = np.where(df_clean['mpaa'] == 'PG-13', 1, 0)
df_clean['is_Summer'] = np.where(df_clean['season'] == 'Summer', 1, 0)


# 2. Select specific columns that align with your hypothesis & controls
# These match the variables in your final regression model
cols_for_matrix = [
    'worldwide_gross',    # DV 1
    'roi',                # DV 2 (Check efficiency)
    'is_PG13',            # Main IV (The Rating Premium)
    'production_cost',    # Control 1 (Scale)
    'runtime',            # Control 2 (Scope)
    'is_Summer',          # Control 3 (Seasonality)
    'year'                # Control 4 (Inflation)
]


# 3. Rename columns for cleaner visualization in the plot
rename_map = {
    'worldwide_gross': 'Gross Revenue',
    'roi': 'ROI',
    'is_PG13': 'Rating: PG-13',
    'production_cost': 'Budget',
    'runtime': 'Runtime',
    'is_Summer': 'Season: Summer',
    'year': 'Year'
}


df_corr = df_clean[cols_for_matrix].rename(columns=rename_map)


# 4. Calculate Correlation
corr_matrix = df_corr.corr()


# 5. Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', vmin=-1, vmax=1, center=0, linewidths=0.5)
plt.title('Correlation Matrix: Final Model Variables', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix_prescriptive.png')
print("Correlation Matrix saved as 'correlation_matrix_prescriptive.png'")




# ==========================================
# 3. Model 1: Revenue Strategy Analysis
# ==========================================


print("\n" + "="*40)
print("MODEL 1: REVENUE REGRESSION RESULTS")
print("="*40)


# Formula Explanation:
# - IV: mpaa (Reference='R') -> To see the premium of PG-13 over R
# - Control: production_cost -> Controls for scale
# - Control: genre -> Controls for content type
# - Control: season (Reference='OffPeak') -> Controls for release timing
# - Control: year -> Controls for inflation
# - Control: runtime -> Controls for movie length/scope


formula_rev = """
worldwide_gross ~ C(mpaa, Treatment(reference="R"))
                  + production_cost
                  + C(genre)
                  + C(season, Treatment(reference="OffPeak"))
                  + year
                  + runtime
"""


model_rev = smf.ols(formula=formula_rev, data=df_clean).fit()
print(model_rev.summary())






# ==========================================
# 4. Prediction Case Study: Skyfall
# ==========================================


print("\n" + "="*40)
print("PREDICTION CASE STUDY: SKYFALL")
print("="*40)


# Locate the movie 'Skyfall' in the cleaned dataset
target_movie_row = df_clean[df_clean['title'].str.contains("Skyfall", case=False)]


if not target_movie_row.empty:
    # Keep as DataFrame to preserve data types
    target_data = target_movie_row.iloc[[0]]
   
    # Run prediction using Model 1
    pred = model_rev.get_prediction(target_data)
    summary_frame = pred.summary_frame(alpha=0.05) # 95% Confidence Interval
   
    # Extract values
    actual_gross = target_data['worldwide_gross'].values[0]
    predicted_gross = summary_frame['mean'].values[0]
    lower_ci = summary_frame['obs_ci_lower'].values[0]
    upper_ci = summary_frame['obs_ci_upper'].values[0]
   
    print(f"Movie Title:      {target_data['title'].values[0]}")
    print(f"Actual Gross:     ${actual_gross:,.2f}")
    print(f"Predicted Gross:  ${predicted_gross:,.2f}")
    print(f"95% Pred Range:   [${lower_ci:,.2f} — ${upper_ci:,.2f}]")
   
    # Check if actual is inside the range
    if lower_ci <= actual_gross <= upper_ci:
        print("Result:           SUCCESS (Actual value falls within the prediction interval)")
    else:
        print("Result:           OUTLIER (Actual value falls outside the prediction interval)")
else:
    print("Error: 'Skyfall' not found in the dataset.")
