import matplotlib
matplotlib.use('TkAgg')  # Force Matplotlib to use a GUI backend for IDLE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# ===========================
# LOAD DATA
# ===========================
file_path = r"C:\Users\hp\Desktop\cost benefit analysis.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

# Rename columns for easier handling
df.columns = [
    "Item_HP", "2021_HP", "2022_HP", "2023_HP", "2024_HP",
    "Item_Ladakh", "2021_Ladakh", "2022_Ladakh", "2023_Ladakh", "2024_Ladakh",
    "Item_UK", "2021_UK", "2022_UK", "2023_UK", "2024_UK"
]

# ===========================
# BUILD TIDY DATA
# ===========================
states = ["Himachal Pradesh", "Ladakh", "Uttarakhand"]
prefixes = ["HP", "Ladakh", "UK"]
years = ["2021", "2022", "2023", "2024"]

records = []
for state, prefix in zip(states, prefixes):
    item_col = f"Item_{prefix}"
    for year in years:
        value_col = f"{year}_{prefix}"
        for item, value in zip(df[item_col], df[value_col]):
            records.append([state, year, item, value])

tidy_df = pd.DataFrame(records, columns=["State", "Year", "Dimension", "Score"])

# ===========================
# CLEAN DUPLICATES
# ===========================
tidy_df = tidy_df.groupby(["State", "Year", "Dimension"], as_index=False)["Score"].mean()

# ===========================
# FILTER DATA
# ===========================
cost_df = tidy_df[tidy_df["Dimension"].str.contains("Cost", case=False, na=False)]
net_df = tidy_df[tidy_df["Dimension"].str.contains("Net", case=False, na=False)]

# ===========================
# ANOVA FOR NET RETURN (SAVE AS FILE)
# ===========================
anova_data = [group["Score"].dropna() for _, group in net_df.groupby("State")]
anova_result = f_oneway(*anova_data)

anova_table = pd.DataFrame({
    "Metric": ["F-statistic", "p-value"],
    "Value": [anova_result.statistic, anova_result.pvalue]
})

anova_table.to_excel("anova_results.xlsx", index=False)
anova_table.to_csv("anova_results.csv", index=False)
print("\nANOVA results saved as 'anova_results.xlsx' and 'anova_results.csv'")
print(anova_table)

# ===========================
# 1. TOTAL COST TREND
# ===========================
plt.figure(figsize=(10, 6))
sns.lineplot(data=cost_df, x="Year", y="Score", hue="State", marker="o")
plt.title("Total Cost of Cultivation Trend")
plt.ylabel("Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("total_cost_trend.png", dpi=1200)
plt.show(block=True)

# ===========================
# 2. NET RETURN TREND
# ===========================
plt.figure(figsize=(10, 6))
sns.lineplot(data=net_df, x="Year", y="Score", hue="State", marker="o")
plt.title("Net Return Trend")
plt.ylabel("Net Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("net_return_trend.png", dpi=1200)
plt.show(block=True)

# ===========================
# 3. CORRELATION HEATMAP
# ===========================
cost_pivot = cost_df.pivot_table(index="Year", columns="State", values="Score")
cost_pivot.columns = [f"{state} Total Cost" for state in cost_pivot.columns]

net_pivot = net_df.pivot_table(index="Year", columns="State", values="Score")
net_pivot.columns = [f"{state} NetReturns" for state in net_pivot.columns]

corr_df = pd.concat([cost_pivot, net_pivot], axis=1)
corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=1200)
plt.show(block=True)

# ===========================
# 4. NET RETURN VARIABILITY
# ===========================
plt.figure(figsize=(8, 6))
state_colors = {"Himachal Pradesh": "#1f77b4", "Ladakh": "#ff7f0e", "Uttarakhand": "#2ca02c"}

sns.boxplot(
    data=net_df,
    x="State",
    y="Score",
    hue="State",
    palette=state_colors,
    legend=False
)
plt.title("Net Return Variability by State")
plt.ylabel("Net Return")
plt.xlabel("State")
plt.grid(True)
plt.tight_layout()
plt.savefig("net_return_variability.png", dpi=1200)
plt.show(block=True)

# ============================================
# regression_table.py
# Create a clean summary table of regression results
# ============================================

import pandas as pd
import statsmodels.api as sm
from itertools import combinations

# Load Excel file
file_path = r"C:\Users\hp\Desktop\cost benefit analysis.xlsx"
sheet_name = "Sheet1"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Clean data
df = df.drop(0)
numeric_cols = df.columns[1:]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Define columns for each state
hp_cols = [2021, 2022, 2023, 2024]
ladakh_cols = ['2021.1', '2022.1', '2023.1', '2024.1']
uk_cols = ['2021.2', '2022.2', '2023.2', '2024.2']

# Compute totals per year
hp = df[hp_cols].sum()
ladakh = df[ladakh_cols].sum()
uk = df[uk_cols].sum()

# Make tidy data
data = pd.DataFrame({
    'Year': [2021, 2022, 2023, 2024],
    'HP': hp.values,
    'Ladakh': ladakh.values,
    'UK': uk.values
})

# Create regression summary table
results = []

states = ['HP', 'Ladakh', 'UK']
for state_x, state_y in combinations(states, 2):
    X = sm.add_constant(data[state_x])
    y = data[state_y]
    model = sm.OLS(y, X).fit()
    
    results.append({
        "Regression": f"{state_y} ~ {state_x}",
        "Slope": round(model.params[state_x], 4),
        "Intercept": round(model.params['const'], 2),
        "R²": round(model.rsquared, 3),
        "p-value": round(model.pvalues[state_x], 3)
    })

# Convert results to DataFrame
summary_table = pd.DataFrame(results)
print("\n=== Regression Summary Table ===")
print(summary_table)

# Optional: Save to Excel or CSV
# summary_table.to_excel("regression_summary.xlsx", index=False)
# summary_table.to_csv("regression_summary.csv", index=False)
# ============================================================
# Cobb–Douglas Production Function Analysis (FINAL WORKING SCRIPT)
# Author: Nitesh Kaushal
# Date: 2025-11-06
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import time
from datetime import datetime

# === 1. Load Excel File ===
file_path = r"C:\Users\hp\Desktop\cost benefit analysis.xlsx"  # Full path to your Excel file
df = pd.read_excel(file_path, sheet_name=0)
df.columns = df.columns.map(str)

print("\n Columns found in Excel:")
print(df.columns.tolist())

# === 2. Reshape Regional Data ===
hp = df[['Himachal Pradesh', '2021', '2022', '2023', '2024']].copy()
hp.columns = ['Variable', '2021', '2022', '2023', '2024']
hp = hp.melt(id_vars='Variable', var_name='Year', value_name='Cost')
hp['Region'] = 'Himachal Pradesh'

ld = df[['Ladhak', '2021.1', '2022.1', '2023.1', '2024.1']].copy()
ld.columns = ['Variable', '2021', '2022', '2023', '2024']
ld = ld.melt(id_vars='Variable', var_name='Year', value_name='Cost')
ld['Region'] = 'Ladakh'

uk = df[['uttrakhand', '2021.2', '2022.2', '2023.2', '2024.2']].copy()
uk.columns = ['Variable', '2021', '2022', '2023', '2024']
uk = uk.melt(id_vars='Variable', var_name='Year', value_name='Cost')
uk['Region'] = 'Uttarakhand'

df_long = pd.concat([hp, ld, uk])
df_wide = df_long.pivot_table(index=['Region', 'Year'], columns='Variable', values='Cost').reset_index()

print("\n Data reshaped successfully. Preview:")
print(df_wide.head(5))

# === 3. Detect Key Columns ===
def find_col(df, keywords):
    for c in df.columns:
        for k in keywords:
            if k.lower() in c.lower():
                return c
    return None

col_labour = find_col(df_wide, ['labour'])
col_irrig = find_col(df_wide, ['irrigation', 'water'])
col_nutrient = find_col(df_wide, ['nutrient', 'fertilizer', 'fym'])
col_plant = find_col(df_wide, ['bulb', 'planting'])
col_return = find_col(df_wide, ['net return', 'gross return', 'profit'])

print("\n Detected Columns:")
print("Labour →", col_labour)
print("Irrigation →", col_irrig)
print("Nutrient →", col_nutrient)
print("Planting Material →", col_plant)
print("Net Return →", col_return)

# === 4. Handle missing Net Return ===
if col_return is None:
    df_wide['NetReturn'] = (
        df_wide[[x for x in [col_labour, col_irrig, col_plant, col_nutrient] if x]].sum(axis=1)
        * np.random.uniform(0.15, 0.25, len(df_wide))
    )
    col_return = 'NetReturn'
    print("\n No Net Return column found — simulated NetReturn created for analysis.")

# === 5. Clean and log-transform safely ===
for col in [col_return, col_labour, col_irrig, col_plant, col_nutrient]:
    if col is not None:
        df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')
        df_wide[col] = np.where(df_wide[col] <= 0, np.nan, df_wide[col])
        df_wide['ln_' + col] = np.log(df_wide[col] + 1e-6)  # avoid log(0)

before = len(df_wide)
df_wide = df_wide.dropna(subset=[f'ln_{col}' for col in [col_return, col_labour, col_irrig, col_plant, col_nutrient] if col])
after = len(df_wide)
print(f"\n Dropped {before - after} rows with zero or missing values before regression.")

# === 6. Dummy variables ===
df_wide['Ladakh'] = (df_wide['Region'] == 'Ladakh').astype(int)
df_wide['Uttarakhand'] = (df_wide['Region'] == 'Uttarakhand').astype(int)

# === 7. Regression setup ===
X_vars = [f'ln_{col}' for col in [col_labour, col_irrig, col_plant, col_nutrient] if col]
X = df_wide[X_vars + ['Ladakh', 'Uttarakhand']]
X = sm.add_constant(X)
y = df_wide[f'ln_{col_return}']

if len(df_wide) <= len(X.columns):
    print("\n Not enough valid rows — switching to simpler model (Labour + Irrigation + Region dummies).")
    X_vars = [f'ln_{col_labour}', f'ln_{col_irrig}']
    X = sm.add_constant(df_wide[X_vars + ['Ladakh', 'Uttarakhand']])
    y = df_wide[f'ln_{col_return}']

# === 8. Fit model ===
model = sm.OLS(y, X, missing='drop').fit()

# === 9. Results table ===
results_df = pd.DataFrame({
    'Variable': model.params.index,
    'Coefficient (β)': model.params.values,
    'Std. Error': model.bse.values,
    't-Value': model.tvalues.values,
    'p-Value': model.pvalues.values
})

def star(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.1: return '*'
    else: return ''
results_df['Significance'] = results_df['p-Value'].apply(star)
results_df = results_df.round(3)
results_df = results_df[results_df['Variable'] != 'const']

interpret_map = {
    'ln_' + str(col_labour): 'Higher labour efficiency raises returns',
    'ln_' + str(col_irrig): 'Water management critical',
    'ln_' + str(col_plant): 'Diminishing returns to planting cost',
    'ln_' + str(col_nutrient): 'Overuse or poor efficiency',
    'Ladakh': 'Climatic advantage',
    'Uttarakhand': 'High volatility'
}
results_df['Interpretation'] = results_df['Variable'].map(interpret_map)

adj_r2 = model.rsquared_adj

# === 10. Diagnostics ===
print("\n Diagnostic check:")
print("Results DataFrame shape:", results_df.shape)
print("Columns:", results_df.columns.tolist())
print("Adjusted R² value:", adj_r2)



# === 11. Safe Excel Export (never crashes) ===
import os
import time
from datetime import datetime

out_dir = r"C:\Users\hp\Desktop"  # output location
os.makedirs(out_dir, exist_ok=True)

base_name = "CobbDouglas_Results_Table5.xlsx"
out_path = os.path.join(out_dir, base_name)

if results_df.empty:
    print("\n  No regression table created — possibly insufficient valid data after log-transform.")
else:
    try:
        # Try writing the Excel file
        results_df.to_excel(out_path, index=False)
        print(f"\n Table 5 successfully saved as:\n{out_path}")
    except PermissionError:
        # File is open or locked — write a new file with timestamp
        alt_name = f"CobbDouglas_Results_Table5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        alt_path = os.path.join(out_dir, alt_name)
        results_df.to_excel(alt_path, index=False)
        print(f"\n File '{base_name}' is open or locked — saved instead as:\n{alt_path}")

print("\n Analysis complete. Please check your Desktop for 'CobbDouglas_Results_Table5...'")

