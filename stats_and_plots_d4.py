import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_excel('matched_rows_d4.xlsx')
print(len(df))
df['Det_Dev'] = df['Det_Dev'].str.strip('%').astype(float)
df['DTA'] = df['DTA'].str.strip('%').astype(float)
df['γ-index'] = df['γ-index'].str.strip('%').astype(float)

# Only select where pass rate < 95
# df = df[df['Pass Rate']<95]

y_metrics = ['Avg_MCS', 'Avg_EM', 'Avg_LeafTravel',	'Avg_ArcLength']

df[y_metrics + ['γ-index']] = df[y_metrics + ['γ-index']].astype(float)

os.makedirs("plots", exist_ok=True)
for metric in y_metrics:
    sns.jointplot(data=df, x='γ-index', y=metric, kind='reg')
    plt.savefig(f"plots/d4_{metric}_vs_gamma.png", dpi=300)
    plt.close()

# sns.jointplot(data=df, x='γ-index', y='PyComplexityMetric', kind='reg')

sns.pairplot(df,vars=y_metrics)
plt.savefig(f"plots/d4_pairplot.png", dpi=300)
# plt.show()
