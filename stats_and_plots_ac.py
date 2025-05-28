import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
df = pd.read_csv('matched_rows_arccheck_deduped_keep_last.csv')
# Only select where pass rate < 95
# df = df[df['Pass Rate']<95]

y_metrics = ['PyComplexityMetric', 'MeanAreaMetricEstimator', 'AreaMetricEstimator', 'ApertureIrregularityMetric', 'ModulationComplexityScore']
df[y_metrics + ['Pass Rate']] = df[y_metrics + ['Pass Rate']].astype(float)

os.makedirs("plots", exist_ok=True)
for metric in y_metrics:
    sns.jointplot(data=df, x='Pass Rate', y=metric, kind='reg', xlim=[87,101])
    plt.savefig(f"plots/ac_{metric}_vs_PassRate.png", dpi=300)
    plt.close()


# sns.jointplot(data=df, x='Pass Rate', y='PyComplexityMetric', kind='reg')

sns.pairplot(df,vars=y_metrics)
plt.savefig(f"plots/ac_pairplot.png", dpi=300)
# plt.show()