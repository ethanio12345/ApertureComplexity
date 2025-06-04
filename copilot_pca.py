import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Load your data
data = pd.read_csv('matched_rows_arccheck_deduped_keep_last.csv')

# Select the relevant columns
relevant_columns = ['PyComplexityMetric', 'MeanAreaMetricEstimator', 'AreaMetricEstimator', 
                    'ApertureIrregularityMetric', 'ModulationComplexityScore', 'Pass Rate']
relevant_data = data[relevant_columns]

# Separate features and target
features = relevant_data.drop('Pass Rate', axis=1)
target = relevant_data['Pass Rate']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features_imputed)

# Perform PCA
pca = PCA(n_components=3)  # Adjust the number of components as needed
principal_components = pca.fit_transform(standardized_features)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['Pass Rate'] = target.values

# Display the DataFrame
print(pca_df.head())

# Plot in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['Pass Rate'], cmap='viridis')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')


loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=features.columns)
print(loadings)


loadings.plot(kind='bar')
plt.title('Loadings for PC1, PC2, and PC3')
plt.xlabel('Variables')
plt.ylabel('Loadings')


plt.show()
