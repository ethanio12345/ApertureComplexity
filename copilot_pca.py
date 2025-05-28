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
pca = PCA(n_components=2)  # Adjust the number of components as needed
principal_components = pca.fit_transform(standardized_features)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Pass Rate'] = target.values

# Display the DataFrame
print(pca_df.head())


import matplotlib.pyplot as plt

plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Pass Rate'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Pass Rate')


loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)
print(loadings)


plt.figure()
loadings.plot(kind='bar')
plt.title('Loadings for PC1 and PC2')
plt.xlabel('Variables')
plt.ylabel('Loadings')



# Perform K-Means clustering based on pass rates
kmeans = KMeans(n_clusters=10) # Adjust the number of clusters as needed
pca_df['Cluster'] = kmeans.fit_predict(pca_df[['Pass Rate']])

# Visualize the clusters based on pass rates
plt.figure()
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.title('K-Means Clustering of Radiotherapy Plans Based on Pass Rates')







plt.show()
