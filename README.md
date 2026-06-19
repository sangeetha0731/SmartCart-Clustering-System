# SmartCart Clustering System

## Project Overview

SmartCart is a growing e-commerce platform that has collected extensive customer data to improve marketing efficiency and customer retention. This project develops an intelligent customer segmentation system using unsupervised machine learning clustering algorithms to discover hidden patterns in customer behavior and group customers into meaningful segments.

## Problem Statement

SmartCart currently uses generic marketing and engagement strategies for all customers without clearly understanding different customer behavior patterns. This approach results in:

- **Inefficient marketing**: Resources are not optimally allocated across customer segments
- **Missed opportunities**: Inability to effectively retain high-value customers
- **Delayed churn identification**: Slow recognition of churn-prone users

This project aims to build an intelligent customer segmentation system using unsupervised machine learning to analyze historical customer transaction data and group customers into meaningful clusters based on purchasing behavior, engagement levels, and loyalty indicators.

## Dataset Description

**File**: `smartcart_customers.csv`

**Dimensions**: 2,240 customer records with 22 attributes

**Missing Values**: 24 missing values detected in Income column

### Data Categories

#### 1. Customer Demographics
- **ID**: Unique customer identifier
- **Year_Birth**: Year of birth of the customer
- **Education**: Highest education level achieved
- **Marital_Status**: Marital status of the customer
- **Income**: Yearly household income
- **Kidhome**: Number of small children in household
- **Teenhome**: Number of teenagers in household
- **Dt_Customer**: Date when customer enrolled with SmartCart

#### 2. Purchase Behavior (Amount Spent)
- **MntWines**: Amount spent on wine products
- **MntFruits**: Amount spent on fruits
- **MntMeatProducts**: Amount spent on meat products
- **MntFishProducts**: Amount spent on fish products
- **MntSweetProducts**: Amount spent on sweet products
- **MntGoldProds**: Amount spent on gold products

#### 3. Purchase Behavior (Frequency)
- **NumDealsPurchases**: Purchases made using discounts
- **NumWebPurchases**: Purchases made through website
- **NumCatalogPurchases**: Purchases made through catalog
- **NumStorePurchases**: Purchases made in physical stores
- **NumWebVisitsMonth**: Number of visits to website per month

#### 4. Customer Feedback & Constants
- **Recency**: Number of days since last purchase
- **Complain**: Customer complained in last 2 years (1 = Yes, 0 = No)
- **Response**: Response to marketing campaign (1 = Yes, 0 = No)

## Methodology/Approach

The analysis follows a structured machine learning pipeline:

### Step 1: Data Loading and Exploration
- Load customer data from CSV file (2,240 × 22)
- Examine data structure and identify missing values
- Discovered 24 missing values in Income column

### Step 2: Data Preprocessing
- **Missing Value Handling**: Fill missing Income values with median
- **Feature Engineering**: Create derived features:
  - **Age**: Calculated from Year_Birth
  - **Customer_Tenure_Days**: Calculated from Dt_Customer enrollment date
  - **Total_Spending**: Aggregate of all product category spending (Mnt* columns)
  - **Total_Children**: Sum of Kidhome and Teenhome
- **Feature Selection**: Select 15 most relevant features
- **Outlier Treatment**: Remove outliers from dataset
- **Processed shape**: 2,236 × 15

### Step 3: Categorical Encoding
- **One-Hot Encoding**: Encode categorical features (Education, Living_With) using OneHotEncoder from scikit-learn
- **Result**: 2,236 × 18 (after encoding adds new columns)

### Step 4: Feature Scaling
- **Standardization**: Apply StandardScaler to normalize all features to mean=0, std=1
- **Purpose**: Ensure all features contribute equally to clustering distance calculations

### Step 5: Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Reduce to 3 components for visualization
- **Method**: Fit PCA on scaled data for 2D and 3D visualization purposes

### Step 6: Optimal Cluster Determination
- **Elbow Method**: Apply K-means clustering with k=1 to 10
  - Calculate WCSS (Within-Cluster Sum of Squares) for each k value
  - Use KneeLocator to identify the elbow point
  - **Optimal k identified**: 4 clusters
- **Silhouette Analysis**: Compute silhouette scores for k=2 to 10 to validate cluster quality

### Step 7: Clustering Algorithm
- **Primary Algorithm**: Agglomerative Clustering (hierarchical clustering)
  - Parameters: n_clusters=4, linkage="ward"
  - Generates cluster assignments for all 2,236 customers

### Step 8: Cluster Visualization and Characterization
- **3D Visualization**: Plot clusters using 3 PCA components
- **Cluster Distribution**: Generate count plot showing cluster sizes
- **Feature Analysis**: Create scatter plots of Income vs. Total_Spending colored by cluster
- **Summary Statistics**: Calculate mean values for all features grouped by cluster

## Technologies Used

**Programming Language**: Python

**Core Libraries**:
- **pandas**: Data manipulation, loading from CSV, feature engineering
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical visualization, styling
- **scikit-learn**: Machine learning algorithms including:
  - OneHotEncoder (categorical encoding)
  - StandardScaler (feature normalization)
  - PCA (dimensionality reduction)
  - KMeans (for elbow method)
  - silhouette_score (clustering evaluation metric)
  - AgglomerativeClustering (hierarchical clustering)
- **kneed**: KneeLocator for automatic elbow point detection

**Environment**: Jupyter Notebook

## How to Run the Notebook

### Prerequisites
- Python 3.x environment
- All libraries listed in requirements.txt installed
- `smartcart_customers.csv` file located in the same directory as the notebook

### Installation Steps

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the data file is present:
```
smartcart_customers.csv (in same directory as .ipynb)
```

### Execution

Launch Jupyter Notebook:
```bash
jupyter notebook smartcart.ipynb
```

Execute all cells sequentially from top to bottom:
1. Import libraries (pandas, matplotlib, seaborn, scikit-learn components, kneed)
2. Load CSV data
3. Explore data structure and missing values
4. Apply preprocessing and feature engineering
5. Handle missing values
6. Scale and encode features
7. Apply PCA
8. Run elbow method analysis
9. Calculate silhouette scores
10. Apply Agglomerative Clustering
11. Visualize and analyze clusters
12. Generate cluster summary statistics

## Results/Insights

### Key Findings

**Optimal Number of Clusters**: 4 clusters identified using Agglomerative Clustering with Ward linkage

**Cluster Sizes**: Unequal distribution across 4 clusters (visible in cluster count plot)

**Income and Spending Patterns**:
- **Cluster 1**: Mean income ~$72,808, higher web purchases (5.69 average)
- **Cluster 0**: Mean income ~$39,681, lower web purchases (3.15 average)
- Strong positive correlation between Income and Total_Spending across clusters
- Clear visual separation of clusters in Income vs. Spending scatter plot

**Purchase Channel Preferences**:
- Clusters show distinct preferences for web purchases, catalog purchases, and store purchases
- Differences in discount usage (NumDealsPurchases) across clusters
- Varying frequency of website visits per month

**Feature Correlations**: Heatmap visualization shows relationships between:
- Income, Recency, Purchase amounts
- Purchase frequency across channels
- Response rates and complaint frequencies

### Cluster Summary Statistics

The analysis generated cluster means for all features including:
- **Income**: Range from ~$36,960 to ~$72,808 across clusters
- **NumWebPurchases**: Range from 2.71 to 5.79 across clusters
- **NumDealsPurchases**: Range from 1.86 to 2.59 across clusters
- **Response Rate**: Varies by cluster (7.6% to 32% response rates)
- **Customer Demographics**: Different age and tenure profiles by cluster
- **Categorical Features**: Education and living situation distribution by cluster

### Visualizations Generated
- 3D scatter plot of clusters (using 3 PCA components)
- Cluster count distribution bar chart
- Income vs. Total_Spending scatter plot colored by cluster assignment
- Correlation heatmap of all features

## Conclusion

The SmartCart Clustering System successfully segments 2,236 customers into 4 distinct and interpretable clusters based on their purchasing behavior, demographic characteristics, and engagement patterns. The use of Agglomerative Clustering with Ward linkage reveals meaningful customer groups with significantly different:

- Income levels and spending power
- Purchase channel preferences
- Frequency of purchases and website visits
- Campaign response rates

These clusters provide a foundation for developing targeted marketing strategies tailored to each segment's characteristics. The clear differentiation in income and spending patterns across clusters validates that this segmentation effectively captures important variations in customer behavior that can drive personalized business decisions.

The identified segments can be leveraged for:
- Personalized marketing campaigns optimized for each cluster
- Targeted customer retention initiatives
- Resource allocation based on customer value
- Development of cluster-specific product recommendations and promotions
