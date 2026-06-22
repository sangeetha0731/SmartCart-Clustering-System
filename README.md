# SmartCart Clustering System - Brief Documentation

## Problem Statement

SmartCart uses generic marketing strategies for all customers without understanding different behavior patterns, resulting in:
- Inefficient marketing resource allocation
- Missed retention opportunities for high-value customers
- Slow churn identification

This project builds a customer segmentation system using unsupervised machine learning to group customers into meaningful clusters based on purchasing behavior and engagement.

---

## Features

### Customer Demographics
- ID, Year_Birth, Education, Marital_Status, Income
- Kidhome, Teenhome, Dt_Customer (enrollment date)

### Purchase Behavior - Amount Spent
- MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds

### Purchase Behavior - Frequency
- NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth

### Customer Engagement
- Recency (days since last purchase)
- Complain (1=Yes, 0=No)
- Response (to marketing campaign: 1=Yes, 0=No)

**Dataset**: 2,240 customer records × 22 attributes | 24 missing values in Income column

---

## Model Information

**Algorithm**: Agglomerative Clustering (hierarchical clustering)
- **Parameters**: n_clusters=4, linkage="ward"
- **Optimal Clusters**: 4 (determined via Elbow Method)

**Key Steps**:
1. Handle missing values (median imputation)
2. Feature engineering (Age, Customer_Tenure_Days, Total_Spending, Total_Children)
3. One-hot encoding for categorical features
4. StandardScaler normalization
5. PCA dimensionality reduction (3 components)
6. Silhouette analysis for validation

**Libraries**: pandas, scikit-learn, matplotlib, seaborn, kneed

---

## Project Structure

```
smartcart-clustering/
├── smartcart.ipynb              # Main notebook
├── smartcart_customers.csv      # Dataset
├── requirements.txt             # Dependencies
└── README.md
```

---

## Installation & Usage

### Prerequisites
- Python 3.x
- `smartcart_customers.csv` in same directory as notebook

### Install Dependencies
```bash
pip install -r requirements.txt
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

---

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

