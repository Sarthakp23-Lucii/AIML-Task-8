## K-Means Clustering on Mall Customers Dataset ##
-This project applies K-Means Clustering to segment customers based on their Annual Income and Spending Score from the Mall Customers dataset.

📌 Steps Performed
1. Load & Prepare Data
-Loaded Mall_Customers.csv.
-Selected numerical features:
-Annual Income (k$)
-Spending Score (1–100)
-Standardized features using StandardScaler.

2. Apply PCA (2D Visualization)
-Reduced data to 2 principal components for easy visualization.

3. K-Means Clustering
-Trained K-Means with k = 5.
-Assigned cluster labels to each customer.

4. Elbow Method
-Tested k values from 1 to 10.
-Plotted inertia to help choose the optimal number of clusters.

5. Silhouette Score
-Evaluated cluster quality using Silhouette Score.

📈 Visualizations Created
-Elbow Method Plot
-PCA-based Cluster Scatter Plot

🛠️ Libraries Used
-pandas
-matplotlib
-scikit-learn
-KMeans
-PCA

StandardScaler

silhouette_score
