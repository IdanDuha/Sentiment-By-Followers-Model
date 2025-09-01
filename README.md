# Sentiment-By-Followers-Model
Small DS project – includes over 150K tweets from Kaggle, used to create a model for forecasting the number of likes based on user popularity and peaceful or provocative sentiment. Created in collaboration with classmates Matan Iser, Omer Topchi and Yuval Ellins 

While Preproccesing we removed all unneccery features and worthelss data (as posts with 0 retweets,0 likes and 0 followers), outliers, and balanced the oversampled data
we tokenized the tweets and used Pytorch and Transformers to extract the sentiment from a tweets content using toxic-bert model.
We built multiple models - RT, DT, XGBOOST and DNN. and ensembled them with a linear regression model for the output of prediciting the amount of likes and retweets.
My direct responsibilty wass the RT and DT

### Decision Tree

### Data preparation
- Load CSV: toxicity_output_combined.csv
- Drop rows with missing followers_count or number_of_likes
- Create follower-based groups to stabilize distributions:
  - Groups 0–6: predefined follower ranges
  - Group 7: split into 7a, 7b, 7c using either percentile cutpoints (33%, 67%) or business breakpoints (25M, 60M)
  - Optionally split 7c into 7c1 and 7c2 at the median if the subgroup is large
- Assign each row to a group label (Group_0 … Group_7a/7b/7c[/7c1/7c2])

### Outlier handling and target transforms
For each group we try multiple strategies and keep the best:
- original: no change
- remove_extreme: drop values above 99th percentile of number_of_likes
- capped_95th: clip number_of_likes at the 95th percentile
- log_transform: model log1p(number_of_likes)
- sqrt_transform: model sqrt(number_of_likes)

Each strategy produces a processed dataframe and a target column name used for training.

### Feature engineering
Minimal feature set used for the tree:
- followers_count
- sentiment

Additional features are created (and cleaned) to support experimentation and mega-account effects:
- log_followers, followers_squared, followers_sqrt
- followers_zscore (groupwise)
- For mega accounts (groups 7a/7b/7c):
  - log_log_followers
  - follower_tier (binned)
  - is_ultra_mega (> 50M)
  - follower_percentile (rank pct)
  - sentiment_squared, high_sentiment (binary)
All infinities and NaNs are replaced safely.

### Training procedure
- Split per group into train/test with a dynamic test size (10%–30% depending on group size)
- DecisionTreeRegressor hyperparameters scale with data size:
  - max_depth: between 5 and 20, after multiple trials anad error we got the best results using a max depth of 15
  - min_samples_split: proportional to group size (min 5)
  - min_samples_leaf: proportional to group size (min 2)
  - max_features: "sqrt" when multiple features exist
- Fit per strategy, then evaluate on the test set

### Metrics and statistics
For each group and strategy we compute:
- MAE, MSE, RMSE
- R2 score
- Pearson correlation (predicted vs actual) and p-value
The best strategy per group is chosen by highest R2.

### Visualization and reporting
The notebook produces:
- Bar charts of R2, MAE, RMSE per group
- P-values and correlation per group
- Sample size vs performance
- Strategy effectiveness summary (average R2 by strategy)
- Subgroup analysis for Group 7 variants (7a, 7b, 7c, 7c1, 7c2)
- A printed summary table of all metrics

### Key takeaways
- Splitting high-follower users into finer subgroups improves balance and model stability
- Log-based target transforms commonly yield the strongest performance on heavy-tailed likes
- Mega accounts follow different like–follower dynamics; targeted features and subgrouping help
- The DT provides an interpretable baseline and feeds into the overall ensemble



### Random Forrest:
Used the same Preproccing for the Decision tree, multiple trial and error I got the best results with 100 trees (n_estimators)
Unfortunatly. the Random Forrest traning and using was taking too long under Google Collab, so we had to forefit using it



### Overall
All models used SHAP for analysis and testing
The DNN gave poor results so we chose not to add it to the final ensemble

### Conclusion:
It's inconclusive to find any direct corrolation between purely sentiment and populatiry of the user alone
Got a grade of 96 :) 







ChatGPT can make mistakes. Check important i

