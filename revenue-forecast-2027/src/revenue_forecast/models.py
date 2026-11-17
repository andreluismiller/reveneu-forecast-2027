from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor

def candidates(seed=42):
    return {
      "elastic_net":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",ElasticNet(alpha=.01,l1_ratio=.25,max_iter=10000,random_state=seed))]),
      "hist_gradient_boosting":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(learning_rate=.04,max_iter=400,max_leaf_nodes=15,l2_regularization=1,random_state=seed))]),
      "random_forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=seed))]),
    }
