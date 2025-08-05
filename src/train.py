import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config
from src.mlflow_handler import init_mlflow
from src.model_selector import train_and_select
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


cfg = load_config()



# Load dataset
df = pd.read_csv(cfg["data"]["path"])

# Encode categorical column
le = LabelEncoder()
df["ocean_proximity"] = le.fit_transform(df["ocean_proximity"])

# Impute missing values
imputer = SimpleImputer(strategy="median")
df[df.columns] = imputer.fit_transform(df)

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg["data"]["test_size"],
    random_state=cfg["data"]["random_state"]
)

# Init MLflow
init_mlflow(cfg["mlflow"]["uri"], cfg["mlflow"]["experiment_name"])

# Train
results = train_and_select(X_train, X_test, y_train, y_test, cfg["models"])
print("Training done. Results:", results)
