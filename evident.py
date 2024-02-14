import pandas as pd
from sklearn.datasets import fetch_california_housing
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab

# Assuming you're using Jupyter Notebook or JupyterLab
from evidently.dashboard import DashboardTabs

data = fetch_california_housing()
# Convert to DataFrame for easier manipulation
data_df = pd.DataFrame(data=data.data, columns=data.feature_names)
data_target = pd.DataFrame(data=data.target, columns=['HousePrice'])

# Split the data to simulate training and new incoming data
train_data = data_df.sample(frac=0.8, random_state=1)
new_data = data_df.drop(train_data.index)

# Reset index for convenience
train_data.reset_index(drop=True, inplace=True)
new_data.reset_index(drop=True, inplace=True)
# Initialize the dashboard with DataDriftTab to check for data drift
data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])

# Calculate the data drift
data_drift_dashboard.calculate(train_data, new_data, column_mapping=None)

# To view the dashboard inside a Jupyter Notebook
data_drift_dashboard.show()
data_drift_dashboard.save("data_drift_report.html")