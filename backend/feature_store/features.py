from datetime import timedelta
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define an entity for the transaction user
user = Entity(name="user_id", value_type=ValueType.STRING, description="The ID of the user")

# Define the source of the data (using the synthetic CSV for now as a placeholder for offline source)
# In production, this would likely be a Parquet file or a SQL query
import os

# Use relative path or environment variable
data_path = os.getenv(
    "FEATURE_STORE_DATA_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "synthetic_financial_qa.csv")
)

transaction_source = FileSource(
    path=data_path,
    timestamp_field="timestamp",
    created_timestamp_column="timestamp",
)

# Define the feature view
transaction_features = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="merchant", dtype=String),
        Field(name="is_fraud_risk", dtype=Int64),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "fraud_detection"},
)
