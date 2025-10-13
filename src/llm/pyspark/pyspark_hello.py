from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Start Spark session
spark = SparkSession.builder.appName("pyspark-hello").getOrCreate()

# Create a small DataFrame (id, feature, label)
data = [
    (1, 10.0, 15.0),
    (2, 20.0, 25.0),
    (3, 30.0, 35.0),
    (4, 40.0, 45.0)
]
columns = ["id", "feature", "label"]

df = spark.createDataFrame(data, columns)
print("Initial DataFrame:")
df.show()

# Assemble features into a vector (required by Spark ML) - specific to Spark ML
assembler = VectorAssembler(inputCols=["feature"], outputCol="features")
df_ml = assembler.transform(df)

# Train a simple linear regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(df_ml)

# Print model coefficients
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")

# Make predictions
predictions = model.transform(df_ml)
print("Predictions:")
predictions.select("id", "feature", "label", "prediction").show()

# Stop Spark session
spark.stop()
