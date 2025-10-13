from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# ---------------------------------
# 1. Initialize Spark Session
# ---------------------------------
spark = SparkSession.builder \
    .appName("CarRepairCostPredictor") \
    .getOrCreate()

# ---------------------------------
# 2. Create dummy training data
# ---------------------------------
data = [
    ("engine knocking sound", 5, 120000, 2500.0),
    ("brake pads worn out", 3, 80000, 800.0),
    ("transmission failure", 8, 200000, 4000.0),
    ("oil leak detected", 6, 150000, 1500.0),
    ("flat tire replacement", 2, 30000, 200.0),
    ("battery dead won't start", 4, 60000, 600.0),
    ("check engine light on", 5, 100000, 1200.0),
    ("radiator overheating issue", 7, 180000, 2200.0),
]

columns = ["description", "age_years", "mileage_km", "repair_cost"]

df = spark.createDataFrame(data, columns)

# ---------------------------------
# 3. Text preprocessing
# ---------------------------------
tokenizer = Tokenizer(inputCol="description", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
idf = IDF(inputCol="rawFeatures", outputCol="textFeatures")

# ---------------------------------
# 4. Combine text + numeric features
# ---------------------------------
assembler = VectorAssembler(
    inputCols=["textFeatures", "age_years", "mileage_km"],
    outputCol="features"
)

# ---------------------------------
# 5. Train regression model
# ---------------------------------
lr = LinearRegression(featuresCol="features", labelCol="repair_cost")

# ---------------------------------
# 6. Build pipeline
# ---------------------------------
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, assembler, lr])

# ---------------------------------
# 7. Train the model
# ---------------------------------
model = pipeline.fit(df)

# ---------------------------------
# 8. Test prediction
# ---------------------------------
test_data = spark.createDataFrame([
    ("suspension noise over bumps", 6, 130000, 0.0),
    ("engine oil low", 3, 90000, 0.0)
], ["description", "age_years", "mileage_km", "repair_cost"])

predictions = model.transform(test_data)
predictions.select("description", "prediction").show(truncate=False)

# ---------------------------------
# 9. Stop Spark session
# ---------------------------------
spark.stop()
