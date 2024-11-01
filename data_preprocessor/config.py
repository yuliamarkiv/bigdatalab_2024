from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

source_schema = StructType([
    StructField('trip_id', StringType(), True),
    StructField('taxi_id', StringType(), True),
    StructField('trip_start_timestamp', TimestampType(), True),
    StructField('trip_end_timestamp', TimestampType(), True),
    StructField('trip_seconds', IntegerType(), True),
    StructField('trip_miles', FloatType(), True),
    StructField('pickup_census_tract', StringType(), True),
    StructField('dropoff_census_tract', StringType(), True),
    StructField('pickup_community_area', IntegerType(), True),
    StructField('dropoff_community_area', IntegerType(), True),
    StructField('fare', FloatType(), True),
    StructField('tips', FloatType(), True),
    StructField('tolls', FloatType(), True),
    StructField('extras', FloatType(), True),
    StructField('trip_total', FloatType(), True),
    StructField('payment_type', StringType(), True),
    StructField('company', StringType(), True),
    StructField('pickup_centroid_latitude', FloatType(), True),
    StructField('pickup_centroid_longitude', FloatType(), True),
    StructField('pickup_centroid_location', StringType(), True),
    StructField('dropoff_centroid_latitude', FloatType(), True),
    StructField('dropoff_centroid_longitude', FloatType(), True),
    StructField('dropoff_centroid_location', StringType(), True)
])


nulls_columns = ["pickup_census_tract", "dropoff_census_tract"]
points_columns = ["pickup_centroid_location", "dropoff_centroid_location"]