import os
from datetime import datetime
from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql.types import DoubleType
from data_preprocessor.config import source_schema, nulls_columns, points_columns


class DataPreprocessor:
    """ Class for data preprocessing """

    def __init__(self, input_directory, output_directory):
        self.component_name = "data_preprocessor"
        self.root_directory = Path(__file__).parent.parent
        self.input_directory = input_directory
        self.output_directory = output_directory
        self._start_time = datetime.now()
        self._complete_time = None
        self._spark = SparkSession.builder.appName(self.component_name).getOrCreate()

    def get_source_schema(self):
        # Define your schema here based on your dataset
        pass

    def read_data(self, path):
        filepath = os.path.join(self.root_directory, self.input_directory, path)
        return self._spark.read.csv(filepath, header=True, schema=source_schema)

    @staticmethod
    def calculate_the_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance."""
        return (
                F.lit(6371) * F.acos(
            F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) *
            F.cos(F.radians(lon2) - F.radians(lon1)) +
            F.sin(F.radians(lat1)) * F.sin(F.radians(lat2))
        )
        )

    def filter_df(self, df, columns):
        return df.select([col for col in df.columns if col not in columns])

    def transform_data(self, df):
        # Step 1: Filter columns with nulls
        df = self.filter_df(df, nulls_columns)

        # Step 2: Extract latitude and longitude from point columns
        for col_name in points_columns:
            df = (
                df.withColumn(f"{col_name}_longitude",
                              F.regexp_extract(col_name, r"POINT \(([-\d.]+)", 1).cast("double"))
                .withColumn(f"{col_name}_latitude",
                            F.regexp_extract(col_name, r"POINT \([-.\d]+ ([\d.]+)\)", 1).cast("double"))
            )

        df = self.filter_df(df, points_columns)

        # Step 3: Calculate direct distance
        df = df.withColumn(
            "direct_distance",
            self.calculate_the_distance(
                F.col("pickup_centroid_latitude"), F.col("pickup_centroid_longitude"),
                F.col("dropoff_centroid_latitude"), F.col("dropoff_centroid_longitude")
            )
        )

        # Step 4: Handle Categorical Variables with Indexing and Encoding
        indexer_payment = StringIndexer(inputCol="payment_type", outputCol="payment_type_index")
        encoder_payment = OneHotEncoder(inputCols=["payment_type_index"], outputCols=["payment_type_encoded"])
        indexer_company = StringIndexer(inputCol="company", outputCol="company_index")
        encoder_company = OneHotEncoder(inputCols=["company_index"], outputCols=["company_encoded"])

        # Apply indexers and encoders
        df = indexer_payment.fit(df).transform(df)
        df = encoder_payment.fit(df).transform(df)
        df = indexer_company.fit(df).transform(df)
        df = encoder_company.fit(df).transform(df)

        # Filter out the original categorical columns
        df = self.filter_df(df, ["payment_type", "company"])

        # Step 5: Handle Missing Values
        df = df.fillna({
            "fare": 0.0, "tips": 0.0, "tolls": 0.0, "extras": 0.0,
            "trip_seconds": 0.0, "trip_miles": 0.0,
            "pickup_community_area": -1, "dropoff_community_area": -1, "direct_distance": 0.0
        })

        # Step 6: Scale Numeric Columns
        numeric_cols = ["trip_seconds", "trip_miles", "fare", "tips", "tolls", "extras", "direct_distance"]
        df = df.select([F.col(c).cast(DoubleType()).alias(c) if c in numeric_cols else F.col(c) for c in df.columns])

        # Assemble numeric features for scaling
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
        df = assembler.transform(df)

        # Scale the assembled numeric features
        scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)

        # Step 7: Combine All Features for Model Training
        feature_cols = ["scaled_numeric_features", "payment_type_encoded", "company_encoded"]
        assembler_final = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_final = assembler_final.transform(df)

        # Select only the final features and target column
        df_final = df_final.select("features", "trip_total").dropna()

        print(f"Final dataframe count: {df_final.count()}")
        return df_final

    def load_data(self, df, output_path):
        df.write.mode("overwrite").parquet(output_path)

    def main(self, input_path):
        """Main function to execute the data preprocessing workflow."""
        try:
            # Read the data
            df = self.read_data(input_path)

            # Transform the data
            df_final = self.transform_data(df)

            # Define output path
            output_path = os.path.join(self.root_directory, self.output_directory, "processed_data.parquet")

            # Load the data
            self.load_data(df_final, output_path)

            # Log completion time
            self._complete_time = datetime.now()
            print(f"Data preprocessing completed in: {self._complete_time - self._start_time}")

        except Exception as e:
            print(f"An error occurred during processing: {e}")



