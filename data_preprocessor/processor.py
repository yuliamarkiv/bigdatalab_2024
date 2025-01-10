import os
from datetime import datetime
from pathlib import Path
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql.types import TimestampType, DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from source.config import source_schema, nulls_columns, points_columns, input_directory, output_directory


class DataPreprocessor:
    """ Class for data preprocessing """

    def __init__(self):
        self.component_name = "data_preprocessor"
        self.root_directory = Path(__file__).parent.parent
        self.input_directory = input_directory
        self.output_directory = output_directory
        self._start_time = datetime.now()
        self._complete_time = None
        self._spark = SparkSession.builder.appName(self.component_name).getOrCreate()


    def read_data(self, path):
        filepath = os.path.join(self.root_directory, self.input_directory, path)
        return self._spark.read.csv(filepath, header=True, schema=source_schema)

    def read_parquet(self, path):
        return self._spark.read.parquet(path)

    @staticmethod
    def calculate_the_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance.
        :param lat1: start latitude
        :param lon1: start longitude
        :param lat2: end latitude
        :param lon2: end longitude
        :return haversine distance """
        return (
                F.lit(6371) * F.acos(
            F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) *
            F.cos(F.radians(lon2) - F.radians(lon1)) +
            F.sin(F.radians(lat1)) * F.sin(F.radians(lat2))
        )
        )

    @staticmethod
    def reorder(df, columns):
        """Gathering from dataframe only the needed columns
        :param df: dataframe
        :param columns: list of columns which we will not include
        :return df: reordered dataframe"""
        return df.select([col for col in df.columns if col not in columns])


    @staticmethod
    def categorize_data(df, input_column, output_column, method):
        """
        Categorize data using StringIndexer or OneHotEncoder.
        :param df: Spark DataFrame
        :param input_column: Column to be transformed
        :param output_column: Output column after transformation
        :param method: 'StringIndexer' or 'OneHotEncoder'
        """
        if method == 'StringIndexer':
            transformer = StringIndexer(inputCol=input_column, outputCol=output_column)
        elif method == 'OneHotEncoder':
            transformer = OneHotEncoder(inputCols=[input_column], outputCols=[output_column])
        else:
            raise ValueError("Invalid method. Choose 'StringIndexer' or 'OneHotEncoder'.")
        return transformer.fit(df).transform(df)

    @staticmethod
    def scale_numeric_columns(df, numeric_cols, output_col="scaled_numeric_features"):
        """
        Scale numeric columns using StandardScaler.
        :param df: Spark DataFrame
        :param numeric_cols: List of numeric columns to scale
        :param output_col: Output column for scaled features
        """
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
        df = assembler.transform(df)
        scaler = StandardScaler(inputCol="numeric_features", outputCol=output_col)
        scaler_model = scaler.fit(df)
        return scaler_model.transform(df)

    @staticmethod
    def assemble_features(df, feature_cols, output_col="features"):
        """
        Assemble multiple feature columns into a single feature vector.
        :param df: Spark DataFrame
        :param feature_cols: List of feature columns to assemble
        :param output_col: Output column for assembled features
        """
        assembler = VectorAssembler(inputCols=feature_cols, outputCol=output_col)
        return assembler.transform(df)

    def prepare_ml_data(self, df):
        """
        Prepare data for machine learning by categorizing, scaling, and assembling features.
        :param df: Spark DataFrame
        """

        df = self.categorize_data(df, "payment_type", "payment_type_index", "StringIndexer")
        df = self.categorize_data(df, "payment_type_index", "payment_type_encoded", "OneHotEncoder")
        df = self.categorize_data(df, "company", "company_index", "StringIndexer")
        df = self.categorize_data(df, "company_index", "company_encoded", "OneHotEncoder")

        df = self.add_interaction_features(df)
        df = df.filter(
            (F.col("trip_seconds") > 0) &
            (F.col("trip_miles") > 0) &
            (F.col("trip_total") > 0)
        )

        df = df.drop("payment_type", "company")
        numeric_cols = ["trip_seconds", "trip_miles", "direct_distance",  "distance_per_second", "minutes", "hours", "speed_mph"]
        df = df.select([F.col(c).cast(DoubleType()).alias(c) if c in numeric_cols else F.col(c) for c in df.columns])
        df = self.scale_numeric_columns(df, numeric_cols)


        feature_cols  = ["scaled_numeric_features",
                        "payment_type_encoded",
                        "company_encoded",
                        "hour_of_day",
                        "is_weekend",
                        ]
        df = df.dropna(subset=["trip_total"] + feature_cols)
        df_final = self.assemble_features(df, feature_cols)


        df_final = df_final.select("month", "features", "trip_total")
        return df_final

    def transform_data(self, df):
        df = self.reorder(df, nulls_columns)

        for col_name in points_columns:
            df = (
                df.withColumn(f"{col_name}_longitude",
                              F.regexp_extract(col_name, r"POINT \(([-\d.]+)", 1).cast("double"))
                .withColumn(f"{col_name}_latitude",
                            F.regexp_extract(col_name, r"POINT \([-.\d]+ ([\d.]+)\)", 1).cast("double"))
            )

        df = self.reorder(df, points_columns)

        df = df.withColumn(
            "direct_distance",
            self.calculate_the_distance(
                F.col("pickup_centroid_latitude"), F.col("pickup_centroid_longitude"),
                F.col("dropoff_centroid_latitude"), F.col("dropoff_centroid_longitude")
            )
        )

        df = df.fillna({
            "fare": 0.0, "tips": 0.0, "tolls": 0.0, "extras": 0.0,
            "trip_seconds": 0.0, "trip_miles": 0.0,
            "pickup_community_area": -1, "dropoff_community_area": -1, "direct_distance": 0.0
        })
        default_datetime = F.lit("1990-01-01 00:00:00").cast(TimestampType())
        df = df.withColumn(
            "trip_start_timestamp",
            F.when(F.col("trip_start_timestamp").isNull(), default_datetime)
            .otherwise(F.col("trip_start_timestamp"))
        )
        df = df.withColumn("month", F.date_format(F.col("trip_start_timestamp"), "yyyy-MM"))
        return df


    def save_data(self, df, output_path):
        df.write.partitionBy("month").parquet(output_path, mode="append")


    def do_etl(self, input_path):
        """Main function to execute the data preprocessing workflow."""
        try:

            df = self.read_data(input_path)
            df.createOrReplaceTempView("table")

            df = self.transform_data(df)
            output_path = os.path.join(self.root_directory, self.output_directory, "processed_data")


            self.save_data(df, output_path)

            self._complete_time = datetime.now()
            print(f"Data preprocessing completed in: {self._complete_time - self._start_time}")

        except Exception as e:
            print(f"An error occurred during processing: {e}")


    def add_interaction_features(self, df):
        """
        Add interaction features to the dataframe.
        :param df: Spark DataFrame
        :return: DataFrame with interaction features
        """
        df = df.withColumn("distance_per_second", df["trip_miles"] / df["trip_seconds"])
        df = df.withColumn("minutes", F.col("trip_seconds") / 60)
        df = df.withColumn("hours", F.col("trip_seconds") / 60 / 60)
        df = df.withColumn("speed_mph", F.when(F.col("trip_seconds") != 0, (
                    F.col("trip_miles") / (F.col("hours")))).otherwise(0))
        df = df.withColumn("hour_of_day", F.hour(df["trip_start_timestamp"]))
        df = df.withColumn("is_weekend", (F.dayofweek(df["trip_start_timestamp"]) >= 6).cast("int"))

        return df


    def predict_data(self):
        df = self.read_parquet(self.output_directory)
        df = self.prepare_ml_data(df)
        print(df.count())

        # train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        #
        # models = {
        #     "Linear Regression": LinearRegression(featuresCol="features", labelCol="trip_total"),
        #     "Random Forest Regressor": RandomForestRegressor(featuresCol="features", labelCol="trip_total"),
        # }
        #
        # results = {}
        # for model_name, model in models.items():
        #     trained_model = model.fit(train_data)
        #     predictions = trained_model.transform(test_data)
        #
        #
        #     evaluator = RegressionEvaluator(labelCol="trip_total", predictionCol="prediction", metricName="rmse")
        #     rmse = evaluator.evaluate(predictions)
        #     r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
        #
        #
        #     results[model_name] = {
        #         "model": trained_model,
        #         "predictions": predictions.toPandas(),
        #         "rmse": rmse,
        #         "r2": r2
        #     }
        #
        #
        #     predictions_df = predictions.select("month", "trip_total", "prediction").toPandas()
        #     predictions_df.to_csv(f"{model_name}_predictions_month_1-2.csv", index=False)
        #
        #
        # for model_name, result in results.items():
        #     print(f"Model: {model_name}")
        #     print(f" - RMSE: {result['rmse']}")
        #     print(f" - R²: {result['r2']}")
        #     print()
        #
        #
        #     plt.figure(figsize=(8, 4))
        #     plt.scatter(result["predictions"]["trip_total"], result["predictions"]["prediction"], alpha=0.6)
        #     plt.plot([min(result["predictions"]["trip_total"]), max(result["predictions"]["trip_total"])],
        #              [min(result["predictions"]["trip_total"]), max(result["predictions"]["trip_total"])],
        #              color="red")
        #     plt.title(f"Actual vs Predicted - {model_name}")
        #     plt.xlabel("Actual Trip Total")
        #     plt.ylabel("Predicted Trip Total")
        #     plt.show()

