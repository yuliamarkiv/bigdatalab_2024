import dash
import pandas
from dash import dcc, html, Input, Output
import plotly.express as px
from pyspark.sql import functions as f, Window
from source.processor import DataPreprocessor


class TaxiTripDashboard:
    def __init__(self, df_spark, simultaneous_trips_data, predicted_trips):
        self.df_spark = df_spark
        self.df_filtered = self.preprocess_data(df_spark)
        self.valid_months = self.get_valid_months()
        self.simultaneous_trips_df = simultaneous_trips_data
        self.aggregated_data = predicted_trips


    def preprocess_data(self, df_spark):
        """Preprocess the dataset by excluding invalid rows and adding columns."""
        df = df_spark.filter(f.col("month") != "1990-01").filter(f.col("month") != "2024-12")
        df = df.withColumn("weekday", f.date_format("trip_start_timestamp", "EEEE"))
        df = df.withColumn("hour", f.hour("trip_start_timestamp"))
        df = df.withColumn(
            "day_period",
            f.when(f.col("hour").between(0, 6), "Night")
             .when(f.col("hour").between(7, 12), "Morning")
             .when(f.col("hour").between(13, 18), "Afternoon")
             .otherwise("Evening"),
        )
        return df

    def get_valid_months(self):
        """Get a list of valid months from the dataset."""
        return (
            self.df_filtered.select("month")
            .distinct()
            .orderBy("month")
            .toPandas()["month"]
            .tolist()
        )

    def get_weekday_distribution(self, selected_month):
        """Get data for the weekday distribution pie chart."""
        month_data = self.df_filtered.filter(f.col("month") == selected_month)
        weekday_data = (
            month_data.groupBy("weekday")
            .count()
            .withColumnRenamed("count", "trip_count")
            .orderBy(f.col("trip_count").desc())
            .toPandas()
        )
        return px.pie(
            weekday_data,
            names="weekday",
            values="trip_count",
            title=f"Trip Distribution by day of the week for {selected_month}",
            labels={"weekday": "Day of the Week", "trip_count": "Number of Trips"}
        )

    def get_peak_hour_distribution(self, selected_month):
        """Get data for the peak hours line plot."""
        month_data = self.df_filtered.filter(f.col("month") == selected_month)
        peak_hour_data = (
            month_data.groupBy("hour")
            .count()
            .withColumnRenamed("count", "trip_count")
            .orderBy("hour")
            .toPandas()
        )
        return px.line(
            peak_hour_data,
            x="hour",
            y="trip_count",
            title=f"Peak Hour Distribution for {selected_month}",
            labels={"hour": "Hour of the Day", "trip_count": "Number of Trips"}
        )

    def get_geospatial_analysis(self, selected_month):
        """Get geospatial data for busiest pickup locations."""
        month_data = self.df_filtered.filter(f.col("month") == selected_month)
        pickup_data = (
            month_data.groupBy("pickup_centroid_latitude", "pickup_centroid_longitude")
            .count()
            .withColumnRenamed("count", "trip_count")
            .orderBy(f.col("trip_count").desc())
            .toPandas()
        )
        return px.scatter_mapbox(
            pickup_data,
            lat="pickup_centroid_latitude",
            lon="pickup_centroid_longitude",
            size="trip_count",
            title=f"Busiest Pickup Locations for {selected_month}",
            mapbox_style="carto-positron",
            zoom=10
        )

    def average_fare_per_trip(self, selected_month):
        """Calculate the average fare per trip."""
        month_data = self.df_filtered.filter(f.col("month") == selected_month)
        fare_data = (
            month_data.groupBy("day_period")
            .agg(f.avg("fare").alias("avg_fare"))
            .orderBy("day_period")
            .toPandas()
        )
        return px.bar(
            fare_data,
            x="day_period",
            y="avg_fare",
            title=f"Average Fare Per Trip by Time of Day ({selected_month})",
            labels={"day_period": "Time of Day", "avg_fare": "Average Fare ($)"}
        )

    def calculate_simultaneous_trips(self, df):
        """
        Calculates the maximum number of simultaneous trips that happened for each month.

        Returns:
            DataFrame: A DataFrame containing the maximum number of simultaneous trips for each month.
        """
        # Creating a dataframe for trip start and end events
        pickup_dataframe = (df.filter(f.col('trip_start_timestamp').isNotNull()).
                            select(f.col('trip_start_timestamp').alias('event_time'),
                                   f.lit(1).alias('event_count')))

        dropoff_dataframe = (df.filter(f.col('trip_end_timestamp').isNotNull()).
                             select(f.col('trip_end_timestamp').alias('event_time'),
                                    f.lit(-1).alias('event_count')))

        # Combining the events
        event_dataframe = pickup_dataframe.union(dropoff_dataframe)

        # Calculating simultaneous trips
        dataframe = event_dataframe.withColumn('sum', f.sum('event_count').over(Window.partitionBy('event_time')
                                                                                .orderBy(f.asc('event_time'))))

        # Group by month
        dataframe = dataframe.withColumn('month', f.date_format('event_time', 'yyyy-MM')) \
            .groupBy('month') \
            .agg(f.max('sum').alias('simultaneous_trips')) \
            .orderBy(f.desc(f.col('simultaneous_trips')))

        # Convert to pandas for easier handling with Plotly/Dash
        df_result = dataframe.toPandas()

        # Save to CSV
        df_result.to_csv('simultaneous_trips_per_month.csv', index=False)

        return df_result
    def revenue_by_payment_method(self, selected_month):
        """Analyze revenue by payment method."""
        month_data = self.df_filtered.filter(f.col("month") == selected_month)
        payment_data = (
            month_data.filter(f.col("payment_type").isNotNull())
            .filter(f.col("payment_type") != "Unknown")
            .groupBy("payment_type")
            .agg(f.sum("fare").alias("total_revenue"))
            .orderBy(f.col("total_revenue").desc())
            .toPandas()
        )
        return px.pie(
            payment_data,
            names="payment_type",
            values="total_revenue",
            title=f"Revenue by Payment Method ({selected_month})",
            labels={"payment_type": "Payment Method", "total_revenue": "Total Revenue ($)"}
        )

    def get_simultaneous_trips(self, selected_month):
        """Get simultaneous trips data for the selected month."""
        filtered_data = self.simultaneous_trips_df[self.simultaneous_trips_df["month"] == selected_month]
        if not filtered_data.empty:
            return filtered_data["simultaneous_trips"].values[0]
        return 0

    def get_actual_vs_predicted(self, selected_month):
        """Get the actual vs predicted data for the selected month."""
        month_data = self.aggregated_data[self.aggregated_data["month"] == selected_month]

        month_data["month"] = pandas.to_datetime(month_data["month"]).dt.strftime('%b %Y')

        month_data_long = month_data.melt(id_vars=["month"], value_vars=["trip_total", "prediction"],
                                          var_name="type", value_name="revenue")

        return px.bar(
            month_data_long,
            x="month",
            y="revenue",
            color="type",
            title=f"Actual vs Predicted Revenue for {selected_month}",
            labels={"month": "Month", "revenue": "Revenue ($)", "type": "Type"},
            barmode='group'
        )

    def create_app(self):
        """Create the Dash app."""
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("Taxi Trip Dashboard"),
            html.Div([
                dcc.Dropdown(
                    id="month-filter",
                    options=[{"label": m, "value": m} for m in self.valid_months],
                    value=self.valid_months[0],
                    clearable=False
                )
            ]),

            html.Div([
                html.H3(id="simultaneous-trips-title", style={"font-size": "24px", "color": "black"}),
                html.H3(id="simultaneous-trips-number", style={"font-size": "32px", "color": "blue"}),
            ], style={"text-align": "center"}),

            html.Div([
                dcc.Graph(id="weekday-distribution"),
                dcc.Graph(id="peak-hours-line-plot"),
            ], style={"display": "flex", "justify-content": "space-between"}),

            html.Div([
                dcc.Graph(id="avg-fare-plot"),
                dcc.Graph(id="revenue-by-payment"),
            ], style={"display": "flex", "justify-content": "space-between"}),
            html.Div([
                dcc.Graph(id="geospatial-map"),
            ]),
            html.Div([
                dcc.Graph(id="actual-vs-predicted"),
            ],  style={"display": "flex", "justify-content": "center"}),
        ])

        @app.callback(
            [
                Output("weekday-distribution", "figure"),
                Output("peak-hours-line-plot", "figure"),
                Output("avg-fare-plot", "figure"),
                Output("revenue-by-payment", "figure"),
                Output("geospatial-map", "figure"),
                Output("simultaneous-trips-title", "children"),
                Output("simultaneous-trips-number", "children"),
                Output("actual-vs-predicted", "figure"),
            ],
            [Input("month-filter", "value")]
        )
        def update_dashboard(selected_month):
            weekday_pie_fig = self.get_weekday_distribution(selected_month)
            peak_hours_line_fig = self.get_peak_hour_distribution(selected_month)
            avg_fare_fig = self.average_fare_per_trip(selected_month)
            revenue_payment_fig = self.revenue_by_payment_method(selected_month)
            geospatial_map_fig = self.get_geospatial_analysis(selected_month)
            simultaneous_trips = self.get_simultaneous_trips(selected_month)
            actual_vs_predicted_revenue_fig = self.get_actual_vs_predicted(selected_month)

            return (
                weekday_pie_fig,
                peak_hours_line_fig,
                avg_fare_fig,
                revenue_payment_fig,
                geospatial_map_fig,
                f"Simultaneous Trips for {selected_month}",
                simultaneous_trips,
                actual_vs_predicted_revenue_fig
            )

        return app


preprocessor = DataPreprocessor()
df_spark = preprocessor.read_parquet("/Users/ymarkiv/PycharmProjects/bigdata-university/data/processed_data")
simultaneous_trips = pandas.read_csv("/Users/ymarkiv/PycharmProjects/bigdata-university/predictions/simultaneous_trips_per_month.csv")
predicted_trips = pandas.read_csv("/Users/ymarkiv/PycharmProjects/bigdata-university/predictions/aggregated_actual_predicted.csv")
dashboard = TaxiTripDashboard(df_spark, simultaneous_trips, predicted_trips)


app = dashboard.create_app()
if __name__ == "__main__":
    app.run_server(debug=True)
