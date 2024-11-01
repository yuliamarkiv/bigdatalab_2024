from data_preprocessor.processor import DataPreprocessor



if __name__ == "__main__":
    input_directory = "data"
    output_directory = "preprocessed_data"
    preprocessor = DataPreprocessor(input_directory, output_directory)
    preprocessor.main("chicago_taxi_data_jan_to_jun_2024.csv")