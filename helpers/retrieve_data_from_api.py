import requests
import os

output_file = os.path.join(os.getcwd(), "chicago_taxi_data_jan_to_jun_2024.csv")

base_url = "https://data.cityofchicago.org/resource/ajtu-isnz.csv?"

query = "$where=trip_start_timestamp between '2024-01-01T00:00:00' and '2024-06-30T23:59:59'"
limit = 1000
offset = 0


with open(output_file, 'w') as file:
    while True:
        url = f"{base_url}{query}&$limit={limit}&$offset={offset}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            break


        if offset == 0:
            file.write(response.text)
        else:

            file.write('\n'.join(response.text.splitlines()[1:]))

        if len(response.text.splitlines()) <= limit:
            break

        offset += limit

print(f"Data has been saved to {output_file}")
