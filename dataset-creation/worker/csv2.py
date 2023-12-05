import os
import csv

def find_zip_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                yield os.path.join(root, file)

def create_csv(directory, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["zip_path", "epoch"])
        zip_files = sorted(list(find_zip_files(directory)))
        for epoch in range(1, 301):
            for zip_file in zip_files:
                writer.writerow([zip_file, epoch])

create_csv('output-text', 'output.csv')
