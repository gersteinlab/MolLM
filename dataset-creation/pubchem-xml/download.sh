#!/bin/bash

# input file that contains the names of the files to download
input="file_names"

# Base URL
base_url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/XML"

# max number of parallel processes
max_jobs=10

# read each line in the input file
while IFS= read -r file_name
do
  # create the full URL
  full_url="${base_url}/${file_name}"

  echo "Downloading ${file_name} from ${full_url}"

  # use wget to download the file and gunzip to decompress it
  (wget -O - "$full_url" | gunzip -c > "${file_name%.gz}") &

  # wait if there are too many jobs
  while (( $(jobs -p | wc -l) >= max_jobs ))
  do
    sleep 1 # wait before checking again
  done

done < "$input"

# wait for all background jobs to finish
wait

echo "Download and uncompression process has been completed."
