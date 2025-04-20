#!/bin/bash

mkdir "data"

years=("2015" "2016" "2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024") 

base_url="https://portal.inmet.gov.br/uploads/dadoshistoricos"

for year in "${years[@]}"; do
    echo "Processing year: $year"
    
    # Define the zip file name and URL
    zip_file="${year}.zip"
    url="${base_url}/${zip_file}"
    
    # Download the zip file
    wget "$url"
    
    # Create a directory for the year if it doesn't exist
    mkdir -p "data/$year"
    
    # Unzip and extract only the files containing "PICO DO COUTO"
    unzip -j $zip_file "**/*PICO DO COUTO*" "*PICO DO COUTO*" -d "data/$year"

    # Delete zip file
    rm $zip_file

done

echo "Processing complete."