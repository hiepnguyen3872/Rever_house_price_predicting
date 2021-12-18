# Rever_house-value_predicting

This project is about using Machine Learning to predict a house's price

## Description
This project is implemented in python language, use Random Forest Regressor Algorithm to predict a house's price

## Getting Started

### Dependencies
- `Python` >= 3.9.7
- `Pip` >= 21.2.3

### Installing

```sh
$ git clone https://github.com/hiepnguyen3872/Rever_house-value_predicting
$ cd Rever_house-value_predicting
```

```sh
$ tar -xf ./model/model.zip --directory ./model  
$ tar -xf ./data_overview/data_overview.zip --directory ./data_overview
```

### Executing program
- To predict house' sell price: 
```sh
$ python main.py data_path output_file_path
```
data_path is the path to data file, and data file must be in .csv type.

output_file_path is the path to file that will store result (house's sell price predict)

- To predict house' rental price, we just have to add flag 1: 
```sh
$ python main.py data_path output_file_path 1
```
