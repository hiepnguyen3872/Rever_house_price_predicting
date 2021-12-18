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
data_path is path to file data, and file data must be in .csv type.

output_file_path is path to file will store result (house's sell price predict)
