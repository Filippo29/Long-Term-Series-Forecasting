# Long Time Series Forecasting
This is a reproduction of the methodology presented in the [MICN](https://openreview.net/pdf?id=zt53IDUR1U) paper applied to weather forecasting.
You can find other train/test data from [here](https://www.bgc-jena.mpg.de/wetter/).  
  
To run the code you first need to install all the needed libraries with
```bash
pip install -r requirements.txt
```
  
If you start from scratch you can change the parameters in the first cell as you prefer and then just press run all to start training and then evaluation. If you simply want to evaluate a pre trained checkpoint you still need to run the first two cell to import dataset and libraries, then to start the evaluation run the cell under "Evaluate on the test set" changing, if needed, the variable ```save_name```.  
  
The training process will go ahead until the max number of epochs is reached or when there is no improvement for a number of epochs higher than ```patience```.

| Parameter  | Value   |
| --------   | ------- |
| lr         | 1e-6    |
| gamma      | 0.9     |
| patience   | 3       |
| d_model    | 256     |
| batch size | 256     |
| train size | 80%     |
| test size  | 10%     |
| valid size | 10%     |
## Results
Over different runs these were the results obtained:
| Input size  | Prediction size  | Best epoch  | Train Loss  | Test Loss  |
| --------    | -------          | --------    | -------     | -------    |
| 100         | 200              | 28          | 56993.57    | 70416.55   |
| 100         | 300              | 29          | 59151.43    | 68915.04   |
| 100         | 500              | 26          | 60565.61    | 69564.67   |