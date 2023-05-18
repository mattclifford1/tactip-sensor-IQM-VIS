# IQM-VIS for tactip sensor calibration

## Setup
Create python environment, e.g. via conda:
```
conda create -n tactip_iqm_vis python=3.9 -y
conda activate tactip_iqm_vis
```
Install dependancies
```
pip install -r requirements.txt
```
## Running the tactip example
Define and run the UI using
```
python make_tactip_UI.py
```
![Alt text](https://github.com/mattclifford1/tactip-sensor-IQM-VIS/blob/main/pics/sensor-calibration.png?raw=true "Tactip UI") 

## Running the example with posenet errors
You can also include posenet errors, but this requires extra data for the error and needs to create the custom data handler for the UI's data API - defined in [posenet/data_holder.py](posenet/data_holder.py)
```
python posenet/make_UI.py
```
Note that you will need the weights for the pytorch posenet (message me for them).
