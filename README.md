# History-independent-knee-point-detection-in-lithium-ion-battery-lifetime

The code in this repository shows how to utilizing the data associated with the paper 'History dependent knee point detection in lithium ion battery lifetime' by Hyunjae Kim, Inwoo Kim, Minsoo KIM et al. The data is available at add link . There you can also find more details about the data.

This analysis was originally performed in python.

The data utilizing codes can be grouped into one of three categories: delta Q0.2C-1C related data, RPT summary data, and DCIR@SOC5 at EOL data, deg-SL and 2ndlifetime regression codes.

delta Q0.2C-1C related data: Extracting  Q0.2C-1C(V), Var (delta Q0.2C-1C(V)) in every RPTs

RPT summary data: Extracting 0.2C, 1C capacity, DCIR@SOC50, dQdV &dVdQ of 0.2C and 1C charging/discharing curves in every RPTs

DCIR@SOC5 at EOL data: Extracting DCIR@SOC5 at EOL data from GITT at EOL.

regresssion_deg-SL  : Machine learning code for estimating deg-SL.
regresssion_2nd_lifetime & dummy_regression_2nd_lifetime :  Machine learning code for predicting 2nd lifetime.
