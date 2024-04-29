# History-independent detection of knee-point of lithium-ion battery lifetime

The code in this repository shows how to utilizing the data associated with the paper 'History-independent detection of knee-point in lithium ion battery lifetime' by Hyunjae Kim, Inwoo Kim, Minsoo KIM et al. The data is available at add link . There you can also find more details about the data.

This analysis was originally performed in python.

The data utilizing codes can be grouped into one of three categories: RPT feature, 0.2C-1C_delta Q(V)_feature, and deg-SL estimation & 2nd lifetime prediction regression codes.

RPT_feature: Extracting 0.2C, 1C capacity, DCIR@SOC50, DCIR versus SOC (SOC0~80) in every RPTs

0.2C-1C_delta Q(V)_feature: Extracting  Q0.2C-1C(V), Var(delta Q0.2C-1C(V)) in every RPTs

regresssion_deg-SL  : Machine learning code for estimating deg-SL.
regresssion_2nd_lifetime & dummy_regression_2nd_lifetime :  Machine learning code for predicting 2nd lifetime.
