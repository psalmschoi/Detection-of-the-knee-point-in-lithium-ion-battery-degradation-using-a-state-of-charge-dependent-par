#Detection  of the knee point in lithium ion battery degradation using a state of charge dependent parameter
The code in this repository shows how to utilizing the data associated with the paper 'Detection  of the knee point in lithium ion battery degradation using a state of charge dependent parameter ' by Hyunjae Kim, Inwoo Kim, Minsoo KIM et al. The data is available at https://doi.org/10.17632/zn82y35zr8.3. There you can also find more details about the data.

This analysis was originally performed in python.

The data utilizing codes can be grouped into one of three categories: RPT feature, 0.2C-1C_delta Q(V)_feature, and deg-SL estimation & 2nd lifetime prediction regression codes.

RPT_feature: Extracting 0.2C, 1C capacity, DCIR@SOC50, DCIR versus SOC (SOC0~80) in every RPTs

0.2C-1C_delta Q(V)_feature: Extracting  Q0.2C-1C(V), Var(delta Q0.2C-1C(V)) in every RPTs

regresssion_deg-SL  : Machine learning code for estimating deg-SL.

regresssion_2nd_lifetime :  Machine learning code for predicting 2nd lifetime.

2nd_lifetime application classification :  Machine learning code for classifying 2nd lifetime application

SOH degradation mode classification : Machine learning code for classifying SOH degradation mode
