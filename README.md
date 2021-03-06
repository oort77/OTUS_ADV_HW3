
**OTUS Machine Learning Advanced**
### **Homework 3**

### Cluster analysis of time series
![clockworks](https://user-images.githubusercontent.com/73858914/153416075-05ab7f45-3186-40af-8727-f62892fc1976.png)  
**Goals:**  

Apply clustering algorithms to multiple time series (TS):
- For a list of cryptocurrencies (CC), including Bitcoin (~100), download ☑︎
    - 30 days of closing prices
    - 3 days hourly closing prices
    - 1 hour minute closing prices
- Compare KNeighbours basic and DTW approaches;  ☑︎
- Try TS clustering after automatic feature generation with TSFEL;  ☑︎
- Analyze composition of clusters without Bitcoin.  ☑︎

**Additional goals:**  

(separate notebooks)
- Build risk profile for CC portfolios: one consisting of top 5 largest CCs and  
another being a mix from two most populous clusters; ☑︎
- Apply Matrix Profile methods from [matrixprofile](https://github.com/matrix-profile-foundation/matrixprofile) library to find repeated  
motifs and discords (anomalies) on CC from various clusters.


**Means:**  

- All TS computations will be done in [tslearn](https://github.com/tslearn-team/tslearn) and
[tsfel](https://github.com/fraunhoferportugal/tsfel).

**Data:**  

- CC prices will be downloaded with [cryptocompare](https://github.com/lagerfeuer/cryptocompare) library.

**Binder notebooks:**

otus_adv_hw3_dtw.ipynb: tslearn and tsfel clustering   

clustering_mef.ipynb: Markowitz Efficient Frontier. Trying to beat 5 largest  
cryptocurrencies with clustering approach

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/oort77/OTUS_ADV_HW3/main)  
