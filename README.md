
**OTUS Machine Learning Advanced**
### **Homework 3** - work in progress

### Cluster analysis of time series
![clockworks](https://user-images.githubusercontent.com/73858914/153416075-05ab7f45-3186-40af-8727-f62892fc1976.png)  
**Goals:**  

Apply clustering algorithms to multiple time series (TS):
- A list of cryptocurrencies (CC), including Bitcoin (~100)
    - 30 days of closing prices
    - 3 days hourly closing prices
    - 1 hour minute closing prices
- Compare KNeighbours basic and DTW approaches;
- Try TS clustering after automatic feature generation with TSFEL;
- Analyze composition of clusters without Bitcoin.

**Additional goals:**  

- Build risk profile for CC portfolios from one cluster and from
different clusters;
- Apply Matrix Profile methods from [matrixprofile](https://github.com/matrix-profile-foundation/matrixprofile) library to find repeated motifs and discords (anomalies) on CC from various clusters.


**Means:**  

- All TS computations will be done in [tslearn](https://github.com/tslearn-team/tslearn) and
[tsfel](https://github.com/fraunhoferportugal/tsfel).

**Data:**  

- CC prices will be downloaded with [cryptocompare](https://github.com/lagerfeuer/cryptocompare) library.

**Choice of models:**  



**Methodology:**  



**Colab notebooks:**

**______**  
  

**______**  
