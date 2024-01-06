import h_index as h

#generating data for table 4
recall = 1
mu = 0
sigma = 0
alpha0 = -0.5  
nu0 = 0
n = [4,5,6,7,8,9,10,11,12]
G = 285    
for i in range(len(n)):
    df = h.data_generating(recall, mu, sigma, alpha0, nu0, n[i], G)
    df.to_csv('table_4.csv', mode='a', header=True, index=True)
    
#generating data for table 5
recall = 1
mu = 0
sigma = 1
alpha0 = -0.5  
nu0 = 0
n = [2,3,4,5,6,7,8,9,10]
G = 285
for i in range(len(n)):
    df = h.data_generating(recall, mu, sigma, alpha0, nu0, n[i], G)
    df.to_csv('table_5.csv', mode='a', header=True, index=True)
