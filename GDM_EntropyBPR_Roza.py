
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%         ****************** DATA3 *******************

Movie_num = 6
User_num = 5
DenseArray = np.zeros((User_num, Movie_num,Movie_num))
#Fix the address below
u1 = pd.read_csv("/Address.../2021101309.1/direct/adrian1.csv", index_col=0)
u2 = pd.read_csv("/Address.../2021101309.1/direct/anis1.csv", index_col=0)
u3 = pd.read_csv("/Address.../2021101309.1/direct/kristian1.csv", index_col=0)
u4 = pd.read_csv("/Address.../2021101309.1/direct/pedro1.csv", index_col=0)
u5 = pd.read_csv("/Address.../2021101309.1/direct/random1.csv", index_col=0)

u1 = pd.read_csv("/Address.../2021101309.1/indirect/adrian2.csv", index_col=0)
u2 = pd.read_csv("/Address.../2021101309.1/indirect/anis2.csv", index_col=0)
u3 = pd.read_csv("/Address.../2021101309.1/indirect/kristian2.csv", index_col=0)
u4 = pd.read_csv("/Address.../2021101309.1/indirect/pedro2.csv", index_col=0)
u5 = pd.read_csv("/Address.../2021101309.1/indirect/random2.csv", index_col=0)

u1 = pd.read_csv("/Address.../2021101309.2/direct/adrian1.csv", index_col=0)
u2 = pd.read_csv("/Address.../2021101309.2/direct/anis1.csv", index_col=0)
u3 = pd.read_csv("/Address.../2021101309.2/direct/kristian1.csv", index_col=0)
u4 = pd.read_csv("/Address.../2021101309.2/direct/pedro1.csv", index_col=0)
u5 = pd.read_csv("/Address.../2021101309.2/direct/random1.csv", index_col=0)

u1 = pd.read_csv("/Address.../2021101309.2/indirect/adrian2.csv", index_col=0)
u2 = pd.read_csv("/Address.../2021101309.2/indirect/anis2.csv", index_col=0)
u3 = pd.read_csv("/Address.../2021101309.2/indirect/kristian2.csv", index_col=0)
u4 = pd.read_csv("/Address.../2021101309.2/indirect/pedro2.csv", index_col=0)
u5 = pd.read_csv("/Address.../2021101309.2/indirect/random2.csv", index_col=0)

TopMovies = u1.columns
DenseArray[0] = u1.to_numpy()
DenseArray[1] = u2.to_numpy()
DenseArray[2] = u3.to_numpy()
DenseArray[3] = u4.to_numpy()
DenseArray[4] = u5.to_numpy()

TopUsers= range(User_num)
TopItems= range(Movie_num)
#%%

user_count =  User_num
item_count =  Movie_num
num_Run = 40  
#RemoveItemsList is list of data that are going to be removed from the training set.    
RemoveItems = pd.DataFrame(columns=['User', 'Movie1','Movie2'])

for n in range(num_Run): 
        item1 = np.random.randint(item_count-1)
        item2 = np.random.randint(item1+1, item_count)
        RemoveItems.loc[len(RemoveItems)] = [np.random.randint(user_count), item1 , item2]


#%%
#Matrix Factorization


user_count =  User_num
item_count =  Movie_num
epoch =1

repeat = 1
Sum_firstDisorder = 0
Sum_CorrectOrders = 0
Sum_Error_Entropy = 0

for rr in range(repeat):
    Error_pairscore_Entropy  = np.zeros((num_Run)) 
    MPlist = pd.DataFrame(index=range(num_Run), columns=TopMovies) #np.zeros((item_count, num_Run))
    MRlist = pd.DataFrame(index=range(num_Run), columns=TopMovies) #np.zeros((item_count, num_Run))  
    
    k=2  #len of embeddings
    lr=0.01  
    reg = 0.01
    n_epoch = 1000
    
    Order = 1
    firstDisorder = num_Run 
    
    for run in range(1,num_Run+1):
        print("repeat:", rr, "   Run:", run, "   Epoch:",epoch)
                          
        biasV = np.random.rand(item_count) * 0.01  
        # Initialize the embedding weights.
        U = np.random.rand(user_count, k) * 0.01  
        V = np.random.rand(item_count, k) * 0.01  

        for epoch in range(n_epoch): 
#            if epoch%1000 == 0:
#                print("repeat:", rr, "   Run:", run, "   Epoch:",epoch)
            for u in range(user_count):
                for i in range(item_count):
                    for j in range(item_count):
                        if [u, i, j] not in RemoveItems.values[:run].tolist():
                            r_uij = DenseArray[u][i][j]
                                                    
                            # Update weights by gradients.  
                            rp_ui = np.dot(U[u], V[i].T) + biasV[i] #rp is predicted rating
                            rp_uj = np.dot(U[u], V[j].T) + biasV[j]
                            rp_uij = rp_ui - rp_uj # rp is predicted pairwise rating
                
                            loss_func =  1.0 / (1 + np.exp(-rp_uij)) - r_uij  #-1.0 / (1 + np.exp(rp_uij))
                            
                              # update U and V
                            U[u] += -lr * (loss_func * (V[i] - V[j]) + reg * U[u]) #I write it according to BPR. Is it correct?????
                            
                            if r_uij >0: #prefered item must increse and less prefer one, decrease.
                                V[i] += -lr * (loss_func * U[u] + reg * V[i])
                                V[j] += -lr * (loss_func * (-U[u]) + reg * V[j])
                                # update biasV
                                biasV[i] += -lr * (loss_func + reg * biasV[i])
                                biasV[j] += -lr * (-loss_func + reg * biasV[j]) 
                            else:
                                V[j] += -lr * (loss_func * U[u] + reg * V[j])
                                V[i] += -lr * (loss_func * (-U[u]) + reg * V[i])       
                                # update biasV
                                biasV[j] += -lr * (loss_func + reg * biasV[j])
                                biasV[i] += -lr * (-loss_func + reg * biasV[i]) 
               
            
        UserEmbedding = U
        MovieEmbedding = V
        
        
        #*********** Evaluation: ************
        
        predict_scores = np.mat(UserEmbedding) * np.mat(MovieEmbedding.T)+ biasV
        PredictMatrix = pd.DataFrame(predict_scores, index=TopUsers, columns=TopMovies)   
        # I normalize the predicted output to [0,1] range, so it will be closer to the missing values range
        max_predicted = PredictMatrix.max().max() #### New update :)
        min_predicted = PredictMatrix.min().min() #### New update :)
        PredictMatrix_normalized = (PredictMatrix - min_predicted)/ (max_predicted - min_predicted) #### New update :)
        movies_predicted_order = PredictMatrix_normalized.mean(axis=0).sort_values(ascending=False).index 
        
        Original_Data = np.zeros((user_count, item_count))
        for u in range (user_count):
            Original_Data[u][:]= np.mean(DenseArray[u], axis=1)
        RealMatrix = pd.DataFrame(Original_Data, index=TopUsers, columns=TopMovies)  #1: Real order of the full matrix    
        max_Real = RealMatrix.max().max() #### New update :)
        min_Real = RealMatrix.min().min() #### New update :)
        RealMatrix_normalized = (RealMatrix - min_Real)/ (max_Real - min_Real) #### New update :)
        movies_real_order =  RealMatrix_normalized.mean(axis=0).sort_values(ascending=False).index                 
        
        
        ##****************  Errooooor:
        predicted_pairscores = np.zeros((user_count,item_count,item_count))
        for u in range(user_count):
            for i in range(item_count):
                for j in range(item_count):
                        rp_ui = np.dot(U[u], V[i].T) + biasV[i] 
                        rp_uj = np.dot(U[u], V[j].T) + biasV[j]
                        rp_uij = rp_ui - rp_uj
                        predicted_pairscores[u][i][j] = 1.0 / (1 + np.exp(-rp_uij))
                        
        SumDif = 0                
        for [u, i, j] in RemoveItems.values[:run].tolist():
            SumDif += abs(predicted_pairscores[u][i][j] - DenseArray[u][i][j])
        
        Error_pairscore_Entropy[run-1] = SumDif/num_Run  
        
        print(Error_pairscore_Entropy[run-1])


    ##****************

    plt.plot(range(1,num_Run+1), Error_pairscore_Entropy)
    plt.show()    
    
    #*********  
    
 
plt.xlabel('Number of missing values')
plt.ylabel('Error')
plt.plot(range(1,num_Run+1), Error_pairscore_Entropy,'g')
plt.plot(range(1,num_Run+1), Error_pairscore_Viedma,'r')
plt.show()    
    
#%%

# Recover the group rank when we have missing values : true ranking vs ranking of predicted matrix:
predict_pairscores_Entropy = np.zeros((user_count, item_count, item_count))   
for u in range(user_count):
    for i in range(item_count):
        for j in range(item_count):    
            if [u, i, j] in RemoveItems.values[:run].tolist():
                predict_pairscores_Entropy[u][i][j] = predicted_pairscores[u][i][j]
            else:
                predict_pairscores_Entropy[u][i][j] = DenseArray[u][i][j]

predict_scores_Entropy = np.zeros((user_count, item_count)) 
for u in range (user_count):
    predict_scores_Entropy[u][:]= np.mean(predict_pairscores_Entropy[u], axis=1)  
    
PredictMatrix_Entropy = pd.DataFrame(predict_scores_Entropy, index=TopUsers, columns=TopMovies)        
predicted_order_Entropy = PredictMatrix_Entropy.mean(axis=0).sort_values(ascending=False).index

Original_Data = np.zeros((user_count, item_count))
for u in range (user_count):
    Original_Data[u][:]= np.mean(DenseArray[u], axis=1)
RealMatrix = pd.DataFrame(Original_Data, index=TopUsers, columns=TopMovies)  #1: Real order of the full matrix    
real_order =  RealMatrix.mean(axis=0).sort_values(ascending=False).index 



#%%

Error=0
k=0
mpList = predicted_order_Entropy.tolist()
mrList = real_order.tolist()
for mm in mpList:
   Error += abs(mrList.index(mm) - k)
   k+= 1
Error_Entropy = float(Error/item_count)



print("ErrorEntropy:", Error_Entropy)
