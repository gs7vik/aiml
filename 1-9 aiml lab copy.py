1. A* algo

def aStarAlgo(start_node, stop_node):
         
        open_set = set(start_node) 
        closed_set = set()
        g = {} 
        parents = {}
 
        g[start_node] = 0
        parents[start_node] = start_node
        while len(open_set) > 0:
            n = None
            for v in open_set:
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                    n = v
            if n == stop_node or Graph_nodes[n] == None:
                pass
            else:
                for (m, weight) in get_neighbors(n):
                   #nodes 'm' not in first and last set are added to first
                    #n is set its parent
                    if m not in open_set and m not in closed_set:
                        open_set.add(m)
                        parents[m] = n
                        g[m] = g[n] + weight
                    else:
                        if g[m] > g[n] + weight:
                            g[m] = g[n] + weight
                            parents[m] = n
                            if m in closed_set:
                                closed_set.remove(m)
                                open_set.add(m)
            if n == None:
                print('Path does not exist!')
                return None
            if n == stop_node:
                path = []
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.append(start_node)
                path.reverse()
                print('Path found: {}'.format(path))
                return path
            open_set.remove(n)
            closed_set.add(n)
        print('Path does not exist!')
        return None
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
def heuristic(n):
        H_dist = {
            'A': 10,
            'B': 8,
            'C': 5,
            'D': 7,
            'E': 3,
            'F': 6,
            'G': 5,
            'H': 3,
            'I': 1,
            'J': 0             
        }
        return H_dist[n] 
Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1),('H', 7)] ,
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)],
     
}
aStarAlgo('A', 'J')


2. AO* algo

 class Graph:
    def __init__(self, graph, heuristicNodeList, startNode):
        self.graph = graph
        self.H=heuristicNodeList
        self.start=startNode
        self.parent={}
        self.status={}
        self.solutionGraph={}
    def applyAOStar(self):        
        self.aoStar(self.start, False)
    def getNeighbors(self, v):     
        return self.graph.get(v,'')
    def getStatus(self,v):        
        return self.status.get(v,0)
    def setStatus(self,v, val):    
        self.status[v]=val
    def getHeuristicNodeValue(self, n):
        return self.H.get(n,0)     
    def setHeuristicNodeValue(self, n, value):
        self.H[n]=value     
    def printSolution(self):
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:",self.start)
        print(self.solutionGraph)
    def computeMinimumCostChildNodes(self, v):   
        minimumCost=0
        costToChildNodeListDict={}
        costToChildNodeListDict[minimumCost]=[]
        flag=True
        for nodeInfoTupleList in self.getNeighbors(v):  
            cost=0
            nodeList=[]
            for c, weight in nodeInfoTupleList:
                cost=cost+self.getHeuristicNodeValue(c)+weight
                nodeList.append(c)
            if flag==True:                       
                minimumCost=cost
                costToChildNodeListDict[minimumCost]=nodeList      
                flag=False
            else:                                  
                if minimumCost>cost:
                    minimumCost=cost
                    costToChildNodeListDict[minimumCost]=nodeList  
        return minimumCost, costToChildNodeListDict[minimumCost]
    def aoStar(self, v, backTracking):
        print("HEURISTIC VALUES  :", self.H)
        print("SOLUTION GRAPH    :", self.solutionGraph)
        print("PROCESSING NODE   :", v)
        if self.getStatus(v) >= 0:
            minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
            self.setHeuristicNodeValue(v, minimumCost)
            self.setStatus(v,len(childNodeList))
            solved=True 
            for childNode in childNodeList:
                self.parent[childNode]=v
                if self.getStatus(childNode)!=-1:
                    solved=solved & False
            if solved==True:    
                self.setStatus(v,-1)    
                self.solutionGraph[v]=childNodeList
            if v!=self.start:   
                self.aoStar(self.parent[v], True)
            if backTracking==False: 
                for childNode in childNodeList: 
                    self.setStatus(childNode,0)   
                    self.aoStar(childNode, False)
h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph1 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]   
}
G1= Graph(graph1, h1, 'A')
G1.applyAOStar() 
G1.printSolution()


 3. Candidate elimination

import pandas as pd
import numpy as np 
data=pd.read_csv('trainingexamplesN.csv')
content=np.array(data)[:,:-1]
target=np.array(data)[:,-1]
def train(con,tar):
     specific=['0' for x in range(5)]
     general=['?' for x in range(5)]
     print("Initial condition")
     print("specific"+str(specific))
     print("General"+str(general)+"\n\n")
 
     specific_h=con[0].copy()
     general_h=[['?' for x in range(len(specific_h))] for x in range(len(specific_h))]
     for i,val in enumerate(con):
         if tar[i]=='Y':
             for x in range(len(specific_h)):
                 if(val[x]!=specific_h[x]):
                     specific_h[x]='?'
                     general_h[x][x]='?'
         else:
             for x in range(len(specific_h)):
                 if(val[x]!=specific_h[x]):
                     general_h[x][x]=specific_h[x]
                 else:
                     general_h[x][x]='?'
         print("Iteration ["+str(i+1)+"]")
         print("specific"+str(specific_h))
         print("General"+str((general_h))+"\n\n")
 
     general_h=[general_h[i] for i,val in enumerate(general_h) if val!=['?' for x in range(len(specific_h))]]
     return specific_h,general_h
 
specific,general=train(content,target)
print("specific"+str(specific))
print("General"+str((general))+"\n\n")

 4.ID3 algorithm


import math
import csv
def load_csv(filename):
    lines=csv.reader(open(filename,"r"));
    dataset = list(lines)
    headers = dataset.pop(0)
    return dataset,headers
class Node:
    def __init__ (self,attribute):
        self.attribute=attribute
        self.children=[]
        self.answer=""
def subtables(data,col,delete):
    dic={}
    coldata=[row[col] for row in data]
    attr=list(set(coldata))
    counts=[0]*len(attr)
    r=len(data)
    c=len(data[0])
    for x in range(len(attr)):
        for y in range(r):
            if data[y][col]==attr[x]:
                counts[x]+=1
    for x in range(len(attr)):
        dic[attr[x]]=[[0 for i in range(c)] for j in range(counts[x])]
        pos=0
        for y in range(r):
            if data[y][col]==attr[x]:
                if delete:
                    del data[y][col]
                dic[attr[x]][pos]=data[y]
                pos+=1
    return attr,dic
def entropy(S):
    attr=list(set(S))
    if len(attr)==1:
        return 0
    counts=[0,0]
    for i in range(2):
        counts[i]=sum([1 for x in S if attr[i]==x])/(len(S)*1.0)
    sums=0
    for cnt in counts:
        sums+=-1*cnt*math.log(cnt,2)
    return sums

def compute_gain(data,col):
    attr,dic = subtables(data,col,delete=False)
    total_size=len(data)
    entropies=[0]*len(attr)
    ratio=[0]*len(attr)
    total_entropy=entropy([row[-1] for row in data])
    for x in range(len(attr)):
        ratio[x]=len(dic[attr[x]])/(total_size*1.0)
        entropies[x]=entropy([row[-1] for row in dic[attr[x]]])

        total_entropy-=ratio[x]*entropies[x]
    return total_entropy
def build_tree(data,features):
    lastcol=[row[-1] for row in data]
    if(len(set(lastcol)))==1:
        node=Node("")
        node.answer=lastcol[0]
        return node
    n=len(data[0])-1
    gains=[0]*n
    for col in range(n):
        gains[col]=compute_gain(data,col)
    split=gains.index(max(gains))
    node=Node(features[split])
    fea = features[:split]+features[split+1:]
    attr,dic=subtables(data,split,delete=True)
    for x in range(len(attr)):
        child=build_tree(dic[attr[x]],fea)
        node.children.append((attr[x],child))
    return node
def print_tree(node,level):
    if node.answer!="":
        print(" "*level,node.answer)
        return
    print(" "*level,node.attribute)
    for value,n in node.children:
        print(" "*(level+1),value)
        print_tree(n,level+2)

def classify(node,x_test,features):
    if node.answer!="":
        print(node.answer)
        return
    pos=features.index(node.attribute)
    for value, n in node.children:
        if x_test[pos]==value:
            classify(n,x_test,features)
dataset,features=load_csv("traintennis.csv")
node1=build_tree(dataset,features)
print("The decision tree for the dataset using ID3 algorithm is")
print_tree(node1,0)
testdata,features=load_csv("testtennis.csv")
for xtest in testdata:
    print("The test instance:",xtest)
    print("The label for test instance:",end=" ")
    classify(node1,xtest,features)



5. ANN algo


 import numpy as np 
 X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) 
 y = np.array(([92], [86], [89]), dtype=float) 
 X = X/np.amax(X,axis=0) 
 y = y/100 
 def sigmoid (x): 
     return 1/(1 + np.exp(-x)) 
 def derivatives_sigmoid(x): 
     return x * (1 - x)
 epoch=5000 	
 lr=0.1 
 inputlayer_neurons = 2 
 hiddenlayer_neurons = 3 
 output_neurons = 1 
 wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons)) 
 bh=np.random.uniform(size=(1,hiddenlayer_neurons)) 
 wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons)) 
 bout=np.random.uniform(size=(1,output_neurons)) 
 for i in range(epoch):
     hinp1=np.dot(X,wh) 
     hinp=hinp1 + bh 
     hlayer_act = sigmoid(hinp) 
     outinp1=np.dot(hlayer_act,wout) 
     outinp= outinp1+ bout 
     output = sigmoid(outinp) 
     EO = y-output 
     outgrad = derivatives_sigmoid(output) 
     d_output = EO* outgrad 
     EH = d_output.dot(wout.T) 
     hiddengrad = derivatives_sigmoid(hlayer_act) 
     d_hiddenlayer = EH * hiddengrad 
 wout += hlayer_act.T.dot(d_output) *lr 
 wh += X.T.dot(d_hiddenlayer) *lr 
 print("Input: \n" + str(X))  
 print("Actual Output: \n" + str(y)) 
 print("Predicted Output: \n" ,output)


 6. naive bayes

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
data=pd.read_csv('traintennis.csv')
print("first five records\n",data.head())
x=data.iloc[:,:-1]
print("first five train data\n",x.head())
y=data.iloc[:,-1]
print("first five train output\n",y.head())
x=x.copy()
le_Outlook = LabelEncoder()
x.Outlook = le_outlook.fit_transform(x.Outlook)
le_Temperature=LabelEncoder()
x.Temperature=le_Outlook.fit_transform(x.Temperature)
le_Humidity=LabelEncoder()
x.Humidity=le_Outlook.fit_transform(x.Humidity)
le_Wind=LabelEncoder()
x.Wind=le_Outlook.fit_transform(x.Wind)
print("after encoding train data\n",x.head())
y=y.copy()
le_PlayTennis=LabelEncoder()
y=le_PlayTennis.fit_transform(y)
print("after encoding test data\n",y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
classifier=GaussianNB()
classifier.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
print("Accuracy is:", accuracy_score(classifier.predict(x_test), y_test))


7. EM algo


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
l1 = [0,1,2]
def rename(s):
	l2 = []
	for i in s:
		if i not in l2:
			l2.append(i)
	for i in range(len(s)):
		pos = l2.index(s[i])
		s[i] = l1[pos]
	return s
iris = datasets.load_iris()
print("\n IRIS DATA :",iris.data);
print("\n IRIS FEATURES :\n",iris.feature_names) 
print("\n IRIS TARGET  :\n",iris.target) 
print("\n IRIS TARGET NAMES:\n",iris.target_names)
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
plt.figure(figsize=(14,7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1,2,1)
plt.scatter(X.Sepal_Length,X.Sepal_Width, c=colormap[y.Targets], s=40)
plt.title('Sepal')
plt.subplot(1,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Petal')
plt.show()
print("Actual Target is:\n", iris.target)
model = KMeans(n_clusters=3)
model.fit(X)
plt.figure(figsize=(14,7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1,2,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
plt.subplot(1,2,2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()
km = rename(model.labels_)
print("\nWhat KMeans thought: \n", km)
print("Accuracy of KMeans is ",sm.accuracy_score(y, km))
print("Confusion Matrix for KMeans is \n",sm.confusion_matrix(y, km))
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
print("\n",xs.sample(5))
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Classification')
plt.show()
em = rename(y_cluster_gmm)
print("\nWhat EM thought: \n", em)
print("Accuracy of EM is ",sm.accuracy_score(y, em))
print("Confusion Matrix for EM is \n", sm.confusion_matrix(y, em))


8.KNN algo


 from sklearn.datasets import load_iris 
 from sklearn.neighbors import KNeighborsClassifier 
 import numpy as np 
 from sklearn.model_selection import train_test_split 
  
 iris_dataset=load_iris() 
 print("\n IRIS FEATURES \ TARGET NAMES: \n ", iris_dataset.target_names) 
 for i in range(len(iris_dataset.target_names)): 
     print("\n[{0}]:[{1}]".format(i,iris_dataset.target_names[i])) 
 print("\n IRIS DATA :\n",iris_dataset["data"])
 X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0) 
 print("\n Target :\n",iris_dataset["target"]) 
 print("\n X TRAIN \n", X_train) 
 print("\n X TEST \n", X_test) 
 print("\n Y TRAIN \n", y_train) 
 print("\n Y TEST \n", y_test) 
 kn = KNeighborsClassifier(n_neighbors=5) 
 kn.fit(X_train, y_train) 
 x_new = np.array([[5, 2.9, 1, 0.2]]) 
 print("\n XNEW \n",x_new) 
 prediction = kn.predict(x_new) 
 print("\n Predicted target value: {}\n".format(prediction)) 
 print("\n Predicted feature name: {}\n".format(iris_dataset["target_names"][prediction])) 
 i=1 
 x= X_test[i] 
 x_new = np.array([x]) 
 print("\n XNEW \n",x_new) 
 for i in range(len(X_test)): 
   x = X_test[i] 
   x_new = np.array([x]) 
   prediction = kn.predict(x_new) 
   print("\n Actual : {0} {1}, Predicted :{2}{3}".format(y_test[i],iris_dataset["target_names"][y_test[i]],prediction,iris_dataset["target_names"][ prediction])) 
 print("\n TEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test)))


 9. regression algo


import numpy as np
import matplotlib.pyplot as plt
def local_regression(x0, X, Y, tau):
    x0 = [1, x0]
    X = [[1, i] for i in X]
    X = np.asarray(X)
    xw = (X.T) * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau))
    beta = np.linalg.pinv(xw @ X) @ xw @ Y @ x0
    return beta
def draw(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.plot(X, Y, 'o', color='black')
    plt.plot(domain, prediction, color='red')
    plt.show()   
X = np.linspace(-3, 3, num=1000)
domain = X
Y = np.log(np.abs(X ** 2 - 1) + .5)
draw(10)
draw(0.1)
draw(0.01)
draw(0.001)



