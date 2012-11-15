'''
Created on Nov 14, 2012

@author: ralekar
'''
import re,sys,math
class Ddict(dict):
    
    def __init__(self, default=None):
        self.default = default

    def __getitem__(self, key):
        if not self.has_key(key):
            self[key] = self.default()
        return dict.__getitem__(self, key)

def dataCleaning():
    
    fread=open("D:/Dropbox/DataMining/Assignment3/salestable.csv","r").readlines()
    fclean=open("D:/Dropbox/DataMining/Assignment3/sales_clean.csv","w")
    
    for line in fread:
        line=re.split(",",line)
        writer=""
        flag=False
        index=0
        for token in line:
            token=re.sub(r'\"',"",token)
            line[index]=token
            index+=1
        
        for token in line:
            token=token.strip()
            if token=="NA":
                flag=True
            if token!=line[-1].strip():        
                writer+=token+","
            else:
                writer+=token+"\n"
        if flag==False:
            fclean.write(writer) 
    fclean.close()
    fclean=open("D:/Dropbox/DataMining/Assignment3/sales_clean.csv","r").readlines()
    dataPartition(fclean)
    
    
def dataPartition(fread):
    
    ftrain=open("D:/Dropbox/DataMining/Assignment3/sales_train.csv","w")
    ftest=open("D:/Dropbox/DataMining/Assignment3/sales_test.csv","w")
    for line in fread:
        line=re.split(",",str(line.strip()))
        string=",".join(line)
        if line[-1].strip()=='fraud' or line[-1].strip()=='ok':
            ftrain.write(string+"\n")    
        if line[-1].strip()=='unkn':
            ftest.write(string+"\n")    
                 
    ftest.close()
    ftrain.close()

def generate2DimHashMap():
    return Ddict(dict)
            
def generateFeatureDataStructure():
    global featureDictionary
    featureDictionary=[]
    
    for feat in range(0,int(sys.argv[1])):
        feature=generate2DimHashMap()
        featureDictionary.append(feature)
    
    
def featureProbablity():
    global featureDictionary
    
    ftrain=open("D:/Dropbox/DataMining/Assignment3/sales_train.csv","r").readlines()
    flabel=open("D:/Dropbox/DataMining/Assignment3/sales_labels.txt","r").readlines()
    
    dictLabel={}
    for label in flabel:
        label=re.split(",",str(label.strip()))
        for lbl in label:
            dictLabel[lbl]=0
    
                            
    for line in ftrain:
        index=0
        line=re.split(",",str(line.strip()))
        
        for feature in featureDictionary:
            if line[index] in feature:
                if line[-1] in feature[line[index]]:  
                    count=int(feature[line[index]][line[-1]])
                    count+=1
                    feature[line[index]][line[-1]]=count
                    featureDictionary[index]=feature
                    
                
            if line[index] not in feature:
                for label in dictLabel:
                    feature[line[index]][label]=0
                    feature[line[index]]["probability_"+label]=0
                
                feature[line[index]][line[-1]]=1
                featureDictionary[index]=feature
                           
            index+=1 
    dictLabel=calculateNormal(dictLabel)      
    calculateDiscreteAttributes(dictLabel)
   
    
def calculateNormal(dictLabel):
    
    global featureDictionary
    
    labelCount={}
    mean={}
    meanSquare={}
    standardDeviation={}
    
    continous_attributes=[]
    
    
    
    for t in range(2,len(sys.argv)):
        continous_attributes.append(int(sys.argv[t]))
        
    
    for attributeIndex in continous_attributes:
        feature=featureDictionary[int(attributeIndex)]
        labelCount,mean=calculateMean(feature,dictLabel)
        meanSquare=calculateMeanSquare(feature,mean,dictLabel)
        standardDeviation=calculateStandardDeviation(meanSquare)
        featureDictionary[int(attributeIndex)]=normalDistribution(feature, mean, meanSquare, standardDeviation, dictLabel)
        
    return labelCount 

def calculateDiscreteAttributes(dictLabel):
    global featureDictionary
    continous_attributes={}
    for t in range(2,len(sys.argv)):
        continous_attributes[int(sys.argv[t])]=0
        index=0   
        for feature in featureDictionary:
            if index not in continous_attributes:
                for feat in feature:
                    for label in dictLabel:
                        value=float(feature[feat][label])
                        denominator=float(dictLabel[label])
                        feature[feat]["probability_"+label]=float(value/denominator)
                     
            index+=1            
    


    
def calculateMean(feature,dictLabel):
    
    
    labelCount={}
    mean={}
    for label in dictLabel:
        mean[label]=0.0
        meanCount=0
        for feat in feature:
            if label in feature[feat]:
                multiplier=float(feature[feat][label])
                meanCount+=multiplier
                number=float(feat)*multiplier
                value=float(mean[label])
                mean[label]=number+value
        value=mean[label]
        labelCount[label]=meanCount
        try:
            mean[label]=float(value/meanCount)
           
        except:
            mean[label]=0.0
            pass
                
    return labelCount,mean          

def calculateMeanSquare(feature,mean,dictLabel):
    
    meanSquare={}
    for label in dictLabel:
        meanSquare[label]=0.0
        count=0
        for feat in feature:
            if label in feature[feat]:
                count+=float(feature[feat][label])
                value=(float(feat)-float(mean[label]))*(float(feat)-float(mean[label]))*float(feature[feat][label])
                number=meanSquare[label]
                meanSquare[label]=number+value
        value=meanSquare[label]
        try:
            meanSquare[label]=float(value/count)           
        except:
            meanSquare[label]=0.0
            pass
    return meanSquare            
                      
def calculateStandardDeviation(meanSquare):
    
    standardDeviation={}    
    for label in meanSquare:
        standardDeviation[label]=math.sqrt(float(meanSquare[label]))
        

    return standardDeviation

def normalDistribution(feature,mean,meanSquare,standardDeviation,dictLabel):
          
                  for feat in feature:
                      for label in dictLabel:
                          means=float(mean[label])
                          deviation=float(standardDeviation[label])
                          key=float(feat)*float(feature[feat][label])
                          pie=(float(1/(math.sqrt(2)*math.pi)))*(1/deviation)
                          exponent=(-0.5)*((key-means)*(key-means))/(deviation*deviation)
                          feature[feat]["probability_"+label]=float(pie*math.exp(exponent))
                  
                  return feature        


def labelTestSet(dictLabel):
    global featureDictonary
    


               
def main():
    
    global featureDictionary
    global labelCount
    #dataCleaning()
    generateFeatureDataStructure()
    featureProbablity()
    files=open("D:/JavaDesignPatterns/Java/NaiveBayes/Data.Mining/TestTrain.txt","w")
    files.write(str(featureDictionary))
    featureDictionary[{},{},{},{}]
    
if __name__=="__main__":
    main()



