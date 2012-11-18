'''
Created on Nov 15, 2012

@author: ralekar
'''
from numpy.lib.function_base import blackman

'''
Created on Nov 14, 2012

@author: ralekar
'''
import re,sys,math,operator
class Ddict(dict):
    
    def __init__(self, default=None):
        self.default = default

    def __getitem__(self, key):
        if not self.has_key(key):
            self[key] = self.default()
        return dict.__getitem__(self, key)

def createBlocks():

    ftrain=open("D:/Dropbox/DataMining/Assignment3/sales_train.csv","r").readlines()
    flabel=open("D:/Dropbox/DataMining/Assignment3/sales_labels.txt","r").readlines()
    step=len(ftrain)/10
    if len(ftrain)%10!=0:
        remainder=len(ftrain)%10
    
    dictLabel={}
    for label in flabel:
        label=re.split(",",str(label.strip()))
        for lbl in label:
            dictLabel[lbl]=0
   
    
    blockPart=[]     
    for testBlock in range(0,len(ftrain),step):
        if testBlock+step+remainder<=len(ftrain):
            blockPart.append(testBlock)
            blockPart.append(testBlock+step)
    blockPart[-1]+=remainder
            
    for i in range(2,len(blockPart),2):
        blockPart[i]+=1            
    
    return ftrain,blockPart,dictLabel


def generate2DimHashMap():
    return Ddict(dict)
            
def generateFeatureDataStructure():
    global featureDictionary
    featureDictionary=[]
    
    for feat in range(0,int(sys.argv[1])):
        feature=generate2DimHashMap()
        featureDictionary.append(feature)
    
def calculateTrainLabels(label,trainLabel):
    if label in trainLabel:
        temp=trainLabel[label]
        temp+=1
        trainLabel[label]=temp
    if label not in trainLabel:
        trainLabel[label]=1    
    return trainLabel    
    
def featureProbablity():
    global featureDictionary
    
    ftrain,blockPart,dictLabel=createBlocks()
    trainBlock=[]
    testBlock=[]
       
    for t in range(0,len(blockPart),2):
        trainLabels={}
        testLabels={}
        generateFeatureDataStructure()
        testBlock=ftrain[blockPart[t]:blockPart[t+1]]
        for ind in range(0,len(blockPart),2):
            if blockPart[ind] != blockPart[t] and blockPart[t+1]!=blockPart[ind+1]:
                trainBlock=ftrain[blockPart[ind]:blockPart[ind+1]]
                for line in trainBlock:
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
        mean,meanSquare,standardDeviation,dictLabel=calculateNormal(dictLabel)      
        calculateDiscreteAttributes(dictLabel)
        labelTestSet(testBlock,mean,meanSquare,standardDeviation,dictLabel)
        

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
        featureDictionary[int(attributeIndex)]=normalDistribution(feature, mean, meanSquare, standardDeviation, dictLabel,True,0.0,"")
        
    return mean,meanSquare,standardDeviation,labelCount 

def calculateDiscreteAttributes(dictLabel):
    global featureDictionary
    global SAMPLE_SIZE
    global PARAMETER
    SAMPLE_SIZE=5
    PARAMETER=0.5
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
                        value=value+(SAMPLE_SIZE*PARAMETER)
                        denominator=denominator+(SAMPLE_SIZE)
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
            meanSquare[label]=float(value/(count-1))           
        except:
            meanSquare[label]=0.0
            pass
    return meanSquare            
                      
def calculateStandardDeviation(meanSquare):
    
    standardDeviation={}    
    for label in meanSquare:
        standardDeviation[label]=math.sqrt(float(meanSquare[label]))
        

    return standardDeviation

def normalDistribution(feature,mean,meanSquare,standardDeviation,dictLabel,flag,testValue,testLabel):
          
               if flag==True:
                  for feat in feature:
                      for label in dictLabel:
                          means=float(mean[label])
                          deviation=float(standardDeviation[label])
                          key=float(feat)*float(feature[feat][label])
                          pie=(float(1/(math.sqrt(2)*math.pi)))*(1/deviation)
                          exponent=(-0.5)*((key-means)*(key-means))/(deviation*deviation)
                          feature[feat]["probability_"+label]=float(pie*math.exp(exponent))
                  
                  return feature        
               if flag==False:
                   probability=0.0
                   means=float(mean[testLabel])
                   deviation=float(standardDeviation[testLabel])
                   key=float(testValue)
                   pie=(float(1/(math.sqrt(2)*math.pi)))*(1/deviation)
                   exponent=(-0.5)*((key-means)*(key-means))/(deviation*deviation)
                   probability[label]=float(pie*math.exp(exponent))
                   return probability
               return {}
           



    
           

def labelTestSet(testBlock,mean,meanSquare,standardDeviation,dictLabel):
    
    global featureDictionary
    global SAMPLE_SIZE
    global PARAMETER
    dicts={}
    dicts["ok"]=0
    dicts["fraud"]=0
    testLabels=[]
    trainLabels=[]
    
    continous_attributes={}
    for t in range(2,len(sys.argv)):
        continous_attributes[int(sys.argv[t])]=0
    probabilityDict={}
    for line in testBlock:
        tokens=re.split(",",str(line))
        trainLabels.append(tokens[-1])
        tokens=tokens[:-1]
        
        probabilityDict=initProbabilityDict(probabilityDict,dictLabel)
        index=0
        for label in dictLabel:
            for feature in featureDictionary:
               if index not in continous_attributes: 
                   attributes=str(tokens[index].strip())
                   if attributes in feature:
                       value=float(probabilityDict[label])
                       probabilityDict[label]=value*float(feature[attributes]["probability_"+label]) 
                       if attributes not in feature:
                           denominator=float(dictLabel[label])+PARAMETER
                           value=SAMPLE_SIZE*PARAMETER
                           probabilityDict[label]*=float(value/denominator)    
               if index in continous_attributes:
                   tempProbability=normalDistribution({},mean,meanSquare,standardDeviation,dictLabel,False,attributes,label)
                   probabilityDict[label]*=float(tempProbability)
        
            
                   
        label=maxLabel(probabilityDict)
        index+=1
        testLabels.append(label) 
    generatePPV(trainLabels,testLabels)   
def generatePPV(trainLabels,testLabels):
   index=0
   truePositives=0.0
   falsePositives=0.0
   trueNegatives=0.0
   falseNegatives=0.0
   
   for train in trainLabels:
       if train.strip()=="ok" and testLabels[index].strip()=="ok":
                    truePositives+=1.0
       if train.strip()=="ok" and testLabels[index].strip()=="fraud":
                    falseNegatives+=1.0
       if train.strip()=="fraud" and testLabels[index].strip()=="fraud":
                    trueNegatives+=1.0
       if train.strip()=="fraud" and testLabels[index].strip()=="ok":
                    falsePositives+=1.0                
       index+=1
   print  "truePositives: ",truePositives,"falseNegatives: ",falseNegatives,"trueNegatives: ",trueNegatives,"falsePositives: ",falsePositives
   
def initProbabilityDict(dicts,labels):
    
    for label in labels:
        dicts[label]=1.0
    return dicts


def maxLabel(probability):
                   
                   sorted_probability = sorted(probability.iteritems(), key=operator.itemgetter(1))
                   label=sorted_probability[0][0]
                   if sorted_probability[0][1]==sorted_probability[1][1]:
                         label="ok"
                   return label
               
               
def main():
    
    global featureDictionary
    global SAMPLE_SIZE
    global PARAMETER
   
    generateFeatureDataStructure()
    featureProbablity()
    
    
if __name__=="__main__":
    main()



