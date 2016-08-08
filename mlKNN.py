# -*- coding: utf-8 -*-
"""


@author: huihui
"""

import performance
import math

#情感词典由DUTIR和卡方检验获得的扩展情感词组成
def dic(fileName):
    dic_intention_polarity={}
    f=open(fileName,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        temp=[lineSet[1],lineSet[2],lineSet[3]]
        dic_intention_polarity[lineSet[0]]=temp
    return dic_intention_polarity



# def for_weight(lineSet,dic_intention_polarity,dic_kafang,dic_smile,a,b):
#     weight=[0 for i in range(21)]
#     emotion_21=['PA','PE','PD','PH','PG','PB','PK','NA','NB','NJ','NH','PF','NI','NC','NG','NE','ND','NN','NK','NL','PC']
#     for word in lineSet:
#         if dic_intention_polarity.has_key(word):
#             for i in range(0,21):
#                 if dic_intention_polarity[word][0]==emotion_21[i]:
#                     #print dic_intention_polarity[word][0]
#                     x=int(dic_intention_polarity[word][1])
#                     y=int(dic_intention_polarity[word][2])
#                     weight[i]=weight[i]+a*x+b*y
#                     break
#         if dic_kafang.has_key(word):
#             for i in range(0,21):
#                 if dic_kafang[word][0]==emotion_21[i]:
#                     #print dic_intention_polarity[word][0]
#                     x=int(dic_kafang[word][1])
#                     y=int(dic_kafang[word][2])
#                     weight[i]=weight[i]+a*x+b*y
#                     break
#         if dic_smile.has_key(word):
#             for i in range(0,21):
#                 if dic_smile[word][0]==emotion_21[i]:
#                     #print dic_intention_polarity[word][0]
#                     x=int(dic_smile[word][1])
#                     y=int(dic_smile[word][2])
#                     weight[i]=weight[i]+a*x+b*y
#                     break
#     return weight

#把一条微博表示为一个向量
#特征选择方法为卡方统计
#获取对应特征的值，可能没有，如果没有x=0  


#将训练集合中有情绪的微博表示为向量
def train_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b):
    train_vec=[]
    f=open('corpus\\sentence_train_quzao.txt_fenci','r')
    for line in f.readlines():
        lineSet=line.strip().split()
        weight=for_weight(lineSet,dic_intention_polarity,dic_kafang,dic_smile,a,b)
        train_vec.append(weight)
    return train_vec


#将测试集合中有情绪的微博表示为向量
def test_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b):
    test_vec=[]
    f_w=open('corpus\\sentence_test_quzao.txt_fenci','r')
    for line in f_w.readlines():
        lineSet=line.strip().split()
        weight=for_weight(lineSet,dic_intention_polarity,dic_kafang,dic_smile,a,b)
        test_vec.append(weight)
    return test_vec
    
#计算一个向量的k近邻,返回各个情绪类别的数量n[],train_test_flag=0 train
def find_neighbor(test_weight,train_vec,k,train_test_flag,train_label):
        #相似度值
        cos=[]
        for train_weight in train_vec:
            #训练集中邻居不包括自己
            if train_test_flag==0 and train_weight==test_weight:
                continue
            up=0.0
            down1=0.0
            down2=0.0
            for i in range(0,len(train_weight)):
                up=up+train_weight[i]*test_weight[i]
                down1=down1+train_weight[i]*train_weight[i]
                down2=down2+test_weight[i]*test_weight[i]
            down1=math.sqrt(down1)
            down2=math.sqrt(down2)
            if down1==0 or down2==0:
                cos_theta=-2
                cos.append(cos_theta)
                continue
            #print("train %r"%(down1))
            #print("test %r"%(down2))
            down=down1*down2
            cos_theta=up*1.0/down
            cos.append(cos_theta)
        #降序排序，取前k个相似度
        cosTemp=sorted(cos)
        cosTemp.reverse()
        train_index=[]
        for i in range(0,k):
            for j in range(0,len(cos)):
                if cos[j]==cosTemp[i]:
                    train_index.append(j)
                    cos[j]=2
                    break
        # 计算结果train_label
        emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
        num=[0 for i in range(0,8)]
        for i in train_index:
            for j in range(0,len(emotion)):
                if train_label[i]==emotion[j]:
                    num[j]=num[j]+1
        return num

########################################################################################################
#只针对有情绪的
def map(k,train_label, train_vec):
    #1从训练集中计算先验概率PH[emotion][1]
    emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
    m=len(train_label)
    #训练集中每个类别的数量
    n=[0 for i in range(0,8)]
    for i in range(0,len(train_label)):
        for j in range(0,len(emotion)):
            if train_label[i]==emotion[j]:
                n[j]=n[j]+1
       #平滑s
    s=1
    #先验概率，计算为类别c的概率ph[i][1]，不是类别c的概率ph[i][0]
    ph=[[0 for col in range(2)] for row in range(8)]
    for i in range(0,len(emotion)):
        ph[i][1]=(s+n[i])*1.0/(s*2+m)
        ph[i][0]=1-ph[i][1]
    #2从训练集中计算后验概率
     #为训练集中的每一条微博计算它的K近邻
    dic_train_neighbor={}
    #dic_train_neighbor[0]=[邻居index]
    index=0
    for train_weight in train_vec:
        neighbor_emotion_num=find_neighbor(train_weight,train_vec,k,0,train_label)
        dic_train_neighbor[index]=neighbor_emotion_num
        index=index+1
    #对于每种情绪，计算出后验概率p(e[emotion][j]|h[emotion][1]),   p(e[emotion][j]|h[emotion][0])
    pe1=[[0 for col in range(k+1)] for row in range(8)]
    pe0=[[0 for col in range(k+1)] for row in range(8)]
    for i in range(0,len(emotion)):
         #情绪label i
         c1=[0 for ii in range(0,k+1)]
         c2=[0 for ii in range(0,k+1)]
         for j in range(0,m):
             #对于第J个训练样本
             temp=dic_train_neighbor[j][i]
             if train_label[j]==emotion[i]:
                 c1[temp]=c1[temp]+1
             else:
                 c2[temp]=c2[temp]+1
         sum_c1=0
         sum_c2=0
         for jj in range(0,k+1):
             sum_c1=sum_c1+c1[jj]
             sum_c2=sum_c2+c2[jj]
         for jj in range(0,k+1):
             pe1[i][jj]=(s+c1[jj])*1.0/(s*(k+1)+sum_c1)
             pe0[i][jj]=(s+c2[jj])*1.0/(s*(k+1)+sum_c2)
    return ph,pe1,pe0,n

#预测测试样本t,R(l)最大的所对应的情绪也就是预测情绪类别
def predict_test(test_weight,k,train_vec,ph,pe1,pe0,n,train_label):
    emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
    r=[0 for i in range(0,8)]
    #测试样本邻居对应的类别的数目
    neighbor_emotion_num=find_neighbor(test_weight,train_vec,k,1,train_label)
    #测试样本对于每个表情的判断
    temp_first=[]
    for i in range(0,len(emotion)):
        up=ph[i][1]*pe1[i][neighbor_emotion_num[i]]
        down=ph[i][1]*pe1[i][neighbor_emotion_num[i]] + ph[i][0]*pe0[i][neighbor_emotion_num[i]]
        r[i]=up*1.0/down
        #输出为向量
        if r[i]>0.5:
            temp_first.append(1)
        else:temp_first.append(0)

    rtemp=sorted(r)
    for i in range(0,len(emotion)):
        if r[i]==rtemp[7]:
            test_emotion_temp=emotion[i]
    return test_emotion_temp,temp_first

#########################################################################################################




#对测试集中的每一条微博，计算与训练集中每一条微博的向量相似度
#k=21
def mlKNN(k,a,b,dic_intention_polarity,dic_kafang,dic_smile):
    # fw=open('test_result_temp','w')
    #训练集label
    train_label=[]
    f_train_label=open('corpus\\sentence_train_label.txt','r')
    for line in f_train_label.readlines():
        lineSet=line.strip().split()
        train_label.append(lineSet[1])
    #训练vec
    train_vec=train_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b)
    #测试vec
    test_vec=test_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b)

    ph,pe1,pe0,n=map(k,train_label, train_vec)
    test_emotion=[]
    test_first=[]
    for test_weight in test_vec:
        test_emotion_temp,temp_first=predict_test(test_weight,k,train_vec,ph,pe1,pe0,n,train_label)
        test_emotion.append(test_emotion_temp)
        test_first.append(temp_first)
        print test_emotion_temp
    print len(test_emotion)
    return test_emotion,test_first

    
#合并情绪判断与情绪识别结果,也就是预测的结果
def merge_result(fileName2,emotion_label):
    #test_result_temp
    # f1=open(fileName1,'r')
    #result_emotion
    f2=open(fileName2,'r')
    f_out=open('result_old.txt','w')
    #记录情绪识别结果
    # emotion_label=[]
    # for line in f1.readlines():
    #     emotion_label.append(line.strip())
    test_out=[]
    index=0
    for line in f2.readlines():
        if line.strip()=='N':
            temp='none'
            test_out.append(temp)
            print>>f_out,temp
        else:
            temp=emotion_label[index]
            index=index+1
            test_out.append(temp)
            print>>f_out,temp
    return test_out


def score_emotion(fileName1,fileName2):
    #标注结果
    label=[]
    f=open(fileName1,'r')
    for line in f.readlines():
        labelLine=line.strip().split()
        label.append(labelLine[2])
    #预测结果
    result=[]
    f1=open(fileName2,'r')
    for line in f1.readlines():
        result.append(line.strip())
    #输出各类指标
    print("weibo情绪识别任务------------")        
 
    #宏平均
    class_dict1={'happiness':'happiness','like':'like','anger':'anger',
    'sadness':'sadness','fear':'fear','disgust':'disgust','surprise':'surprise'}
    macro_dict1=performance.calc_macro_average(result,label,class_dict1)
 
    
    #每一类情绪
    class_dict={'happiness':'happiness','like':'like','anger':'anger',
    'sadness':'sadness','fear':'fear','disgust':'disgust','surprise':'surprise',
    'none':'none'}
    #precision
    precision_dict=performance.calc_precision(result,label,class_dict)
    print("macro_precision——%r"%(macro_dict1['macro_p']))
    for i in class_dict:
        print("%r:%r"%(class_dict[i],precision_dict[class_dict[i]]))
    #recall
    recall_dict=performance.calc_recall(result,label,class_dict)
    print("macro_recall——%r"%(macro_dict1['macro_r']))
    for i in class_dict:
        print("%r:%r"%(class_dict[i],recall_dict[class_dict[i]]))   
    #f-measure
    fscore_dict=performance.calc_fscore(result,label,class_dict)
    print("macro_fscore——%r"%(macro_dict1['macro_f1']))
    for i in class_dict:
        print("%r:%r"%(class_dict[i],fscore_dict[class_dict[i]]))
    print("-------------------------")



if __name__ == '__main__':
    #人工标注标签
    # label=[]
    # f_label=open('label.txt','r')
    # for line in f_label.readlines():
    #     lineSet=line.strip().split()
    #     label.append(lineSet[2])
    # #
    # dic_intention_polarity=dic('dic_intention_polarity.txt')
    # dic_kafang=dic('kafang3000')
    # #smile
    # dic_smile=dic('smile_new_new.txt')

    # test_emotion_old=mlKNN(21,0.6,0.4,dic_intention_polarity,dic_kafang,dic_smile)
    # result_old= merge_result("result_emotion_bo.txt",test_emotion_old)
    # score_emotion('label.txt',result_old) 
    score_emotion('label.txt','result_old.txt') 
