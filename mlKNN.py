# -*- coding: utf-8 -*-
"""
样本 x=[.....]  train_vec
类别 y=[1,0,1,0...]  train_label
test_weight 一个样本
k 近邻
@author: huihui
"""

import math

    
#计算一个样本test_weight的k近邻,返回各个情绪类别的数量n[],train_test_flag=0 
#样本x在训练集中的K近邻，根据这些邻居的标签集，可以计算属于第ι个类别的邻居个数
#train_vec为训练样本集合
#train_label为 [1,0,1,0,0,...]
#emotion_num为情绪类别个数
def find_neighbor(test_weight,train_vec,k,train_test_flag,train_label,emotion_num):
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
            down=down1*down2
            cos_theta=up*1.0/down
            cos.append(cos_theta)

        #降序排序，取前k个相似度
        #前k个近邻的下标存储在train_index中
        cosTemp=sorted(cos)
        cosTemp.reverse()
        train_index=[]
        for i in range(0,k):
            for j in range(0,len(cos)):
                if cos[j]==cosTemp[i]:
                    train_index.append(j)
                    cos[j]=2
                    break

        # 计算C_x结果train_label
        num=[0 for i in range(0,emotion_num)]
        for i in train_index:
            #train_label[i]为y_a(l)
            for j in range(emotion_num):
                num[j] += train_label[i][j]
        return num

#从训练集中计算频度得到先验概率p(HLb) 和后验概率 p(ELC⃗ t(L)|HLb)
def map(k,train_label, train_vec,emotion_num):
    #1从训练集中计算先验概率PH[emotion][1]
    #m为训练样本数量
    m=len(train_label)
    #平滑
    s=1
    #训练样本中每个类别的数量
    sum_y = [ 0 for i in range(emotion_num)]
    for i in range(m):
        for j in range(emotion_num):
            sum_y[j] += train_label[i][j]
    #先验概率，计算为类别j的概率ph[i][1]，不是类别j的概率ph[i][0]
    ph=[[0 for col in range(2)] for row in range(emotion_num)]
    for j in range(emotion_num):
        #\sum_{1}^{m}\vec y_{x_i}(L)
        ph[j][1] = (s + sum_y[j])*1.0/(s*2+m)
        ph[j][0]=1-ph[j][1]

    #2从训练集中计算后验概率
    #为训练集中的每一条微博计算它的K近邻
    dic_train_neighbor={}
    #dic_train_neighbor[0] 第0个样本的邻居的标签集[2,3,0,3,...]
    index=0
    for train_weight in train_vec:
        neighbor_emotion_num=find_neighbor(train_weight,train_vec,k,0,train_label,emotion_num)
        dic_train_neighbor[index]=neighbor_emotion_num
        index=index+1
    #对于每种情绪，计算出后验概率p(e[emotion][j]|h[emotion][1]),   p(e[emotion][j]|h[emotion][0])
    pe1=[[0 for col in range(k+1)] for row in range(emotion_num)]
    pe0=[[0 for col in range(k+1)] for row in range(emotion_num)]
    for j in range(emotion_num):
         #情绪label i
         c1=[0 for ii in range(0,k+1)]
         c2=[0 for ii in range(0,k+1)]
         for i in range(m):
             #对于第i个训练样本的邻居的第j个类别的总和
             temp=dic_train_neighbor[i][j]
             if train_label[i][j] == 1:
                c1[temp] += 1
            else:
                c2[temp]=c2[temp]+1
         sum_c1=0
         sum_c2=0
         for jj in range(0,k+1):
             sum_c1=sum_c1+c1[jj]
             sum_c2=sum_c2+c2[jj]
         for jj in range(0,k+1):
             pe1[j][jj]=(s+c1[jj])*1.0/(s*(k+1)+sum_c1)
             pe0[j][jj]=(s+c2[jj])*1.0/(s*(k+1)+sum_c2)
    return ph,pe1,pe0


#预测测试样本t,R(l)最大的所对应的情绪也就是预测情绪类别
def predict_test(test_weight,k,train_vec,ph,pe1,pe0,train_label,emotion_num):
    # emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
    r=[0 for j in range(emotion_num)]
    #测试样本邻居对应的类别的数目
    neighbor_emotion_num=find_neighbor(test_weight,train_vec,k,1,train_label,emotion_num)
    #测试样本对于每个表情的判断[1,0,1,0,1...]
    temp_first=[]
    for j in range(emotion_num):
        up=ph[j][1]*pe1[j][neighbor_emotion_num[j]]
        down=ph[j][1]*pe1[j][neighbor_emotion_num[j]] + ph[j][0]*pe0[j][neighbor_emotion_num[j]]
        r[j]=up*1.0/down
        #输出为向量
        if r[j]>0.5:
            temp_first.append(1)
        else:
            temp_first.append(0)
    # rtemp=sorted(r)
    # for i in range(emotion_num):
    #     #r[i]最大，则输出类别标签为i
    #     if r[i]==rtemp[emotion_num-1]
    #         test_emotion_temp=i
    return temp_first

#输入train_vec,train_label,test_vec,emotion_num,k
#对于每一个测试样本输出test_label [1,0,1,0]
def mlknn_demo(train_vec,train_label,test_vec,emotion_num,k):
    ph,pe1,pe0 = map(k,train_label, train_vec,emotion_num)
    test_label = []
    for test_weight in test_vec:
        temp_first = predict_test(test_weight,k,train_vec,ph,pe1,pe0,train_label,emotion_num)
        test_label.append(temp_first)
    return test_label


if __name__ == '__main__':
   print "this is MLKNN algorithm"