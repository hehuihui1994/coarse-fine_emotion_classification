# -*- coding: utf-8 -*-
"""
输入 类别 y = [1,0,1,0...]  train_label

输出 两种情绪转移概率

emotion_num = 8

七种情绪种类+none
4种相邻句子的情绪转移概率p1[8][8],p2[8][8],p3[8][8],p4[8][8]

微博整体与句子之间的情绪转移概率
4种转移概率p_weibo1[8][8],p_weibo2[8][8],p_weibo3[8][8],p_weibo4[8][8]

train_label 每个句子的[0,1,0...]
train_label_group 以微博为单位整理的句子的[[0,1,1,0,..],[],...]

"""

#计算相邻句子的四种转移概率p1,p2,p3,p4
def get_transfer_p_4(train_label_group, emotion_sentence_num, train_instance_num, emotion_num):
    #训练集合样本数
    sum1 = train_instance_num
    #i->j
    p1=[[0 for j in range(emotion_num)] for i in range(emotion_num)]
    #非i -> j
    p2=[[0 for j in range(emotion_num)] for i in range(emotion_num)]
    #i -> 非j
    p3=[[0 for j in range(emotion_num)] for i in range(emotion_num)]
    #非i -> 非j
    p4=[[0 for j in range(emotion_num)] for i in range(emotion_num)]

    for i in range(emotion_num):
        for j in range(emotion_num):
            down1=emotion_sentence_num[j]
            down0=sum1-emotion_sentence_num[j]
            #相邻句子之间
            up1=0
            up2=0
            up3=0
            up4=0 
            for train_instance_labels in train_label_group:
                if len(train_instance_labels) == 1:
                    continue
                for index in range(1,len(train_instance_labels)):
                    if train_instance_labels[index][j] == 1 and train_instance_labels[index-1][i] == 1:
                        up1 += 1
                    if train_instance_labels[index][j] == 1 and train_instance_labels[index-1][i] == 0:
                        up2 += 1
                    if train_instance_labels[index][j] == 0 and train_instance_labels[index-1][i] == 1:
                        up3 += 1
                    if train_instance_labels[index][j] == 0 and train_instance_labels[index-1][i] == 0:
                        up4 += 1
            p1[i][j]=up1*1.0/down1
            # print("%r - %r : %r"%(i,j,p1[i][j]))
            p2[i][j]=up2*1.0/down1
            p3[i][j]=up3*1.0/down0
            p4[i][j]=up4*1.0/down0
    return p1,p2,p3,p4

#计算微博整体与句子之间的四种情绪转移概率
#
def get_transfer_p_4_weibo(weibo_train_label, train_label_group, emotion_sentence_num, train_instance_num, emotion_num):
    #训练集合样本数
    sum1 = train_instance_num
    #i->j
    p1=[[0 for j in range(emotion_num)] for i in range(emotion_num)]
    #非i -> j
    p2=[[0 for j in range(emotion_num)] for i in range(emotion_num)]
    #i -> 非j
    p3=[[0 for j in range(emotion_num)] for i in range(emotion_num)]
    #非i -> 非j
    p4=[[0 for j in range(emotion_num)] for i in range(emotion_num)]

    for i in range(emotion_num):
        for j in range(emotion_num):
            down1=emotion_sentence_num[j]
            down0=sum1-emotion_sentence_num[j]
            up1=0
            up2=0
            up3=0
            up4=0 

            for k in range(len(weibo_train_label)):
                #第k条微博
                for sentence_index in range(len(train_label_group[k])):
                    if train_label_group[k][sentence_index][j] == 1 and weibo_train_label[k][i] == 1:
                        up1 += 1
                    if train_label_group[k][sentence_index][j] == 1 and weibo_train_label[k][i] == 0:
                        up2 += 1
                    if train_label_group[k][sentence_index][j] == 0 and weibo_train_label[k][i] == 1:
                        up3 += 1
                    if train_label_group[k][sentence_index][j] == 0 and weibo_train_label[k][i] == 0:
                        up4 += 1
            p1[i][j]=up1*1.0/down1
            # print("%r - %r : %r"%(i,j,p1[i][j]))
            p2[i][j]=up2*1.0/down1
            p3[i][j]=up3*1.0/down0
            p4[i][j]=up4*1.0/down0
    return p1,p2,p3,p4


if __name__ == '__main__':
    print(u"情绪转移概率模块..")