# -*- coding: utf-8 -*-
"""
两种情绪转移概率

训练集中每个句子的主要情绪
七种情绪种类+none
4种相邻句子的情绪转移概率p1[8][8],p2[8][8],p3[8][8],p4[8][8]

微博整体与句子之间的情绪转移概率
4种转移概率p_weibo1[8][8],p_weibo2[8][8],p_weibo3[8][8],p_weibo4[8][8]
@author: huihui
"""

import  xml.dom.minidom
import sys

reload(sys)
sys.setdefaultencoding('utf8')


#从xml文件中得到数组label[4000][]
def get_label():
    #训练集微博级别的标签
    weibo_label=[]
	#训练集句子级别的标签，每一行是一个微博的
    label=[]
    #训练集中句子情绪分布
    emotion_sentence_num=[0 for i in range(0,8)]
    #所有情绪类别
    emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise']
    #打开xml文档
    dom = xml.dom.minidom.parse('corpus\\train.xml')
    #得到文档元素对象
    root = dom.documentElement  
    #获得标签属性值
    itemlist = root.getElementsByTagName('weibo')

    for item in itemlist:
        label_sentence=[]
        weibo_emotion_type=item.getAttribute("emotion-type")
        weibo_label.append(weibo_emotion_type)
        cc=item.getElementsByTagName('sentence')
        for i in range(0,len(cc)):
            if cc[i].getAttribute("emotion_tag")!="N":
                emotion_type_1=cc[i].getAttribute("emotion-1-type")
                label_sentence.append(emotion_type_1)
                for i in range(0,len(emotion)):
                    if emotion_type_1==emotion[i]:
                        emotion_sentence_num[i]+=1
                        break
            else:
                label_sentence.append("none")
                emotion_sentence_num[7]+=1
                
        label.append(label_sentence)
    # print label[0][0]
    # print len(label)
    sum1=0
    for i in range(0,len(emotion_sentence_num)):
        sum1+=emotion_sentence_num[i]
    	# print emotion_sentence_num[i]
    # print sum1
    return label,emotion_sentence_num,sum1,weibo_label

#计算相邻句子的四种转移概率p1,p2,p3,p4
def get_transfer_p_4():
    label,emotion_sentence_num,sum1,weibo_label=get_label()
    emotion_all=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
    ############句子间的情绪转移概率
    #a->b
    p1=[[0 for j in range(8)] for i in range(8)]
    #非a -> b
    p2=[[0 for j in range(8)] for i in range(8)]
    #a -> 非b
    p3=[[0 for j in range(8)] for i in range(8)]
    #非a -> 非b
    p4=[[0 for j in range(8)] for i in range(8)]
    for i in range(0,8):
        for j in range(0,8):
            down1=emotion_sentence_num[j]
            down0=sum1-emotion_sentence_num[j]
            #相邻句子之间
            up1=0
            up2=0
            up3=0
            up4=0
            for x in label:
                if len(x)==1:
                    continue
                for x_label in range(1,len(x)):
                    if x[x_label]==emotion_all[j] and x[x_label-1]==emotion_all[i]:
                        up1+=1
                    if x[x_label]==emotion_all[j] and x[x_label-1]!=emotion_all[i]:
                        up2+=1
                    if x[x_label]!=emotion_all[j] and x[x_label-1]==emotion_all[i]:
                        up3+=1
                    if x[x_label]!=emotion_all[j] and x[x_label-1]!=emotion_all[i]:
                        up4+=1
            p1[i][j]=up1*1.0/down1
            # print("%r->%r : %r"%(i,j,p1[i][j]))
            p2[i][j]=up2*1.0/down1
            p3[i][j]=up3*1.0/down0
            p4[i][j]=up4*1.0/down0
    return p1,p2,p3,p4

#计算微博整体与句子之间的四种情绪转移概率
def get_transfer_p_4_weibo():
    label,emotion_sentence_num,sum1,weibo_label=get_label()
    emotion_all=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
    ############句子间的情绪转移概率
    #a->b
    p1=[[0 for j in range(8)] for i in range(8)]
    #非a -> b
    p2=[[0 for j in range(8)] for i in range(8)]
    #a -> 非b
    p3=[[0 for j in range(8)] for i in range(8)]
    #非a -> 非b
    p4=[[0 for j in range(8)] for i in range(8)]
    for i in range(0,8):
        for j in range(0,8):
            down1=emotion_sentence_num[j]
            down0=sum1-emotion_sentence_num[j]
            #微博整体与句子之间
            up1=0
            up2=0
            up3=0
            up4=0
            index=0
            for x in label:
                for x_label in range(0,len(x)):
                    if x[x_label]==emotion_all[j] and weibo_label[index]==emotion_all[i]:
                        up1+=1
                    if x[x_label]==emotion_all[j] and weibo_label[index]!=emotion_all[i]:
                        up2+=1
                    if x[x_label]!=emotion_all[j] and weibo_label[index]==emotion_all[i]:
                        up3+=1
                    if x[x_label]!=emotion_all[j] and weibo_label[index]!=emotion_all[i]:
                        up4+=1
                index+=1
            p1[i][j]=up1*1.0/down1
            # print("%r->%r : %r"%(i,j,p1[i][j]))
            p2[i][j]=up2*1.0/down1
            p3[i][j]=up3*1.0/down0
            p4[i][j]=up4*1.0/down0
    return p1,p2,p3,p4



if __name__ == '__main__':
    #句子间的四种情绪转移概率
    # get_transfer_p_4()
    #整体与句子之间的四种情绪转移概率
    get_transfer_p_4_weibo()
