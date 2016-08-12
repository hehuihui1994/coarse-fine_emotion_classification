# -*- coding: utf-8 -*-

"""

结合上下文和篇章特征的多标签情绪分类算法

"""
import all_emotion_transfer
import mlKNN
import pytc
import  xml.dom.minidom
import evaluation_for_MLL

#提取训练集合标签，输出到corpus/sentence_train_label.txt
def get_sentence_train_label(file_out):
    fw=open(file_out,'w')
    #打开xml文档
    dom = xml.dom.minidom.parse('corpus/train.xml')
    #得到文档元素对象
    root = dom.documentElement  
    #获得标签属性值
    itemlist = root.getElementsByTagName('weibo')
    for item in itemlist:
        stringPre = item.getAttribute("id")+" "
        cc=item.getElementsByTagName('sentence')
        for i in range(len(cc)):
            stringFinal=stringPre+cc[i].getAttribute("id")+" "+cc[i].getAttribute("emotion_tag")+" "
            if cc[i].getAttribute("emotion_tag")=='N':
                stringFinal += "none none"
            else:
                stringFinal += cc[i].getAttribute("emotion-1-type")+" "+cc[i].getAttribute("emotion-2-type")
            print>>fw,stringFinal

#处理for_train_x,变成mxn维度的数组
def process_for_train_x_or_test_x(for_train_x,len_term_set):
	row=len(for_train_x)
	col=len_term_set
	train_x=[[0 for j in range(col)] for i in range(row)]
	for i in range(0,row):
		for j in range(0,col):
			#train_sample中的特征从1开始，对应于向量中的0
			if for_train_x[i].has_key(j+1):
				train_x[i][j]=for_train_x[i][j+1]
	return train_x

#得到train_vec,test_vec
def get_vec(fname_samp_train,fname_samp_test,len_term_set):
	#for_train_x每一行是词典，train_y是list,存储的是主要情绪标签
	for_train_x,train_y=pytc.load_samps(fname_samp_train, fs_num=0)
	for_test_x,test_y=pytc.load_samps(fname_samp_test, fs_num=0)
	train_x=process_for_train_x_or_test_x(for_train_x,len_term_set)
	test_x=process_for_train_x_or_test_x(for_test_x,len_term_set)
	return train_x,test_x

#得到train_label,test_label,多标签,[1,0,1,0,...]
def get_label(fname_label,emotion_num):
    label = []
    dic_label2int = {'anger':0,'disgust':1,'fear':2,'happiness':3,'like':4,'none':5,'sadness':6,'surprise':7}
    fr_label = open(fname_label,'r')
    for line in fr_label.readlines():
        temp_label = [0 for i in range(emotion_num)]
        lineSet = line.strip().split()
        main_emotion = lineSet[3]
        minor_emotion = lineSet[4]
        temp_label[dic_label2int[main_emotion]] = 1
        temp_label[dic_label2int[minor_emotion]] = 1
        label.append(temp_label)
    return label



if __name__ == '__main__':
    #得到训练集标签
    # file_out="corpus/sentence_train_label.txt"
    # get_sentence_train_label(file_out)
    # #句子间的四种情绪转移概率
    # print("first step --------------")
    # print(u"句子间的四种情绪转移概率")
    # p1,p2,p3,p4=all_emotion_transfer.get_transfer_p_4()
    # #整体与句子之间的四种情绪转移概率
    # print(u"整体与句子之间的四种情绪转移概率")
    # p_weibo1,p_weibo2,p_weibo3,p_weibo4=all_emotion_transfer.get_transfer_p_4_weibo()
    # #利用MLKNN计算句子的初始分类
    # print("sencond step------------")
    print(u"初始分类..")
    #把数据处理成train_vec,test_vec,特征选择方法为卡方统计,存储到temporary文件夹中
    #产生svm标准格式,返回特征的个数
    len_term_set = pytc.demo_m_hhh()  
    print(u"得到tarin_vec,test_vec..")
    fname_samp_train = "temporary/train.sample"
    fname_samp_test = "temporary/test.sample" 
    train_vec,test_vec = get_vec(fname_samp_train,fname_samp_test,len_term_set)
    print(u"得到train_label,test_label..")
    fname_train_label = 'corpus/sentence_train_label.txt'
    fname_test_label = 'corpus/sentence_test_label.txt'
    emotion_num = 8
    train_label = get_label(fname_train_label, emotion_num)
    #测试集真实标签
    test_label = get_label(fname_test_label, emotion_num)
    print(u"mlKNN..")
    #[1,0,1,0..] test为预测的标签
    #outputs为f(xi,y)
    k = 7
    test, outputs = mlKNN.mlknn_demo(train_vec,train_label,test_vec,emotion_num,k)
    print(u"评估结果 ..")
    print("AVG : %r"%(evaluation_for_MLL.Average_Precision(outputs,test)))
    print("OE : %r"%(evaluation_for_MLL.One_error(outputs,test)))
    print("HL : %r"%(evaluation_for_MLL.Hamming_Loss(predict,test)))
    print("RL : %r"%(evaluation_for_MLL.RankingLoss(outputs,test)))

    # #每个句子输出结果是一个向量，对于每个表情，分别计算该句子有无该表情的概率
    # print(u"句子的初始分类")
    # #test_emotion_old为一维向量，R，每条句子输出的是一个表情
    # #test_first[][8],每条句子输出的是一个向量
    # test_emotion_old,test_first=mlKNN.mlKNN(7,0.6,0.4,dic_intention_polarity,dic_kafang,dic_smile)
    # #############
    # fw=open("test_first_result",'w')
    # for result in test_emotion_old:
    #     print>>fw,result
    # ###########
    # print("third step --------------")
    # print(u"开始迭代---")
    # print(u"计算平均精度")