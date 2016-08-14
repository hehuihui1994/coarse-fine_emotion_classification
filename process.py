# -*- coding: utf-8 -*-

"""

结合上下文和篇章特征的多标签情绪分类算法

"""
# import all_emotion_transfer
import mlKNN
import pytc
import  xml.dom.minidom
import evaluation_for_MLL
import all_emotion_transfer
import copy

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
    '''
    none ，none         [0, 0, 0, 0, 0, 1, 0, 0]
    like, none          [0, 0, 0, 0, 1, 0, 0, 0]
    anger,  disgust     [1, 1, 0, 0, 0, 0, 0, 0]

    '''
    fr_label = open(fname_label,'r')
    for line in fr_label.readlines():
        temp_label = [0 for i in range(emotion_num)]
        lineSet = line.strip().split()
        main_emotion = lineSet[3]
        minor_emotion = lineSet[4]
        temp_label[dic_label2int[main_emotion]] = 1
        if minor_emotion == 'none':
            label.append(temp_label)
            continue
        temp_label[dic_label2int[minor_emotion]] = 1
        label.append(temp_label)
    return label

#把句子以微博为单位整理
def get_label_group(train_label, fname_label):
    train_label_group = []
    train_label_temp = []
    fr = open(fname_label, 'r')
    index_current = 1
    train_label_index = 0
    for line in fr.readlines():
        lineSet = line.strip().split()
        weibo_index = lineSet[0]
        if int(weibo_index) != index_current:
            train_label_group.append(train_label_temp)
            train_label_temp = []
            index_current += 1
        train_label_temp.append(train_label[train_label_index])
        train_label_index += 1
    train_label_group.append(train_label_temp)
    return train_label_group

#微博整体的情绪   只有一种情绪类别   [1,0,0,0...]
def get_weibo_label(fname_label, emotion_num):
    weibo_label = []
    dic_label2int = {'anger':0,'disgust':1,'fear':2,'happiness':3,'like':4,'none':5,'sadness':6,'surprise':7}
    fr = open(fname_label, 'r')
    for line in fr.readlines():
        temp_label = [0 for i in range(emotion_num)]
        temp_label[dic_label2int[line.strip()]] = 1
        weibo_label.append(temp_label)
    return weibo_label

#迭代收敛 修正结果 输出预测的标签predict 以及  概率 outputs
def iteration(test_label_group_first, mlknn_p1_group_first, mlknn_p0_group_first, ps, pw, outputs_first, test_label,emotion_num):
    #初始分类结果的AVP
    avp1 = evaluation_for_MLL.Average_Precision(outputs_first, test_label))
    avp0 = avp1 - 10
    while(avp1 - avp0 > 0.0001):
        #重新计算新的 test_label_group_first, outputs_first
        outputs_new = []
        test_label_group_predict_new = []
        #每条微博
        for weibo_i in range(len(test_label_group_first)):
            #[[1,0,1,0,..],[],]
            test_label_weibo_predict = []
            #上一轮结果中句子中出现最多情绪类别作为该微博整体情绪类别
            weibo_predict_label = [0 for j in range(emotion_num)]
            weibo_predict_label_sum = [0 for j in range(emotion_num)]
            for j in range(emotion_num):
                for sentence_j in range(len(test_label_group_first[weibo_i])):
                    weibo_predict_label_sum[j] += test_label_group_first[weibo_i][sentence_j][j]
            #找出最大的情绪类别
            max_digit = -1
            label_index = 0
            for index in range(emotion_num):
                if weibo_predict_label_sum[index] > max_digit:
                    label_index = index
            weibo_predict_label[label_index] = 1
            #每个句子

            for sentence_j in range(len(test_label_group_first[weibo_i])):
                r=[0 for j in range(emotion_num)]
                test_label_sentence_predict = []
                #对于所有的情绪类别
                for j in range(emotion_num):
                    #该情绪类别预测为1的概率
                    temp_p1 = mlknn_p1_group_first[weibo_i][sentence_j][j]
                    #该情绪类别预测为0的概率
                    temp_p0 = mlknn_p0_group_first[weibo_i][sentence_j][j]
                    #加上转移概率，更新temp_p1, temp_p0
                    for ep in range(emotion_num):
                        if sentence_j != 0:
                            pre_sentence_label = test_label_group_first[weibo_i][sentence_j-1][ep]
                            if pre_sentence_label == 1:
                                temp_p1 = temp_p1 * ps[0][ep][j]
                                temp_p0 = temp_p0 * ps[2][ep][j]
                            else:
                                temp_p1 = temp_p1 * ps[1][ep][j]
                                temp_p0 = temp_p0 * ps[3][ep][j]
                        if sentence_j != len(test_label_group_first[weibo_i]):
                            next_sentence_label = test_label_group_first[weibo_i][sentence_j+1][ep]
                            if next_sentence_label == 1:
                                temp_p1 = temp_p1 * ps[0][j][ep]
                                temp_p0 = temp_p0 * ps[1][j][ep]
                            else:
                                temp_p1 = temp_p1 * ps[2][j][ep]
                                temp_p0 = temp_p0 * ps[3][j][ep]
                        #微博整体
                        if weibo_predict_label[ep] == 1:
                            temp_p1 = temp_p1 * pw[0][ep][j]
                            temp_p0 = temp_p0 * pw[2][ep][j]
                        else:
                            temp_p1 = temp_p1 * pw[1][ep][j]
                            temp_p0 = temp_p0 * pw[3][ep][j]
                    #第weibo_i条微博的第sentence_j句子的第j个情绪类别
                    r[j] = temp_p1 *1.0 /(temp_p1 + temp_p0)
                    if r[j] > 0.5:
                        test_label_sentence_predict.append(1)
                    else:
                        test_label_sentence_predict.append(0)
                #以句子为单位，非整合
                outputs_new.append(r)
                test_label_weibo_predict.append(test_label_sentence_predict)
            #以weibo为单位
            test_label_group_predict_new.append(test_label_weibo_predict)
        #AVG  test_label_group_predict_new  outputs_new
        avp0 = avp1
        avp1 = evaluation_for_MLL.Average_Precision(outputs_new, test_label))
        test_label_group_first =  copy.deepcopy(test_label_group_predict_new)
        outputs_first = copy.deepcopy(outputs_new)
    return test_label_group_first, outputs_first


if __name__ == '__main__':
    #把数据处理成train_vec,test_vec,特征选择方法为卡方统计,存储到temporary文件夹中
    #产生svm标准格式,返回特征的个数
    len_term_set = pytc.demo_m_hhh()  
    print(u"得到tarin_vec,test_vec..")
    fname_samp_train = "temporary/train.sample"
    fname_samp_test = "temporary/test.sample" 
    train_vec,test_vec = get_vec(fname_samp_train,fname_samp_test,len_term_set)
    print(u"得到train_label..")
    fname_train_label = 'corpus/sentence_train_label.txt'
    fname_test_label = 'corpus/sentence_test_label.txt'
    emotion_num = 8
    train_label = get_label(fname_train_label, emotion_num)
    #微博整体的情绪   只有一种情绪类别   [1,0,0,0...]
    fname_weibo_train_label = 'corpus/weibo_train_label.txt'
    weibo_train_label = get_weibo_label(fname_weibo_train_label, emotion_num)
    print(u"把句子以微博为单位整理..")
    train_label_group = get_label_group(train_label, fname_train_label)
    print(u"转移概率..")
    #每种情绪的微博数量
    emotion_sentence_num = [0 for i in range(emotion_num)]
    for train_label_temp in train_label:
        for j in range(emotion_num):
            emotion_sentence_num[j] += train_label_temp[j]
    #训练集合样本数
    train_instance_num = len(train_label)
    print(u"计算相邻句子的四种转移概率p1_s,p2_s,p3_s,p4_s..")
    p1_s,p2_s,p3_s,p4_s = all_emotion_transfer.get_transfer_p_4(train_label_group, emotion_sentence_num, train_instance_num, emotion_num)
    ps = [p1_s, p2_s, p3_s, p4_s]
    print(u"计算微博整体与句子之间的四种情绪转移概率p1_w,p2_w,p3_w,p4_w..")
    p1_w,p2_w,p3_w,p4_w = all_emotion_transfer.get_transfer_p_4_weibo(weibo_train_label, train_label_group, emotion_sentence_num, train_instance_num, emotion_num)
    pw = [p1_w,p2_w,p3_w,p4_w]
    print(u"计算初始分类")
    #[1,0,1,0..] test为预测的标签
    #outputs为f(xi,y)
    k = 7
    test_first, outputs_first, mlknn_p1_first, mlknn_p0_first = mlKNN.mlknn_demo(train_vec,train_label,test_vec,emotion_num,k)
    #以微博为单位进行整理
    test_label_group_first = get_label_group(test_first, fname_test_label)
    mlknn_p1_group_first = get_label_group(mlknn_p1_first, fname_test_label)
    mlknn_p0_group_first = get_label_group(mlknn_p0_first, fname_test_label)
    #测试集真实标签
    test_label = get_label(fname_test_label, emotion_num)
    print(u"mlKNN迭代进行分类修正..")
    test_label_group_first, outputs_first = iteration(test_label_group_first, mlknn_p1_group_first, mlknn_p0_group_first, ps, pw, outputs_first, test_label,emotion_num)
    print(u"评估结果 ..")
    outputs = outputs_first
    predict = []
    for weibo_i in range(len(test_label_group_first)):
        for sentence_j in range(len(test_label_group_first[weibo_i])):
            predict.append(test_label_group_first[weibo_i][sentence_j])
    print("AVG : %r"%(evaluation_for_MLL.Average_Precision(outputs,test_label)))
    print("OE : %r"%(evaluation_for_MLL.One_error(outputs,test_label)))
    print("HL : %r"%(evaluation_for_MLL.Hamming_Loss(predict,test_label)))
    print("RL : %r"%(evaluation_for_MLL.RankingLoss(outputs,test_label)))