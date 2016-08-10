# -*- coding: utf-8 -*-

'''
http://blog.csdn.net/hayigeqiu/article/details/51791794
http://www.codeforge.cn/read/171141
http://scikit-learn.org/stable/modules/model_evaluation.html
test取到该类别值为1，取不到值为0
predict为预测的标签
outputs为f(xi,y)
'''

from sklearn.metrics import hamming_loss

#评估一个样本被错分多少次
def Hamming_Loss(predict,test):
    res = 0
    num_instance = len(test)
    num_class = len(test[0])
    for i in range(len(test)):
        temp=0
        for j in range(len(test[i])):
            temp += (predict[i][j] != test[i][j])
        res += temp
    res = res*1.0/(num_class*num_instance)
    return res

#评估在输出结果中排序第一的标签并不属于实际标签集中的概率
def One_error(outputs,test):
    num_class = len(test[0])
    temp_outputs=[]
    temp_test=[]
    #1去除测试集真实标签全部为1或者全部为0的样例
    #这两种情况下没有排序第一
    for i in range(len(test)):
        sum_temp=0
        for j in range(len(test[i])):
            sum_temp += test[i][j]
        #所有类别都取到、所有类别都取不到
        if (sum_temp!=num_class)&(sum_temp!=0):
            temp_outputs.append(outputs[i])
            temp_test.append(test[i])

    #2接下来都用处理好的temp_outputs,temp_test
    num_class = len(temp_outputs[0])
    num_instance = len(temp_outputs)
    #测试集真实标签的类别集合
    label=[]
    for i in range(num_instance):
        temp_label=[]
        for j in range(len(temp_test[i])):
            if temp_test[i][j] == 1:
                temp_label.append(j)
        label.append(temp_label)

    #3
    OneError = 0
    for i in range(num_instance):
        indicator = 0
        #预测的情绪标签中的最大值以及其下标
        maximum = max(temp_outputs[i])
        for j in range(num_class):
            if temp_outputs[i][j] == maximum:
                if j in label[i]:
                    indicator=1
                    break
        #输出结果中排序第一的标签并不属于实际标签集
        if indicator==0:
            OneError+=1
    OneError=OneError*1.0/num_instance
    return OneError

#评价我们平均还差多远
def Coverage(outputs,test):
    num_class=len(outputs[0])
    num_instance=len(outputs)
    #1测试集真实标签的类别集合
    label=[]
    #测试集真实标签中每个sample的1的总个数
    label_size=[]
    for i in range(num_instance):
        temp_label=[]
        temp_label_size=0
        for j in range(len(test[i])):
            if test[i][j] == 1:
                temp_label.append(j)
                temp_label_size+=1
        label.append(temp_label)
        label_size.append(temp_label_size)
    #2
    cover=0
    for i in range(num_instance):
        #给output[i]排序升序，并且保留排序前的坐标
        tempvalue = sorted(outputs[i])
        index = []
        for k in range(len(tempvalue)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == tempvalue[k] and (j not in index):
                    index.append(j)

        temp_min = num_class + 1
        for m in range(label_size[i]):
            loc=temp_min
            for loc_temp in range(len(index)):
                if label[i][m] == index[loc_temp]:
                    loc = loc_temp
                    break
            if loc < temp_min:
                temp_min = loc
        cover += (num_class - temp_min + 1)
    coverage = (cover*1.0/num_instance) - 1
    return coverage


#不属于相关标签集中的项目被排在了属于相关标签集中项目的概率的平均
def RankingLoss(outputs,test):
    num_class = len(outputs[0])
    #1去除测试集真实标签全部为1或者全部为0的样例
    #这两种情况下没有排序问题
    temp_outputs = []
    temp_test = []
    for i in range(len(test)):
        sum_temp=0
        for j in range(len(test[i])):
            sum_temp += test[i][j]
        #所有类别都取到、所有类别都取不到
        if (sum_temp!=num_class)&(sum_temp!=0):
            temp_outputs.append(outputs[i])
            temp_test.append(test[i])
    #2
    num_class = len(temp_outputs[0])
    num_instance = len(temp_outputs)
    #测试集合实际标签中每个样例i的属于1的下标
    label = []
    #不属于1的下标
    not_label = []
    #测试集真实标签中每个sample的1的总个数
    label_size=[]
    for i in range(num_instance):
        temp_label=[]
        temp_not_label=[]
        temp_label_size=0
        for j in range(len(temp_test[i])):
            if temp_test[i][j] == 1:
                temp_label.append(j)
                temp_label_size+=1
            else:
                temp_not_label.append(j)
        label.append(temp_label)
        not_label.append(temp_not_label)
        label_size.append(temp_label_size)
    #3
    rankloss = 0
    for i in range(num_instance):
        temp = 0
        #第i个测试集样例的1的实际个数
        for m in range(label_size[i]):
            #第i个测试集样例的0的实际个数
            for n in range(num_class - label_size[i] ):
                if temp_outputs[i][label[i][m]] <= temp_outputs[i][not_label[i][n]]:
                    temp += 1
        m+=1
        n+=1
        rankloss += temp*1.0/(m*n)
    rankloss = rankloss/num_instance
    return rankloss

#排序排在相关标签集的标签前面，且属于相关标签集的概率
def Average_Precision(outputs,test):
    num_class = len(outputs[0])
    #1去除测试集真实标签全部为1或者全部为0的样例
    #这两种情况下没有排序问题
    temp_outputs = []
    temp_test = []
    for i in range(len(test)):
        sum_temp=0
        for j in range(len(test[i])):
            sum_temp += test[i][j]
        #所有类别都取到、所有类别都取不到
        if (sum_temp!=num_class)&(sum_temp!=0):
            temp_outputs.append(outputs[i])
            temp_test.append(test[i])
    #2
    num_class = len(temp_outputs[0])
    num_instance = len(temp_outputs)
    #测试集合实际标签中每个样例i的属于1的下标
    label = []
    #不属于1的下标
    not_label = []
    #测试集真实标签中每个sample的1的总个数
    label_size=[]
    for i in range(num_instance):
        temp_label=[]
        temp_not_label=[]
        temp_label_size=0
        for j in range(len(temp_test[i])):
            if temp_test[i][j] == 1:
                temp_label.append(j)
                temp_label_size+=1
            else:
                temp_not_label.append(j)
        label.append(temp_label)
        not_label.append(temp_not_label)
        label_size.append(temp_label_size)
    #3
    aveprec = 0
    for i in range(num_instance):
        #给output[i]排序升序，并且保留排序前的坐标
        tempvalue = sorted(outputs[i])
        index = []
        for k in range(len(tempvalue)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == tempvalue[k] and (j not in index):
                    index.append(j)
        ##
        indicator = {}
        for m in range(label_size[i]):
            loc=-1
            for loc_temp in range(len(index)):
                if label[i][m] == index[loc_temp]:
                    loc = loc_temp
                    break
            indicator[loc] = 1
        ##
        summary = 0
        for m in range(label_size[i]):
            loc=num_class
            for loc_temp in range(len(index)):
                if label[i][m] == index[loc_temp]:
                    loc = loc_temp
                    break
            sencond_up = 0
            for loc_i in range(loc,num_class):
                if indicator.has_key(loc_i):
                    sencond_up += 1
            summary += sencond_up*1.0/(num_class - loc + 1)
        #
        aveprec += summary*1.0/label_size[i]
    Average_Precision = aveprec/num_instance
    return Average_Precision


if __name__ == '__main__':
    # y_pred = [[1,0,1],[0,0,1]]
    # y_true = [[1,0,0],[0,0,1]]
    # test_pred = np.array(y_pred)
    # test_true = np.array(y_true)
    # print("hamming_loss : %r"%(hamming_loss(test_true, test_pred)))
    # print(" my hamming_loss : %r"%(Hamming_Loss1(test_pred,test_true)))
    # outputs=[[0.1,0.2,0.7],[0.1,0.1,0.1],[0.2,0,0.8]]
    # test = [[0,0,1],[1,1,1],[1,0,0]]
    # print("OneError should be %r"%(1.0/2))
    # print("my OneError is %r"%(One_error(outputs,test)))
    # print("RankingLoss should be 0.25")
    # print("my RankingLoss is %r"%(RankingLoss(outputs,test)))
    # outputs=[[0.75,0.5,1],[1,0.2,0.1]]
    # test=[[1,0,0],[0,0,1]]
    # print("Coverage should be 2.5")
    # print("my Coverage is %r "%(Coverage(outputs,test)))
    print("my Average_Precision is %r"%(Average_Precision(outputs,test)))