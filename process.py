# -*- coding: utf-8 -*-

"""

结合上下文和篇章特征的多标签情绪分类算法

"""
import all_emotion_transfer
import mlKNN
import pytc



if __name__ == '__main__':
	# #句子间的四种情绪转移概率
	# print("first step --------------")
	# print("句子间的四种情绪转移概率")
	# p1,p2,p3,p4=all_emotion_transfer.get_transfer_p_4()
	# #整体与句子之间的四种情绪转移概率
	# print("整体与句子之间的四种情绪转移概率")
	# p_weibo1,p_weibo2,p_weibo3,p_weibo4=all_emotion_transfer.get_transfer_p_4_weibo()
	# #利用MLKNN计算句子的初始分类
	# print("sencond step------------")
	# pytc.demo_hhh()

	

	# #每个句子输出结果是一个向量，对于每个表情，分别计算该句子有无该表情的概率
	# print("句子的初始分类")
	# #test_emotion_old为一维向量，R，每条句子输出的是一个表情
	# #test_first[][8],每条句子输出的是一个向量
	# test_emotion_old,test_first=mlKNN.mlKNN(7,0.6,0.4,dic_intention_polarity,dic_kafang,dic_smile)
	# #############
	# fw=open("test_first_result",'w')
	# for result in test_emotion_old:
	# 	print>>fw,result
	# ###########
	# print("third step --------------")
	# print("开始迭代---")
	# print("计算平均精度")



