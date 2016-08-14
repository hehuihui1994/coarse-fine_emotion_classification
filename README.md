# coarse-fine_emotion_classification
结合上下文和篇章特征的多标签情绪分类  

------

##简介
  主要针对微博本文的每一个句子进行情绪分类，实现论文[面向微博文本的情绪分析方法研究](http://cdmd.cnki.com.cn/Article/CDMD-10213-1015979455.htm)中的结合上下文和篇章特征的多标签情绪分类方法，使用python编程实现。
  
##方法
  1 首先利用句内特征结合MLKNN 算法构造基分类器，并对微博中的每个句子进行初始情绪分类，特征选择方法使用卡方统计。<br>
  2 在初始情绪分类的基础上，利用句子相邻句子的情绪转移概率以及微博整体与句子之间的情绪转换概率，进行分类修正，获得新的分类结果。在更新   后的句子分类结果的基础上，迭代利用上下文特征和篇章特征进行分类更新，直至分类结果收敛。<br>
  3 在迭代过程中，句子的上下句的情绪使用上一轮迭代过程中该句子上下句的情绪类别。微博整体的情绪则根据上一轮迭代中该微博的所有句子分类结   果，将所有句子中出现最多情绪类别作为该微博整体情绪类别。<br>
  
##文件说明

* all_emotion_transfer.py<br>
  计算相邻句子的情绪转移概率以及微博整体与句子之间的情绪转移概率。

* evaluation_for_MLL.py<br>
  计算评估指标Hamming_Loss、One_error、Coverage、RankingLoss、Average_Precision。

* mlKNN.py<br>
  [mlknn算法](http://blog.csdn.net/hayigeqiu/article/details/51791794)

* performance.py<br>
  performance库,多种常用的评估指标如acc、precision、recall等。

* process.py<br>
  结合上下文和篇章特征的多标签情绪分类方法。

* pytc.py<br>
  pytc库，文本处理工具库。
