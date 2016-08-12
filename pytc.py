# coding: utf-8
''' Functions V4.00
Author: Rui Xia (rxia.cn@gmail.com)
Date: Last updated on 2014-9-5
'''

import os, re, sys, random, math, subprocess
from nltk.stem import WordNetLemmatizer


########### Global Parameters ###########

TOOL_PATH = 'D:\\Toolkits'
NB_LEARN_EXE = TOOL_PATH + '\\openpr-nb_v1.16\\windows\\nb_learn.exe'
NB_CLASSIFY_EXE = TOOL_PATH + '\\openpr-nb_v1.16\\windows\\nb_classify.exe'
SVM_LEARN_EXE = TOOL_PATH + '\\svm_light\\svm_learn.exe'
SVM_CLASSIFY_EXE = TOOL_PATH + '\\svm_light\\svm_classify.exe'
LIBSVM_LEARN_EXE = TOOL_PATH + '\\libsvm-3.18\\windows\\svm-train.exe'
LIBSVM_CLASSIFY_EXE = TOOL_PATH + '\\libsvm-3.18\\windows\\svm-predict.exe'
LIBLINEAR_LEARN_EXE = TOOL_PATH + '\\liblinear-1.94\\windows\\train.exe'
LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + '\\liblinear-1.94\\windows\\predict.exe'

LOG_LIM = 1E-300


########## File Access Fuctions ##########

def gen_nfolds_f2(input_dir, output_dir, nfolds_num, fname_list, samp_tag,
                  random_tag=False):
    '''Generate nfolds, with each fold containing a training fold and test fold
    '''
    for fname in fname_list:
        file_str = open(input_dir + os.sep + fname, 'r').read()
        patn = '<' + samp_tag + '>(.*?)</' + samp_tag + '>'
        doc_str_list = re.findall(patn, file_str, re.S)
        if random_tag == True:
            random.shuffle(doc_str_list)
        doc_num = len(doc_str_list)
        begin_pos = 0
        for fold_id in range(nfolds_num):
            fold_dir = output_dir + os.sep + 'fold' + str(fold_id+1)
            if not os.path.exists(fold_dir):
                os.mkdir(fold_dir)
            pos_range = int(doc_num / nfolds_num)
            if fold_id != nfolds_num - 1:
                end_pos = begin_pos + pos_range
            else:
                end_pos = len(doc_str_list)
            doc_str_list_test = doc_str_list[begin_pos:end_pos]
            doc_str_list_train = doc_str_list[:begin_pos] + \
                doc_str_list[end_pos:]
            train_dir = fold_dir + os.sep + 'train'
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            fout = open(train_dir + os.sep + fname, 'w')
            fout.writelines(['<' + samp_tag + '>' + x + '</' + samp_tag + \
                '>\n' for x in doc_str_list_train])
            fout.close()
            test_dir = fold_dir + os.sep + 'test'
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            fout = open(test_dir + os.sep + fname, 'w')
            fout.writelines(['<' + samp_tag + '>' + x + '</' + samp_tag + \
                '>\n' for x in doc_str_list_test])
            fout.close()
            begin_pos = end_pos

def split_text_f2(input_dir, output_dir, split_map, fname_list, samp_tag,
                  random_tag=False):
    '''Split the dataset according to split map
    split_map -- such as {'train': 0.8, 'test': 0.2}
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fname_list = sorted([x for x in os.listdir(input_dir) if \
        os.path.isfile(input_dir + os.sep + x)])
    for fname in fname_list:
        file_str = open(input_dir + os.sep + fname, 'r').read()
        patn = '<' + samp_tag + '>(.*?)</' + samp_tag + '>'
        doc_str_list = re.findall(patn, file_str, re.S)
        if random_tag == True:
            random.shuffle(doc_str_list)
        doc_num = len(doc_str_list)
        split_id = 0
        begin_pos = 0
        for fold in split_map:
            if not os.path.exists(output_dir + os.sep + fold):
                os.mkdir(output_dir + os.sep + fold)
            fold_range = int(doc_num * split_map[fold])
            if split_id != len(split_map) - 1:
                end_pos = begin_pos + fold_range
            else:
                end_pos = len(doc_str_list)
            doc_str_list_fold = doc_str_list[begin_pos:end_pos]
            fout = open(output_dir + os.sep + fold + os.sep + fname, 'w')
            fout.writelines(['<' + samp_tag + '>' + x + '</' + samp_tag + \
                '>\n' for x in doc_str_list_fold])
            fout.close()
            begin_pos = end_pos
            split_id += 1

def read_text_f1(parent_dir):
    '''read text format 1: one doc one file, one class one dir
    '''
    dir_list = os.listdir(parent_dir)
    doc_str_list = []
    doc_class_list = []
    for each_dir in dir_list:
        fname_list = [(parent_dir + os.sep + each_dir + os.sep + x) for x in \
            os.listdir(parent_dir + os.sep + each_dir) if \
            os.path.isfile(parent_dir + os.sep + each_dir + os.sep + x)]
        doc_str_list_one_class = []
        for fname in fname_list:
            doc_str = open(fname, 'r').read()
            doc_str_list_one_class.append(doc_str)
        doc_str_list.extend(doc_str_list_one_class)
        doc_class_list.extend([each_dir] * len(doc_str_list_one_class))
    return doc_str_list, doc_class_list

def read_file_f2(fname, sample_tag):
    all_str = open(fname, 'r').read()
    patn = '<' + sample_tag + '>(.*?)</' + sample_tag + '>'
    doc_str_list = re.findall(patn, all_str, re.S)
    return doc_str_list

def read_text_f2(fname_list, samp_tag):
    '''text format 2: one class one file, docs are sperated by samp_tag
    '''
    doc_class_list = []
    doc_str_list = []
    for fname in fname_list: # for fname in sorted(fname_list):
        # print 'Reading', fname
        doc_str = open(fname, 'r').read()
        patn = '<' + samp_tag + '>(.*?)</' + samp_tag + '>'
        str_list_one_class = re.findall(patn, doc_str, re.S)
        class_label = os.path.basename(fname)
        doc_str_list.extend(str_list_one_class)
        doc_class_list.extend([class_label] * len(str_list_one_class))
    doc_str_list = [x.strip() for x in doc_str_list]
    return doc_str_list, doc_class_list

def save_text_f2(save_dir, samp_tag, doc_str_list, doc_class_list):
    '''text format 2: one class one file, docs are sperated by samp_tag
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    class_set = sorted(list(set(doc_class_list)))
    doc_str_class = [''] * len(class_set)
    for k in range(len(doc_class_list)):
        class_id = class_set.index(doc_class_list[k])
        #doc_str = ' '.join(doc_terms_list[k])
        doc_str = doc_str_list[k]
        doc_str_class[class_id] += ('<' + samp_tag + '>\n' + doc_str + \
            '</' + samp_tag + '>\n')
    for class_id in range(len(class_set)):
        class_label = class_set[class_id]
        fobj = open(save_dir + os.sep + class_label, 'w')
        fobj.write(doc_str_class[class_id])
        fobj.close()

def read_text_f3(fname):
    '''text format 3: all in one file, one doc one line
    '''
    doc_str_list = []
    doc_class_list = []
    fin = open(fname, 'r')
    for str_line in fin.readlines():
        doc_class = str_line.split('\t')[0]
        doc_str = str_line.split('\t')[1]
        doc_str_list.append(doc_str)
        doc_class_list.append(doc_class)
    return doc_str_list, doc_class_list

def save_text_f3(doc_str_list, doc_class_list, fname):
    '''text format 3: all in one file, one doc one line
    '''
    fout = open(fname, 'w')
    for k in range(len(doc_class_list)):
        class_label = doc_class_list[k]
        raw_str = doc_str_list[k]
        line_str = class_label + '\t' + ' '.join(raw_str.split()) + '\n'
        fout.write(line_str)
    fout.close()


########## Feature Extraction Fuctions ##########

def get_doc_terms_list(doc_str_list):
    return [x.split() for x in doc_str_list]

def get_doc_unis_list(str_list):
    unis_list = [x.strip().split() for x in str_list]
    return unis_list

def get_doc_bis_list(doc_str_list):
    unis_list = [x.split() for x in doc_str_list]
    doc_bis_list = []
    for k in range(len(doc_str_list)):
        unis = unis_list[k]
        if len(unis) <= 1:
            doc_bis_list.append([])
            continue
        unis_shift = unis[1:] + [unis[0]]
        bis = [unis[j] + '<w-w>' + unis_shift[j] for j in \
            range(len(unis))][0:-1]
        doc_bis_list.append(bis)
    return doc_bis_list

def get_joint_sets(doc_terms_list1, doc_terms_list2):
    joint_list = []
    for k in range(len(doc_terms_list1)):
        doc_terms1 = doc_terms_list1[k]
        doc_terms2 = doc_terms_list2[k]
        joint_list.append(doc_terms1 + doc_terms2)
    return joint_list

def remove_stop_words(stopwords_file, term_set):
    stopwords = [x.strip() for x in open(stopwords_file).readlines()]
    term_set_sw = set(term_set)
    for term in term_set:
        if term in stopwords:
            term_set_sw.remove(term)
    return list(term_set_sw)

def word_lemma(doc_unis_list):
    wnl = WordNetLemmatizer()
    doc_stems_list = []
    for doc_unis in doc_unis_list:
        doc_stems = []
        for uni in doc_unis:
            stem_uni = wnl.lemmatize(uni)
            doc_stems.append(stem_uni)
        doc_stems_list.append(doc_stems)
    return doc_stems_list


########## Text Statistic Fuctions ##########

def get_class_set(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    return class_set

def save_class_set(class_set, fname):
    open(fname, 'w').writelines([x + '\n' for x in class_set])

def load_class_set(fname):
    class_set = [x.strip() for x in open(fname, 'r').readlines()]
    return class_set

def get_term_set(doc_terms_list):
    term_set = set()
    for doc_terms in doc_terms_list:
        term_set.update(doc_terms)
    return sorted(list(term_set))

def save_term_set(term_set, fname):
    open(fname, 'w').writelines([x + '\n' for x in term_set])

def load_term_set(fname):
    term_set = [x.strip() for x in open(fname, 'r').readlines()]
    return term_set

def stat_df_term(term_set, doc_terms_list):
    '''
    df_term is a dict
    '''
    df_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
#        cand_terms = set(term_set) & set(doc_terms) # much more cost!!!
        for term in set(doc_terms):
            if df_term.has_key(term):
                df_term[term] += 1
    return df_term

def stat_tf_term(term_set, doc_terms_list):
    '''
    tf_term is a dict
    '''
    tf_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            if tf_term.has_key(term):
                tf_term[term] += 1
    return tf_term

def stat_df_class(class_set, doc_class_list):
    '''
    df_class is a list
    '''
    df_class = [doc_class_list.count(x) for x in class_set]
    return df_class

def save_df_class(df_class, fname):
    open(fname, 'w').write(' '.join([str(x) for x in df_class]))

def load_df_class(fname):
    df_class = [int(x) for x in open(fname, 'r').read().split()]
    return df_class

def stat_df_term_class(term_set, class_set, doc_terms_list, doc_class_list):
    '''
    df_term_class is a dict-list

    '''
    class_id_dict = dict(zip(class_set, range(len(class_set))))
    df_term_class = {}
    for term in term_set:
        df_term_class[term] = [0]*len(class_set)
    for k in range(len(doc_class_list)):
        class_label = doc_class_list[k]
        class_id = class_id_dict[class_label]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            if df_term_class.has_key(term):
                df_term_class[term][class_id] += 1
    return df_term_class

def save_df_term_class(df_term_class, fname):
    open(fname, 'w').writelines([term + ' ' + ' '.join([str(y) for y in \
        df_term_class[term]])+'\n' for term in sorted(df_term_class.keys())])

def load_df_term_class(fname):
    df_term_class = {}
    for line in open(fname, 'r'):
        term = line.strip().split()[0]
        df_value = [int(x) for x in line.strip().split()[1:]]
        df_term_class[term] = df_value
    return df_term_class

def stat_idf_term(doc_num, df_term):
    '''
    idf_term is a dict
    '''
    idf_term = {}.fromkeys(df_term.keys())
    for term in idf_term:
        idf_term[term] = math.log(float(doc_num/df_term[term]))
    return idf_term

def cal_kld(p_tf_term, q_tf_term, term_set):
    p_sum = sum([float(p_tf_term[t]) for t in term_set])
    q_sum = sum([float(q_tf_term[t]) for t in term_set])
    kld = 0.0
    #print p_tf_term
    for term in term_set:
        p_t = p_tf_term[term]/p_sum
        q_t = q_tf_term[term]/q_sum
        if p_t <= LOG_LIM:
            p_t = LOG_LIM
        if q_t <= LOG_LIM:
            q_t = LOG_LIM
        kld += p_t*(math.log(p_t)-math.log(q_t))
    return kld


########## Feature Selection Functions ##########

def feature_selection_df(df_term, thrd):
    term_set_df = []
    for term in sorted(df_term.keys()):
        if df_term[term] >= thrd:
            term_set_df.append(term)
    return term_set_df

def supervised_feature_selection(df_class, df_term_class, fs_method='IG',
                                 fs_num=0, fs_class=-1):
    if fs_method == 'MI':
        term_set_fs, term_score_dict = feature_selection_mi(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'IG':
        term_set_fs, term_score_dict = feature_selection_ig(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'CHI':
        term_set_fs, term_score_dict = feature_selection_chi(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'WLLR':
        term_set_fs, term_score_dict = feature_selection_wllr(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'LLR':
        term_set_fs, term_score_dict = feature_selection_llr(df_class, \
            df_term_class, fs_num, fs_class)
    return term_set_fs, term_score_dict

def feature_selection_mi(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
            p_c = float(cap_a+cap_c) / cap_n
            score = math.log(p_c_t / p_c)
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_ig(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            p_c = float(cap_a + cap_c) / cap_n
            p_t = float(cap_a + cap_b) / cap_n
            p_nt = 1 - p_t
            p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
            p_c_nt = (cap_c + 1.0) / (cap_c + cap_d + class_set_size)
            score = - p_c * math.log(p_c) + p_t * p_c_t * math.log(p_c_t) + \
                p_nt * p_c_nt * math.log(p_c_nt)
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_chi(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            cap_nu = float(cap_a * cap_d - cap_c * cap_b)
            cap_x1 = cap_nu / ((cap_a + cap_c) * (cap_b + cap_d))
            cap_x2 = cap_nu / ((cap_a+cap_b) * (cap_c+cap_d))
            score = cap_nu * cap_x1 * cap_x2
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_wllr(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
#    	doc_set_size = len(df_class)
        cap_n = sum(df_class)
        term_set_size = len(df_term_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            p_t_c = (cap_a + 1E-6) / (cap_a + cap_c + 1E-6*term_set_size)
            p_t_not_c = (cap_b + 1E-6)/(cap_b + cap_d + 1E-6*term_set_size)
            score = p_t_c * math.log(p_t_c / p_t_not_c)
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_llr(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            p_c_t = (cap_a + 1.0)/(cap_a + cap_b + class_set_size)
            p_nc_t = 1 - p_c_t
            p_c = float(cap_a + cap_c)/ cap_n
            p_nc = 1 - p_c
            score = math.log(p_c_t * p_nc / (p_nc_t * p_c))
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def save_term_score(term_score_dict, fname):
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    fout = open(fname, 'w')
    for term_score in term_score_list:
        fout.write(term_score[0] + '\t' + str(term_score[1]) + '\n')
    fout.close()

def load_term_score(fname):
    term_score_dict = {}
    for line in fname:
        term_score = line.strip().split('\t')
        term_score_dict[term_score[0]] = term_score[1]
    return term_score_dict


########## Sample Building ##########

def build_samps(term_dict, class_dict, doc_terms_list, doc_class_list,
                term_weight, idf_term=None):
    '''Building samples with sparse format
    term_dict -- term1: 1; term2:2; term3:3, ...
    class_dict -- negative:1; postive:2; unlabel:0
    '''
    #文本表示为向量
    samp_dict_list = []
    #类别
    samp_class_list = []
    for k in range(len(doc_class_list)):
        doc_class = doc_class_list[k]
        samp_class = class_dict[doc_class]
        samp_class_list.append(samp_class)
        doc_terms = doc_terms_list[k]
        samp_dict = {}
        for term in doc_terms:
            if term_dict.has_key(term):
                term_id = term_dict[term]
                if term_weight == 'BOOL':
                    samp_dict[term_id] = 1
                elif term_weight == 'TF':
                    if samp_dict.has_key(term_id):
                        samp_dict[term_id] += 1
                    else:
                        samp_dict[term_id] = 1
                elif term_weight == 'TFIDF':
                    if samp_dict.has_key(term_id):
                        samp_dict[term_id] += idf_term[term]
                    else:
                        samp_dict[term_id] = idf_term[term]
        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

def samp_length_norm(samp_dict_list):
    for samp_dict in samp_dict_list:
        doc_len = 0.0
        for i in samp_dict:
            doc_len += samp_dict[i]
        for j in samp_dict:
            samp_dict[j] /= doc_len

def save_samps(samp_dict_list, samp_class_list, fname, feat_num=0):
    length = len(samp_class_list)
    fout = open(fname, 'w')
    for k in range(length):
        samp_dict = samp_dict_list[k]
        samp_class = samp_class_list[k]
        fout.write(str(samp_class) + '\t')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()

def save_samps_unlabel(samp_dict_list, fname, feat_num=0):
    length = len(samp_dict_list)
    fout = open(fname, 'w')
    for k in range(length):
        samp_dict = samp_dict_list[k]
        #fout.write('0\t')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()

def load_samps(fname, fs_num=0):
    fsample = open(fname, 'r')
    samp_class_list = []
    samp_dict_list = []
    for strline in fsample:
        samp_class_list.append(strline.strip().split()[0])
        if fs_num > 0:
            samp_dict = dict([[int(x.split(':')[0]), float(x.split(':')[1])] \
                for x in strline.strip().split()[1:] if int(x.split(':')[0]) \
                < fs_num])
        else:
            samp_dict = dict([[int(x.split(':')[0]), float(x.split(':')[1])] \
                for x in strline.strip().split()[1:]])
        samp_dict_list.append(samp_dict)
    fsample.close()
    return samp_dict_list, samp_class_list


########## Classification Functions ##########

def nb_exe(fname_samp_train, fname_samp_test, fname_model, fname_output,
           learn_opt='', classify_opt='-f 2'):
    print '\nNB executive classifing...'
    pop = subprocess.Popen(NB_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model)
    pop.wait()
    pop = subprocess.Popen(NB_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_output)
    pop.wait()
    samp_class_list_test = [x.split()[0] for x in \
        open(fname_samp_test).readlines()]
    samp_class_list_nb = [x.split()[0] for x in \
        open(fname_output).readlines()]
#    neg_scores = sorted([float(x.split()[1]) for x in \
#        open(fname_output).readlines()])
#    pos_scores = sorted([float(x.split()[2]) for x in \
#        open(fname_output).readlines()])
#    print 'NEG\n', neg_scores
#    print 'POS\n', pos_scores
    acc = calc_acc(samp_class_list_nb, samp_class_list_test)
    return acc

def svm_light_exe(fname_samp_train, fname_samp_test, fname_model, fname_output,
                  learn_opt='', classify_opt=''):
    print '\nSVM_light executive classifing...'
    pop = subprocess.Popen(SVM_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model)
    pop.wait()
    pop = subprocess.Popen(SVM_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_output)
    pop.wait()
    samp_class_list_test = [x.split()[0] for x in \
        open(fname_samp_test).readlines()]
    samp_class_list_pred = []
    for line in open(fname_output):
        score = float(line.strip())
        if score < 0:
            pred_class = '-1'
        else:
            pred_class = '1' #pred_class = '+1'
        samp_class_list_pred.append(pred_class)
    acc = calc_acc(samp_class_list_pred, samp_class_list_test)
    print 'Accuracy:', acc
    return acc

def libsvm_exe(fname_samp_train, fname_samp_test, fname_model, fname_output,
               learn_opt='-t 0 -c 1 -b 1', classify_opt='-b 1'):
    print '\nLibSVM executive classifing...'
    pop = subprocess.Popen(LIBSVM_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model)
    pop.wait()
    pop = subprocess.Popen(LIBSVM_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_output)
    pop.wait()
    samp_class_list_test = [x.split()[0] for x in \
        open(fname_samp_test).readlines()]
    samp_class_list_svm = [x.split()[0] for x in \
        open(fname_output).readlines()[1:]]
    acc = calc_acc(samp_class_list_svm, samp_class_list_test)
    return acc

def liblinear_exe(fname_samp_train, fname_samp_test, fname_model, fname_output,
                  learn_opt='-s 7 -c 1', classify_opt='-b 1'):
    print '\nLiblinear executive classifing...'
    pop = subprocess.Popen(LIBLINEAR_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model)
    pop.wait()
    pop = subprocess.Popen(LIBLINEAR_CLASSIFY_EXE + ' ' + classify_opt + ' ' \
        + fname_samp_test + ' ' + fname_model + ' ' + fname_output)
    pop.wait()
    samp_class_list_test = [x.split()[0] for x in \
        open(fname_samp_test).readlines()]
    samp_class_list_svm = [x.split()[0] for x in \
        open(fname_output).readlines()[1:]]
    acc = calc_acc(samp_class_list_svm, samp_class_list_test)
    return acc

def load_predictions_nb(prd_fname):
    samp_class_list = []
    samp_prb_list = []
    for line in open(prd_fname):
        samp_class_list.append(int(line.split()[0]))
        samp_prb = dict()        
        for term in line.split()[1:]:
            samp_prb[term.split(':')[0]] = float(term.split(':')[1])
        samp_prb_list.append(samp_prb)
    return samp_class_list, samp_prb_list

def load_predictions_liblinear(prd_fname):
    samp_class_list = []
    samp_prb_list = []
    class_id = [int(x) for x in open(prd_fname).readlines()[0].split()[1:]]
    for line in open(prd_fname).readlines()[1:]:
        samp_class_list.append(int(line.split()[0]))
        samp_prb = dict(zip(class_id, [float(line.split()[1]), float(line.split()[2])]))
        samp_prb_list.append(samp_prb)
    return samp_class_list, samp_prb_list

def load_predictions_libsvm(prd_fname):
    samp_class_list = []
    samp_prb_list = []
    class_id = [int(x) for x in open(prd_fname).readlines()[0].split()]
    for line in open(prd_fname).readlines()[1:]:
        samp_class_list.append(int(line.split()[0]))
        samp_prb = dict(zip(class_id, [float(line.split()[1]), float(line.split()[2])]))
        samp_prb_list.append(samp_prb)
    return samp_class_list, samp_prb_list

def load_predictions_svmlight(prd_fname):
    samp_class_list = []
    samp_score_list = []    
    for line in open(prd_fname):
        score = float(line)
        if score < 0:
            samp_class_list.append(-1)
        else:
            samp_class_list.append(1)
        samp_score_list.append(score)    
    return samp_class_list, samp_score_list

def save_predictions_nb(samp_class_list, samp_prb_list, pred_fname):
    pred_file = open(pred_fname, 'w')
    for k in range(len(samp_class_list)):
        samp_class = samp_class_list[k]
        samp_prb = samp_prb_list[k]
        prb_str = ''
        for key in samp_prb:
            prb_str += str(key) + ':' + str(samp_prb[key]) + ' '
        pred_file.write(str(samp_class) + '\t' + prb_str + '\n')
    pred_file.close()


########## Evalutation Functions ##########

def calc_acc(labellist1, labellist2):
    if len(labellist1) != len(labellist2):
        print 'Error: different lenghts!'
        return 0
    else:
        samelist = [int(x == y) for (x, y) in zip(labellist1, labellist2)]
        acc = float((samelist.count(1)))/len(samelist)
        return acc

def calc_recall(label_list_test, label_list_pred):
    true_pos = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (1, 1)])
    false_pos = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (0, 1)])
    true_neg = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (1, 0)])
    false_neg = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (0, 0)])
    recall_pos = true_pos/(true_pos+false_neg)
    recall_neg = true_neg/(true_neg+false_pos)
    return (recall_pos, recall_neg)

def calc_fscore(label_list_test, label_list_pred):
    true_pos = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (1, 1)])
    false_pos = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (0, 1)])
    true_neg = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (1, 0)])
    false_neg = sum([1 for (x, y) in zip(label_list_test, label_list_pred) \
        if (x, y) == (0, 0)])
    precision_pos = true_pos/(true_pos+false_pos)
    precision_neg = true_neg/(true_neg+false_neg)
    recall_pos = true_pos/(true_pos+false_neg)
    recall_neg = true_neg/(true_neg+false_pos)
    fscore_pos = 2*precision_pos*recall_pos/(precision_pos+recall_pos)
    fscore_neg = 2*precision_neg*recall_neg/(precision_neg+recall_neg)
    return (fscore_pos, fscore_neg)


def demo():
    '''A demo for sentiment classification
    '''
#    print 'Spliting datasets...'
#    input_dir = 'dataset\movie2.0_token_f2'
#    split_map = {'train': 0.8, 'test': 0.2}
#    split_text_f2(input_dir, input_dir, split_map, ['neg','pos'], 'text', True)

    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    output_dir = sys.argv[3]
#    train_dir = r'data\kitchen_token_5folds\fold1\train'
#    test_dir = r'data\kitchen_token_5folds\fold1\test'
#    output_dir = r'data\kitchen_token_5folds\fold1\result'
    #fname_list = ['negative', 'positive']
    fname_list = ['negative', 'positive', 'neutral']
    samp_tag = 'review_text'
#    fname_list = ['neg', 'pos']
#    samp_tag = 'text'
    term_weight = 'TFIDF'
    fs_method = 'CHI'
    fs_num = 50000

    fname_class_set = output_dir + os.sep + 'class.set'
    fname_term_set = output_dir + os.sep + 'term.set'
    fname_df_term_class = output_dir + os.sep + 'df.term.class'
    fname_df_class = output_dir + os.sep + 'df.class'
    fname_term_set_fs = output_dir + os.sep + 'term.set.fs'
    fname_samps_test = output_dir + os.sep + 'test.samp'
    fname_samps_train = output_dir + os.sep + 'train.samp'
    fname_model_nb = output_dir + os.sep + 'nb.model'
    fname_output_nb = output_dir + os.sep + 'nb.result'

    print 'Reading text...'
    doc_str_list_train, doc_class_list_train = read_text_f2([train_dir + \
        os.sep + x for x in fname_list], samp_tag)
    doc_str_list_test, doc_class_list_test = read_text_f2([test_dir + \
        os.sep + x for x in fname_list], samp_tag)
    doc_terms_list_train = get_doc_terms_list(doc_str_list_train)
    doc_terms_list_test = get_doc_terms_list(doc_str_list_test)
    class_set = get_class_set(doc_class_list_train)
    term_set = get_term_set(doc_terms_list_train)
    save_class_set(class_set, fname_class_set)
    save_term_set(term_set, fname_term_set)

    print 'Filtering features (DF>=4)...'
    term_df = stat_df_term(term_set, doc_terms_list_train)
    term_set_df = feature_selection_df(term_df, 4)
    term_set = term_set_df

    print 'Selecting features...'
    df_class = stat_df_class(class_set, doc_class_list_train)
    df_term_class = stat_df_term_class(term_set, class_set, \
        doc_terms_list_train, doc_class_list_train)
    save_df_class(df_class, fname_df_class)
    save_df_term_class(df_term_class, fname_df_term_class)

    term_set_fs, term_score_dict = supervised_feature_selection(df_class, \
        df_term_class, fs_method, fs_num)
    save_term_score(term_score_dict, fname_term_set_fs)
    term_set = term_set_fs

    print 'Building samples...'
    term_dict = dict(zip(term_set, range(1, len(term_set)+1)))
    class_dict = dict(zip(class_set, range(1, 1+len(class_set))))
    samp_list_train, class_list_train = build_samps(term_dict, class_dict, \
        doc_terms_list_train, doc_class_list_train, term_weight)
    samp_list_test, class_list_test = build_samps(term_dict, class_dict, \
        doc_terms_list_test, doc_class_list_test, term_weight)
    save_samps(samp_list_train, class_list_train, fname_samps_train)
    save_samps(samp_list_test, class_list_test, fname_samps_test)

    print 'Naive Bayes classification...'
    acc_nb = nb_exe(fname_samps_train, fname_samps_test, fname_model_nb, \
        fname_output_nb)
    print '\nFianl accuracy:', acc_nb

#文本和情绪类别在两个文件中
def read_text_f_hhh(file_text,file_class):
    #文本
    doc_str_list = []
    f_text = open(file_text,'r')
    for line in f_text.readlines():
        doc_str_list.append(line.strip())
    #主要情绪类别
    doc_class_list = []
    f_class = open(file_class,'r')
    index = 0
    for line in f_class.readlines():
        lineSet = line.strip().split()
        doc_class_list.append(lineSet[3])
    return doc_str_list, doc_class_list 


#多类别
def demo_m_hhh():
    print u'读训练集、测试集...'
    #加上主要情绪类别，以便观察
    #训练集
    file_train_text = "corpus/sentence_train_quzao.txt_fenci"
    file_train_class = "corpus/sentence_train_label.txt"
    doc_str_list_train, doc_class_list_train = read_text_f_hhh(file_train_text,file_train_class)
    #测试集
    file_test_text = "corpus/sentence_test_quzao.txt_fenci"
    file_test_class = "corpus/sentence_test_label.txt"
    doc_str_list_test, doc_class_list_test = read_text_f_hhh(file_test_text,file_test_class)
    #处理
    doc_terms_list_train=get_doc_terms_list(doc_str_list_train)
    doc_terms_list_test =get_doc_terms_list(doc_str_list_test)
    class_set = get_class_set(doc_class_list_train)
    #查看class_set
    for i in range(0,len(class_set)):
        string=str(i)+" "+class_set[i]
        print string
    term_set = get_term_set(doc_terms_list_train)

    print u'特征过滤 (DF>=4)...'
    term_df = stat_df_term(term_set, doc_terms_list_train)
    term_set_df = feature_selection_df(term_df, 4)
    term_set = term_set_df

    print u'特征选择 卡方...'
    df_class = stat_df_class(class_set, doc_class_list_train)
    df_term_class = stat_df_term_class(term_set, class_set, \
        doc_terms_list_train, doc_class_list_train)
    
    fs_method='CHI'
    fs_num=0
    term_set_fs, term_score_dict = supervised_feature_selection(df_class, \
        df_term_class, fs_method, fs_num)
    term_set = term_set_fs
    len_term_set = len(term_set)
    print(u"特征个数：%r"%(len_term_set))

    # print u'文本表示为向量 BOOL...'
    # term_dict = dict(zip(term_set, range(1, len(term_set)+1)))
    # class_dict = dict(zip(class_set, range(1, 1+len(class_set))))
    # #samp_list_train
    # term_weight = 'BOOL'
    # samp_list_train, class_list_train = build_samps(term_dict, class_dict, \
    #     doc_terms_list_train, doc_class_list_train, term_weight)
    # samp_list_test, class_list_test = build_samps(term_dict, class_dict, \
    #     doc_terms_list_test, doc_class_list_test, term_weight)

    # #将向量输出
    # save_samps(samp_list_train, class_list_train, 'temporary\\train.sample')
    # save_samps(samp_list_test, class_list_test, 'temporary\\test.sample')
    return len_term_set


if __name__ == '__main__':
    demo()   