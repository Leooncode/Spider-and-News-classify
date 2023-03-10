import pandas as pd
import jieba 

df = pd.read_csv('./data.csv', encoding='utf-8')
# df['col3'] = '足球'
# df.to_csv('./train_data/足球.csv', sep=',')
# print(df['col3'])

# 列表化处理
if df.col2.values.any() != None:
    content = df.col2.values.tolist()
else:
    content = df.col1.values.tolist()

# 结巴分词
ontent_S = []
content_S = []
for line in content:
    current_segment = jieba.lcut(str(line))
    # if len(current_segment) > 1 and current_segment != '\r\n': #换行符
    content_S.append(current_segment)

# 转换为DataFrame
df_content = pd.DataFrame({'content_S':content_S})

# 使用停用词
stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8') #list
#print(stopwords.head(20))

def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words
    # print (contents_clean)
contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords) ###使用停用词
df_content = pd.DataFrame({'contents_clean':contents_clean}) ##每一列的分词
df_all_words = pd.DataFrame({'all_words':all_words})  ###所有词语

# 关键词提取
import jieba.analyse
index = 0
content_S_str = "".join(content_S[index])
print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))#3输出前五个关键词

# LDA建模
from gensim import corpora, models, similarities
import gensim
#做映射，相当于词袋
dictionary = corpora.Dictionary(contents_clean) ##格式要求：list of list形式，分词好的的整个语料
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]  #语料
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10) #类似Kmeans自己指定K值
print (lda.print_topic(1, topn=5)) ##第一个主题，关键词5个

# for topic in lda.print_topics(num_topics=10, num_words=5):
#     print (topic[1])

df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df['col3']})
print(df_train.tail())


label_mapping = {"财经": 1, "彩票": 2, "股市": 3, "军事": 4, "科技":5, "篮球": 6,"跑步": 7,"赛车": 8,"娱乐": 9,"足球": 0}
df_train['label'] = df_train['label'].map(label_mapping) ##变换label


# 制作训练集测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, train_size=0.7)
words = []
for line_index in range(len(x_train)):  
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print (line_index)
test_words = []
for line_index in range(len(x_test)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index)
# 朴素贝叶斯     
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vec.fit(words)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)
print('朴素贝叶斯准确率：',classifier.score(vec.transform(test_words), y_test))

# SVM    
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(vec.transform(words), y_train)
print('SVM准确率：',classifier.score(vec.transform(test_words), y_test))

# 随机森林
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(vec.transform(words), y_train)
print('随机森林准确率：',classifier.score(vec.transform(test_words), y_test))

# XGBoost
import xgboost as xgb
classifier = xgb.XGBClassifier()
classifier.fit(vec.transform(words), y_train)
print('XGBoost准确率：',classifier.score(vec.transform(test_words), y_test))

# 混淆矩阵可视化展示
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 计算混淆矩阵
cm = confusion_matrix(y_true=y_test, y_pred=classifier.predict(vec.transform(test_words)))

# 绘制热力图
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()



