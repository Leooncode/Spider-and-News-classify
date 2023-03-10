# 导入需要的库
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import nltk
nltk.download('stopwords')

# 定义新闻类别
categories = ['国内', '国际', '军事', '财经', '体育', '娱乐', '科技', '汽车', '健康', '时尚']

# 爬取新闻数据
def get_news_data():
    print('开始爬取新闻数据')
    news_data = []
    for category in categories:
        for page in range(1, 11): # 爬取每个类别前10页新闻
            url = f'https://news.sina.com.cn/{category}/'
            if page > 1:
                url += f'index_{page}.shtml'
            response = requests.get(url)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            for article in soup.find_all('div', class_='clearfix'):
                a_tag = article.find('a')
                if not a_tag:
                    continue
                href = a_tag.get('href')
                if not re.match('^https://news\.sina\.com\.cn/.*?/\d{4}-\d{2}-\d{2}/.*?\.shtml$', href):
                    continue
                title = a_tag.text.strip()
                response = requests.get(href)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.find('div', class_='article').text.strip()
                news_data.append({'title': title, 'content': content, 'category': category})
    print('-----爬取结束-----')
    return news_data

# 数据预处理
def preprocess_data(data):
    print('开始数据预处理')
    stop_words = set(nltk.corpus.stopwords.words('chinese'))
    vectorizer = CountVectorizer(stop_words=stop_words)
    corpus = [article['content'] for article in data]
    X = vectorizer.fit_transform(corpus)
    y = [categories.index(article['category']) for article in data]
    print('数据预处理结束')
    return X, y
# 朴素贝叶斯算法
def train_naive_bayes(X_train, y_train):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf

# 测试模型
def test_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 主函数
def main():
    # 爬取新闻数据
    news_data = get_news_data()

    # 数据预处理
    X, y = preprocess_data(news_data)

    # 训练模型
    clf = train_naive_bayes(X, y)

    # 测试模型
    accuracy = test_model(clf, X, y)
    print(f'模型精确度：{accuracy:.2f}')

if __name__ == '__main__':
    main()
