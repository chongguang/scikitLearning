from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
print type(news.data), type(news.target), type(news.target_names)
print news.target_names

print len(news.data)
print len(news.target)

print news.data[0]
print news.target[0], news.target_names[news.target[0]]

SPLIT_PERC = 0.75
split_size = int(len(news.data)*SPLIT_PERC)
X_train = news.data[:split_size]
X_test = news.data[split_size:]
y_train = news.target[:split_size]
y_test = news.target[split_size:]

