import pandas
import numpy
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

CLUSTER_NUMBER = 10
RANDOM_STATE = 10


def read_data():
    print('Loading CSV file...')
    csv_data = pandas.read_csv("F:\\Python3\\1DataSet\\sf-crime\\train.csv")
    print(csv_data.shape)
    print(csv_data.head(5))
    print('Generating Leaning Data...')
    train_data = csv_data[['X', 'Y']]
    final_data = numpy.array(train_data)
    print(train_data.shape)
    print(train_data.head(5))
    return final_data


def kmeans_sort(data):
    print("Start K-Means...")
    kmeans = KMeans(n_clusters=CLUSTER_NUMBER, random_state=RANDOM_STATE, n_jobs=-1)
    predicted = kmeans.fit_predict(data)
    print(predicted[:10])
    print("Centers: ")
    print(kmeans.cluster_centers_)
    print("Score: ")
    print(metrics.calinski_harabasz_score(data, predicted))
    return predicted, kmeans.cluster_centers_, metrics.calinski_harabasz_score(data, predicted)


def draw_graph(data, predicted, centers, scrored):
    print('Processing Data...')
    x = [x[0] for x in data]
    y = [x[1] for x in data]
    plt.scatter(x, y, c=predicted, marker='o')
    plt.scatter(centers[:, 0], centers[:, 1], marker='+')

    plt.title('San Francisco Criminal Activities Sorted in K-Means Algorithm\nTotal Score: ' + str(scrored))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.show()


def slim_kmeans():
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2],
                      random_state=9)
    kmeans = KMeans(n_clusters=4, random_state=9)
    y_pred = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
    score = metrics.calinski_harabasz_score(X, y_pred)
    plt.title('Total Score: ' + str(score))
    print(score)
    plt.show()


def iter_kmeans():
    '''
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=1000, n_features=4, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.4], random_state=10)
    '''

    for counter in range(1, 20):
        X = [[i, j] for i in range(20) for j in range(20)]

        kmeans = KMeans(n_clusters=4, max_iter=counter, n_init=1, random_state=10, init='random',
                        precompute_distances=False)
        pred = kmeans.fit_predict(X)
        # plt.scatter(X[:, 0], X[:, 1], c=pred)
        dx = [i[0] for i in X]
        dy = [i[1] for i in X]
        plt.scatter(dx, dy, c=pred)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
        plt.title('STEP: ' + str(counter) + ' Scored: ' + str(metrics.calinski_harabasz_score(X, pred)))
        print('Scored: ' + str(metrics.calinski_harabasz_score(X, pred)))
        plt.ion()
        plt.show()
        plt.pause(1)


def criminal_kmeans():
    data = read_data()
    prediceted, centers, marks = kmeans_sort(data)
    draw_graph(data, prediceted, centers, marks)


'''
选择模式
criminal_kmeans(): 对旧金山878049条犯罪记录地点进行K-Means聚类，调整CLUSTER_NUMBER = 10（默认）以设置分类簇的数量，调整RANDOM_STATE = 10（默认）以调整随机数生成器种子
slim_kmeans(): 轻量化的数据，快速得结果和plt图
iter_kmeans(): 用于观察每次迭代的变化情况
'''
# criminal_kmeans()
# slim_kmeans()
iter_kmeans()
