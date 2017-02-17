from sklearn.datasets import load_digits
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print digits.keys()


import matplotlib.pyplot as plt
n_row, n_col = 2,5

def print_digits(images, y, max_n=10):
    fig=plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    i=0
    while i < max_n and i<images.shape[0]:
        p = fig.add_subplot(n_row, n_col, i+1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone, interpolation='nearest')
        p.text(0, -1, str(y[i]))
        i = i + 1


#print_digits(digits.images, digits.target, max_n=10)
#plt.show()

from sklearn.decomposition import PCA
estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(X_digits)

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in xrange(len(colors)):
        px = X_pca[:, 0][y_digits == i]
        py = X_pca[:, 1][y_digits == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

#plot_pca_scatter()
#plt.show()

n_components = n_row * n_col
def print_pca_components(images, n_col, n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(comp.reshape((8,8)), interpolation='nearest')
        plt.text(0, -1, str(i+1) + '-component')
        plt.xticks(())
        plt.yticks(())

print_pca_components(estimator.components_[:n_components], n_col, n_row)
plt.show()