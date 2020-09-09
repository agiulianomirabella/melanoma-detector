from root.utils import * # pylint: disable= unused-wildcard-import

import matplotlib.pyplot as plt

'''
AUC
model0 = [0.85, 0.85, 0.85, 0.86, 0.86, 0.86, 0.85]
model1 = [0.85, 0.85, 0.86, 0.85, 0.86, 0.86, 0.85]
model2 = [0.85, 0.85, 0.87, 0.86, 0.86, 0.86, 0.86]

models = [model0, model1, model2]
models_names = ['model0', 'model1', 'model2']

for model in models:
    plt.plot(model)

plt.xticks(np.arange(0, len(model0)+1), ['2', '4', '8', '16', '32', '64', '128'])
plt.legend(models_names, loc='lower right')
plt.ylabel('Max validation AUC')
plt.xlabel('Batch size')
plt.title('Models max validation AUC VS. batch size')

plt.show()
'''





'''
accuracy

'''
model0 = [0.80, 0.81, 0.81, 0.81, 0.81, 0.82, 0.82]
model1 = [0.80, 0.80, 0.80, 0.79, 0.82, 0.81, 0.80]
model2 = [0.78, 0.79, 0.82, 0.80, 0.81, 0.81, 0.80]

models = [model0, model1, model2]
models_names = ['model0', 'model1', 'model2']

for model in models:
    plt.plot(model)

plt.xticks(np.arange(0, len(model0)+1), ['2', '4', '8', '16', '32', '64', '128'])
plt.legend(models_names, loc='lower right')
plt.ylabel('Max validation accuracy')
plt.xlabel('Batch size')
plt.title('Models max validation accuracy VS. batch size')

plt.show()




