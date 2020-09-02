from root.utils import * # pylint: disable= unused-wildcard-import

import matplotlib.pyplot as plt

'''
model0 = [0.69, 0.76, 0.74, 0.76, 0.78, 0.75, 0.75]
model1 = [0.78, 0.77, 0.78, 0.77, 0.8, 0.78, 0.79]
model2 = [0.78, 0.79, 0.78, 0.78, 0.78, 0.78, 0.78]
model3 = [0.79, 0.79, 0.79, 0.78, 0.79, 0.78, 0.79]
model4 = [0.78, 0.79, 0.79, 0.79, 0.75, 0.78, 0.77]
model5 = [0.8, 0.78, 0.79, 0.78, 0.77, 0.79, 0.78]
model1t = [0.79, 0.77, 0.78, 0.78, 0.78, 0.78, 0.78]

models = [model0, model1, model2, model3, model4, model5, model1t]
models_names = ['model0', 'model1', 'model2', 'model3', 'model4', 'model5', 'model1t']

for model in models:
    plt.plot(model)

plt.xticks(np.arange(0, len(model0)+1), ['2', '4', '8', '16', '32', '64', '128'])
plt.legend(models_names, loc='lower right')
plt.ylabel('Max validation AUC')
plt.xlabel('Batch size')
plt.title('Models max validation AUC VS. batch size')

plt.show()
'''





model0  = [0.65, 0.69, 0.68, 0.69, 0.71, 0.69, 0.7]
model1  = [0.71, 0.7 , 0.7 , 0.7 , 0.73, 0.71, 0.71]
model2  = [0.7 , 0.71, 0.7 , 0.71, 0.71, 0.71, 0.71]
model3  = [0.7 , 0.71, 0.72, 0.69, 0.7 , 0.7 , 0.71]
model4  = [0.7 , 0.71, 0.71, 0.72, 0.69, 0.71, 0.69]
model5  = [0.71, 0.7 , 0.71, 0.7 , 0.7 , 0.7 , 0.7]
model1t = [0.72, 0.69, 0.71, 0.7 , 0.7, 0.71 , 0.7]

models = [model0, model1, model2, model3, model4, model5, model1t]
models_names = ['model0', 'model1', 'model2', 'model3', 'model4', 'model5', 'model1t']

for model in models:
    plt.plot(model)

plt.xticks(np.arange(0, len(model0)+1), ['2', '4', '8', '16', '32', '64', '128'])
plt.legend(models_names, loc='lower right')
plt.ylabel('Max validation accuracy')
plt.xlabel('Batch size')
plt.title('Models max validation accuracy VS. batch size')

plt.show()
'''
'''




