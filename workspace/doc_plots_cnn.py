from root.utils import * # pylint: disable= unused-wildcard-import

import matplotlib.pyplot as plt

model0 = [0.69, 0.76, 0.74, 0.76, 0.78, 0.75, 0.75]
model1 = [0.78, 0.77, 0.78, 0.77, 0.8, 0.78, 0.79]
model2 = [0.78, 0.79, 0.78, 0.78, 0.78, 0.78, 0.78]
model3 = [0.79, 0.79, 0.79, 0.78, 0.79, 0.78, 0.79]
model3t = [0.77, 0.77, 0.79, 0.79, 0.77, 0.77, 0.77]
model4 = [0.78, 0.79, 0.79, 0.79, 0.75, 0.78, 0.77]
model5 = [0.8, 0.78, 0.79, 0.78, 0.77, 0.79, 0.78]
model5t = [0.77, 0.78, 0.76, 0.77, 0.78, 0.77, 0.78]

models = [model0, model1, model2, model3, model4, model5, model3t, model5t]
models_names = ['model0', 'model1', 'model2', 'model3', 'model4', 'model5', 'model3t', 'model5t']

for model in models:
    plt.plot(model)

plt.xticks(np.arange(0, len(model0)+1), ['2', '4', '8', '16', '32', '64', '128'])
plt.legend(models_names, loc='lower right')
plt.ylabel('Max validation AUC')
plt.xlabel('Batch size')
plt.title('Models max validation AUC VS. batch size')

plt.show()


