from root.final.exp.mlp import create_mlp_experiment

for b in [64, 32]:
    create_mlp_experiment('prueba_deep_B' + str(b) + '_', 'all', batch_size= b, epochs= 300)
for b in [16, 8]:
    create_mlp_experiment('prueba_deep_B' + str(b) + '_', 'all', batch_size= b, epochs= 200)
for b in [4, 2]:
    create_mlp_experiment('prueba_deep_B' + str(b) + '_', 'all', batch_size= b, epochs= 100)
    