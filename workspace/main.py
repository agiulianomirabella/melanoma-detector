from root.final.detector.detector import final_detector_mlp

n_exp = 2
for csv_number in [0, 1, 2, 3, 4]:
    for b in [128, 64, 32]:
        final_detector_mlp('exp' + str(n_exp) + '_data' + str(csv_number) + '_B' + str(b), 0, batch_size=b, epochs=200)
    for b in [16, 8, 4, 2]:
        final_detector_mlp('exp' + str(n_exp) + '_data' + str(csv_number) + '_B' + str(b), 0, batch_size=b, epochs=100)
