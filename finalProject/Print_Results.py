import os, csv
import numpy as np


algorithms = os.listdir('Results/CMO')


for Obs_Type in ['CMO', 'RMO']:
    with open(Obs_Type+'.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Algorithm', ' ', 'Score', 'SuccessPathLenth', 'Time Per Planning(s)', 'Arrival Rate'])

        for ALGO in algorithms:
            TPPs = np.load(f'Results/{Obs_Type}/{ALGO}/TPPs.npy')
            Scores = np.load(f'Results/{Obs_Type}/{ALGO}/Scores.npy')  # Score=Path Lenth，Score=0
            Arrived = np.load(f'Results/{Obs_Type}/{ALGO}/Arrived.npy')

            SuccessPathLenth = Scores[Scores>0].copy()
            Scores[Scores>0] = 1/Scores[Scores>0]

            csv_writer.writerow([ALGO, 'mean', round(Scores.mean(), 5), round(SuccessPathLenth.mean(), 3), round(TPPs.mean(),3), Arrived.mean()])
            csv_writer.writerow([ALGO, 'std.', round(Scores.std(), 5), round(SuccessPathLenth.std(), 3), round(TPPs.std(),3), Arrived.mean()])


            print(f'Env:{Obs_Type}, Algorithm:{ALGO}, '
                  f'Score_mean:{round(Scores.mean(), 5)}, Score_std:{round(Scores.std(),5)}, '
                  f'SPL_mean:{round(SuccessPathLenth.mean(), 3)}, SPL_std:{round(SuccessPathLenth.std(),3)}, '
                  f'TPP_mean:{round(TPPs.mean(),3)}, TPP_std:{round(TPPs.std(),5)}, Arrived Rate:{Arrived.mean()}')
        print('\n')


