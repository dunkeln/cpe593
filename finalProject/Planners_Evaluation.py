from Env_for_SearchPlanner import PathPlanningEnv
import numpy as np
import torch
import os



def Planner_evaluation(PlannerIdx, Obs_Type, seed):
    # Seed Everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    doe = PathPlanningEnv(PlannerIdx, Obs_Type)
    # doe.Render_Init()
    return doe.Dynamic_Plan()  # Collide:0 ; Succeed:travel lenth


if __name__ == '__main__':
    os.mkdir('Results')
    os.mkdir('Results/CMO')
    os.mkdir('Results/RMO')
    eval_times = 20
    Algo_Name = ['A*', 'Dijkstra', 'BestFirst', 'Bi-directional A*', 'Breadth First Search', 'Minimum Spanning Tree']

    for Obs_Type in ['CMO', 'RMO']:
        for planner in range(6):
            TPPs = np.zeros(eval_times)
            Scores = np.zeros(eval_times)
            Arrived = np.zeros(eval_times)

            for seed in range(eval_times):
                current_score, TPP = Planner_evaluation(planner, Obs_Type, seed)  # current_score

                TPPs[seed] = TPP
                Scores[seed] = current_score
                Arrived[seed] = (current_score!=0)
                print(f'Env:{Obs_Type}, Algo:{Algo_Name[planner]}, Seed:{seed}, TPP:{round(TPP,4)}, Score:{round(current_score,4)}, Arrived:{(current_score!=0)}')

            os.mkdir(f'Results/{Obs_Type}/'+Algo_Name[planner])
            np.save(f'Results/{Obs_Type}/{Algo_Name[planner]}/TPPs.npy',TPPs)
            np.save(f'Results/{Obs_Type}/{Algo_Name[planner]}/Scores.npy', Scores)
            np.save(f'Results/{Obs_Type}/{Algo_Name[planner]}/Arrived.npy', Arrived)

    import Print_Results

