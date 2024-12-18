from Env_for_SearchPlanner import PathPlanningEnv
import numpy as np
import torch



# [A*, Dijkstra, BestFirst, Bi-directional A*, Breadth First Search, Minimum Spanning Tree]
Planner = 2
Obs_Type = 'CMO'

def Planner_evaluation(PlannerIdx, Obs_Type, seed):
    # Seed Everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    doe = PathPlanningEnv(PlannerIdx, Obs_Type)
    doe.Render_Init()
    return doe.Dynamic_Plan()  # Collide:0 ; Succeed:travel lenth


if __name__ == '__main__':
    for seed in range(100):
        current_score, TPP = Planner_evaluation(Planner, Obs_Type, seed)
        print(f'Seed:{seed}, Travel Distance:{current_score}, TPP:{TPP}', '\n')
