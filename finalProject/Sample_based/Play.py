from Env_for_SamplePlanner import PathPlanningEnv
import numpy as np
import torch



# ['rrt', 'rrt_star']
Planner = 1
Obs_Type = 'CMO'  # CMO/RMO

def Planner_evaluation(PlannerIdx, Obs_Type, seed):
    # Seed Everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    doe = PathPlanningEnv(PlannerIdx, Obs_Type)
    doe.Render_Init(FPS=5)
    return doe.Dynamic_Plan()  # Collide:0 ; Succeed:travel lenth


if __name__ == '__main__':
    for seed in range(0,100):
        current_score, TPP = Planner_evaluation(Planner, Obs_Type, seed)
        print(f'Seed:{seed}, Travel Distance:{current_score}, TPP:{TPP}', '\n')
