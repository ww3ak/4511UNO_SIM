import os
import pandas as pd
import numpy as np

from src.game import tournament
import config as conf

def main():

    run = tournament(
        iterations = conf.params['iterations'],
        algo       = conf.params['algorithm'],
        comment    = conf.params['logging'],
        agent_info = conf.params['model']
    )

    result = pd.concat([
        pd.Series(run[0], name='winner'), 
        pd.Series(run[1], name='turns')
    ], axis = 1)
    
    result["win_rate"] = np.where(result["winner"]==conf.player_name_1,1,0)
    result["win_rate"] = result["win_rate"].cumsum()/(result.index+1)

    q_vals = pd.DataFrame(run[2].q)
    q_vals.index.rename("id", inplace=True)

    if not os.path.exists("assets"):
        os.makedirs("assets")

    # Access the algorithm name from config
    algorithm_name = conf.params['algorithm']

    # Remove any special characters and replace spaces or hyphens

    # Modify your DataFrame saving calls
    q_vals.to_csv(f"assets/{algorithm_name}-q-values.csv", index=True)
    result.to_csv(f"assets/{algorithm_name}-results.csv", index=False)
    

if __name__ == "__main__":
    main()