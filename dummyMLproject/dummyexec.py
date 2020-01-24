import papermill as pm
import mlflow
from pathlib import Path
import sys


experimentName = sys.argv[2]
mlflow.set_experiment(experimentName)
mlflow.start_run()

import jupyter_client

print(jupyter_client.kernelspec.find_kernel_specs())

runID = mlflow.active_run().info.run_id
experimentID = mlflow.active_run().info.experiment_id
print("experiment id is:\n", experimentID)

# ipython kernelspec list

pm.execute_notebook(
   './dummytrain.ipynb',
   './dummytrain_out.ipynb',
   parameters=dict(runID=runID, experimentName=experimentName),
   kernel_name='python3'
)

mlflow.log_artifact('./dummytrain_out.ipynb')
mlflow.end_run()