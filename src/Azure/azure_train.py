from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies

from azureml.widgets import RunDetails

# Create a Python environment for the experiment
azure_env = Environment.from_pip_requirements(name="azure-env", file_path="azure_requirements.txt")

# Create a script config
script_config = ScriptRunConfig(source_directory='Azure',
                                script='azure_main.py',
                                environment=azure_env) 

# submit the experiment run
experiment_name = 'First_MNIST_azure'
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

# Show the running experiment run in the notebook widget
RunDetails(run).show()

# Block until the experiment run has completed
run.wait_for_completion()