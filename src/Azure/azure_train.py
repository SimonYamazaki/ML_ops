from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies

from azureml.widgets import RunDetails

# Create a Python environment for the experiment
with open('../../azure_requirements.txt') as f:
    lines = f.readlines()

req_lines = []
for l in lines:
    no_ver_line = l.split('==')[0]
    req_lines.append(no_ver_line)

with open('req.txt', 'w') as f:
    f.writelines(req_lines)

azure_env = Environment.from_pip_requirements(name="azure-env", file_path="req.txt")
#azure_env = Environment.from_conda_specification(name='azure-env', file_path='../../ml_ops.yml')

# Create a script config
script_config = ScriptRunConfig(source_directory='',
                                script='azure_main.py',
                                environment=azure_env,
                                arguments = ['command', 'train']) 

# submit the experiment run
experiment_name = 'First_MNIST_azure'

from azureml.core.authentication import InteractiveLoginAuthentication
ia = InteractiveLoginAuthentication(tenant_id='f251f123-c9ce-448e-9277-34bb285911d9')
# You can find tenant id under azure active directory->properties
ws = Workspace.get(name='ML_ops',
                     subscription_id='3256dba5-2e98-4c8b-b9a4-f34b68b28b8b',
                     resource_group='Best_recource_group',auth=ia)

#ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

# Show the running experiment run in the notebook widget
#RunDetails(run).show()

# Block until the experiment run has completed
run.wait_for_completion()