from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
import os
from azureml.widgets import RunDetails
"""
# Create a Python environment for the experiment
with open('../../azure_requirements.txt') as f:
    lines = f.readlines()

req_lines = []
for l in lines:
    no_ver_line = l.split('==')[0]
    req_lines.append(no_ver_line+'\n')

print(req_lines)

with open('req.txt', 'w') as f:
    f.writelines(req_lines)

"""
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


azure_env = Environment.from_pip_requirements(name="azure-env", file_path=dir_path+'/src/azure/req3.txt')
#azure_env = Environment.from_conda_specification(name='azure-env', file_path='../../ml_ops.yml')

#azure_env = Environment("azure_env")
#conda_dep = CondaDependencies()
#for pkg in req_lines:
    #conda_dep.add_conda_package(pkg)
    #conda_dep.add_pip_package("pillow==5.4.1")
    #conda_dep.set_pip_requirements(pip_requirements) # list of strings 
#conda_dep.set_python_version(string)
#azure_env.python.conda_dependencies=conda_dep

#azure_env.register(workspace=ws)

# Ensure the required packages are installed (we need pip, scikit-learn and Azure ML defaults)
#packages = CondaDependencies.create(conda_dependencies_file_path='../../ml_ops.yml')
#azure_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory=dir_path+'/',
                                script='src/azure/azure_main.py',
                                environment=azure_env,
                                arguments = ['train']) 

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
#run.wait_for_completion()