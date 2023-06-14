# AzureML Job for distribute Traning 
from pathlib import Path
from azureml.core import Workspace
from azureml.core import (ScriptRunConfig, Experiment, 
                          Environment, Dataset, ComputeTarget)
from azureml.core.runconfig import MpiConfiguration, DockerConfiguration

# get workspace
ws = Workspace.from_config("config/workspace.json")

# Connect to training cluster
compute_name = 'My-Cluster' 
compute_target = ComputeTarget(workspace=ws, name=compute_name)

# Connect to experiment
experiment_name = 'My-Awesome-Experiment'
exp = Experiment(workspace=ws, name=experiment_name)

# Connect to curated environment
environment_name = 'My-Environment'
env = Environment.get(workspace=ws, name=environment_name)

# Connect to to dataset
dataset = Dataset.get_by_name(ws, name='MyDataset')

# training script
prefix = Path(__file__).parent
source_dir = str(prefix.joinpath('../src'))
script_name = 'trainer.py'

# Number of GPUs per Node, Nodes and Workers per GPU
devices = 8
num_nodes = 2
num_workers = 5

args = ['--data_path', dataset.as_named_input('MyDataset').as_download(),
        '--devices', devices,
        '--num_nodes', num_nodes,
        '--num_workers', num_workers]

# Create distributed config
distr_config = MpiConfiguration(node_count=num_nodes, 
                                process_count_per_node=devices)

# Docker config
docker_config = DockerConfiguration(use_docker=True, shm_size='32g')

# create job config
src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=args,
    compute_target=compute_name,
    environment=env,
    distributed_job_config=distr_config,
    docker_runtime_config=docker_config
)

# submit job
run = Experiment(ws, experiment_name).submit(src)