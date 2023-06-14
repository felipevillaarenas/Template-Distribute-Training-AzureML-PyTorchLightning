import argparse
import torch
import pytorch_lightning

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from datamodule import MyAwesomeDataModule

from model import MyAwesomeModel

from azureml.core.run import Run

def train(args):
    """
    My Awesome Trainer.
    """
    # Seed everything
    pytorch_lightning.seed_everything(42)

    # DataModule
    dm = MyAwesomeDataModule(args)
        
    # Model definition   
    model = MyAwesomeModel(args)

    # Callbacks
    lr_monitor = LearningRateMonitor()

    model_checkpoint = ModelCheckpoint(
        dirpath="./logs",
        monitor="val/acc",
        save_last=True,
        mode="max",
        save_top_k = 1
    )

    callbacks = [lr_monitor, model_checkpoint] 
    
    # MLFlow Logger
    run = Run.get_context()
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri() 
    mlf_logger = MLFlowLogger(experiment_name=run.experiment.name,
                              tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True),
        plugins=MPIEnvironment(),
        precision=args.precision,
        callbacks=callbacks,
        logger=mlf_logger,
        sync_batchnorm=True,
        use_distributed_sampler=True
    )

    trainer.fit(model, datamodule=dm)
    
    # Saving model in outputs folder
    torch.save(model, './outputs/model.pt')

def create_parser():
    """
    Create Parser. This arguments can be overwriten
    by args in the azure job.
    """
    parser = argparse.ArgumentParser()

    # Data Loader.
    parser.add_argument("--data_path", 
                        default="/path_to_my_awesome_data", 
                        type=str)

    # Data Transforms
    parser.add_argument("--batch_size", default=16, type=int)

    # Optim params
    parser.add_argument("--learning_rate", default=1.8, type=float)
    parser.add_argument("--max_epochs", default=25, type=int)
    parser.add_argument("--precision", default="16-mixed", type=str)

    # Trainer & Infrastructure
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--devices", default=8, type=int)
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)

    args = parser.parse_args()

def main():
    """
    Main functionality. 
    """
    args = create_parser()
    train(args)

if __name__ == "__main__":
    main()