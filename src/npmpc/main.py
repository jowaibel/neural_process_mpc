#!/usr/bin/env python3

import os
os.environ["MKL_DEBUG_CPU_TYPE"] = "5" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"


"""Entry point for the NP-MPC project pipeline."""

import argparse
import json
import logging
import torch

from npmpc.data.generators import GENERATOR_REGISTRY
from npmpc.data import DATASET_REGISTRY
from npmpc.data.recording.recorder import DataRecorder
from npmpc.data.recording.estimator import post_experiment_analysis
from npmpc.training import TRAINING_REGISTRY
from npmpc.mpc import MPC_REGISTRY
from npmpc.nps.NeuralProcess import NeuralProcess
from npmpc.utils.utils import set_folder, resolve_path, setup_logging


# -----------------------------------------------------------
# Default configurations
# -----------------------------------------------------------

def create_dataset_mode_config():
    return {
        "system": "furuta",
        "create_dataset": "data/furuta_KAN/furuta_system0.json",
        "output": "datasets/train_dataset/",
        "device": "mps",
        "debug": False,
    }

def training_mode_config():
    return {
        "system": "furuta",
        "train": "data/furuta_KAN/furuta_training.json",
        "model": "data/furuta_KAN/furuta_np.json",
        "train_data": "data/furuta_KAN/train_dataset/",
        "test_data": "data/furuta_KAN/test_dataset/",
        "debug": False,
    }

def testing_mode_config():
    return {
        "system": "furuta",
        "test": "data/furuta_KAN/furuta_training.json",
        "model": "data/furuta_KAN/furuta_np.json",
        "test_data": "data/furuta_KAN/test_dataset/",
        "debug": False,
    }

def deploy_mode_config():
    return {
        "system": "furuta",
        "deploy": "data/furuta_KAN/furuta_mpc.json",
        "model": "data/furuta_KAN/furuta_np.json",
        "test_data": "data/furuta_KAN/test_dataset/",
        "debug": False,
    }

def experiment_mode_config():
    return {
        "system": "furuta",
        "model": "data/furuta_KAN/furuta_np.json",
        "test_data": "data/furuta_KAN/test_dataset/",
        "experiment": "data/furuta_KAN/furuta_experiment.json",
        "debug": False,
    }


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NP-MPC pipeline.")

    # Actions
    parser.add_argument("--create_dataset", type=str, help="Create dataset from system params JSON.")
    parser.add_argument("--train", type=str, help="Train a model. Path to training config JSON.")
    parser.add_argument("--test", type=str, help="Test a model. Path to testing config JSON.")
    parser.add_argument("--deploy", type=str, help="Deploy MPC. Path to MPC config JSON.")
    parser.add_argument("--experiment", type=str, help="Run simulation experiment. Path to experiment config JSON.")

    # Create dataset options
    parser.add_argument("--output", type=str, help="Output directory for --create_dataset.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for dataset generation (cpu or cuda).")

    # Specifications
    parser.add_argument("--system", type=str, default="furuta", help="System (e.g. 'furuta').")
    parser.add_argument("--model", type=str, help="Path to model config JSON.")
    parser.add_argument("--train_data", type=str, help="Path to training dataset directory.")
    parser.add_argument("--test_data", type=str, help="Path to test dataset directory.")

    # Predefined modes
    parser.add_argument("--config", type=str, help="Path to full config JSON.")
    parser.add_argument("--create_dataset_mode", action="store_true", help="Use default create_dataset config.")
    parser.add_argument("--train_mode", action="store_true", help="Use default training config.")
    parser.add_argument("--test_mode", action="store_true", help="Use default testing config.")
    parser.add_argument("--deploy_mode", action="store_true", help="Use default deploy config.")
    parser.add_argument("--experiment_mode", action="store_true", help="Use default experiment config.")

    parser.add_argument("--folder", type=str, default=None,
                        help="Base folder for bare-filename paths (CLI args and JSON path values).")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(resolve_path(path)) as f:
        return json.load(f)


# -----------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------

def create_dataset(config: dict):
    GeneratorClass = GENERATOR_REGISTRY[config['system']]
    system_params = load_config(config['create_dataset'])
    generator = GeneratorClass(system_params)
    generator.save_to_disk(config['output'], device=config.get('device', 'cpu'))


def train_model(config: dict):
    DatasetClass = DATASET_REGISTRY[config['system']]
    TrainerClass = TRAINING_REGISTRY[config['system']]
    model_params = load_config(config['model'])
    training_params = load_config(config['train'])
    train_dataset = DatasetClass(config['train_data'])
    test_dataset = DatasetClass(config['test_data'])
    model = NeuralProcess(model_params, scaler_params=train_dataset.get_scaling())
    trainer = TrainerClass(model)
    trainer.train(train_dataset=train_dataset, test_dataset=test_dataset, params=training_params)


def test_model(config: dict):
    DatasetClass = DATASET_REGISTRY[config['system']]
    TrainerClass = TRAINING_REGISTRY[config['system']]
    model_params = load_config(config['model'])
    testing_params = load_config(config['test'])
    test_dataset = DatasetClass(config['test_data'])
    model = NeuralProcess(model_params)
    tester = TrainerClass(model)
    val_loss, val_r2 = tester.validate(dataset=test_dataset, params=testing_params)
    logging.info(f"Test: loss={val_loss:+.4f}  R2={val_r2:5.2f}%")


def deploy_model(config: dict):
    TrainerClass = TRAINING_REGISTRY[config['system']]
    MPCClass = MPC_REGISTRY[config['system']]
    model_params = load_config(config['model'])
    deploy_params = load_config(config['deploy'])
    model = NeuralProcess(model_params)
    if deploy_params.get('z') is None:
        experiment_params = load_config(deploy_params['experiment'])
        traj = DataRecorder.get_trajectory(experiment_params)
        trainer = TrainerClass(model)
        z = trainer.estimate_z(traj[0], experiment_params['n_context'])
        deploy_params['z'] = z.squeeze().tolist()
        logging.info(f"Estimated z: {deploy_params['z']}")
        # Optional analysis on the same trajectory before running the MPC.
        post_experiment_analysis(traj, experiment_params, trainer=trainer)
    mpc_controller = MPCClass(model, params=deploy_params)
    mpc_controller.run_controller()


def run_experiment(config: dict):
    experiment_params = load_config(config['experiment'])
    traj = DataRecorder.get_trajectory(experiment_params)

    trainer = None
    if experiment_params.get('compare_with_np', False):
        model = NeuralProcess(load_config(config['model']))
        trainer = TRAINING_REGISTRY[config['system']](model)

    post_experiment_analysis(traj, experiment_params, trainer=trainer)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    args = parse_args()

    set_folder(args.folder)
    log_file = setup_logging(debug=args.debug)
    logging.info(f"Logging to {log_file}")

    if args.config:
        config = load_config(args.config)
    elif args.create_dataset_mode:
        config = create_dataset_mode_config()
    elif args.train_mode:
        config = training_mode_config()
    elif args.test_mode:
        config = testing_mode_config()
    elif args.deploy_mode:
        config = deploy_mode_config()
    elif args.experiment_mode:
        config = experiment_mode_config()
    else:
        config = {k: v for k, v in {
            "system": args.system, "model": args.model,
            "create_dataset": args.create_dataset, "output": args.output, "device": args.device,
            "train": args.train, "train_data": args.train_data,
            "test": args.test, "test_data": args.test_data,
            "experiment": args.experiment,
            "deploy": args.deploy,
        }.items() if v is not None}

    if args.debug:
        config["debug"] = True

    # Dispatch
    if "create_dataset" in config:
        create_dataset(config)
    elif "train" in config:
        train_model(config)
    elif "test" in config:
        test_model(config)
    elif "deploy" in config:
        deploy_model(config)
    elif "experiment" in config:
        run_experiment(config)


if __name__ == "__main__":
    main()
