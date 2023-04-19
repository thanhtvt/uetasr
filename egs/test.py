import os
import argparse
import pyrootutils
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

tf.get_logger().setLevel("DEBUG")

pyrootutils.setup_root(
    Path(__file__).resolve().parents[1],
    indicator=".project_root",
    pythonpath=True
)


parser = argparse.ArgumentParser(
    description="Test a VinSpeech experiment",
)

parser.add_argument(
    "config",
    type=str,
    help="Name of a yaml-formatted file using the extended YAML syntax "
    "defined by VinSpeech.",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    default='checkpoints',
    help="Name of folder or file to load pretrained model in exp_dir.",
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Beam size in beam search.",
)

parser.add_argument(
    "--devices",
    default="-1",
    type=str,
    help="Devices for training, separate by comma.",
)

parser.add_argument(
    "--loss",
    action="store_true",
    help="Enable to evaluate loss in test data.",
)

parser.add_argument(
    "--summary",
    action="store_true",
    help="Enable to print summary.",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable to print samples in test set.",
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

exp_dir = os.path.dirname(args.config)


def test(config_file):
    with open(config_file) as fin:
        modules = load_hyperpyyaml(fin)

        if args.summary:
            model = modules['model']
            model.summary()

        test_loader = modules['test_loader']
        trainer = modules['trainer']

        checkpoint_path = args.checkpoint_path
        checkpoint_name = os.path.basename(checkpoint_path)
        trainer.load_model(checkpoint_path)

        result_dir = os.path.join(exp_dir, 'results', checkpoint_name)

        trainer.evaluate(test_loader,
                         result_dir=result_dir,
                         num_samples=args.num_samples,
                         return_loss=args.loss,
                         verbose=args.verbose)


config_file = args.config
if args.devices != '-1' and len(args.devices.split(',')) > 1:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        test(config_file)
else:
    test(config_file)
