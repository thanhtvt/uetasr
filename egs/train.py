import argparse
import os
import pyrootutils
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml

tf.get_logger().setLevel("DEBUG")

pyrootutils.setup_root(
    os.path.dirname(os.path.dirname(__file__)),
    indicator=".project_root",
    pythonpath=True
)


parser = argparse.ArgumentParser(
    description="Train a UETSpeech experiment",
)
parser.add_argument(
    "config_file",
    type=str,
    help="A yaml-formatted file using the extended YAML syntax.",
)
parser.add_argument(
    "--dataset",
    default="vlsp2022",
    type=str,
    help="Dataset name."
)
parser.add_argument(
    "--devices",
    default="-1",
    type=str,
    help="Devices for training, separate by comma.",
)

parser.add_argument(
    "--mxp",
    action="store_true",
    help="Enable to train with mixed-precision float 16 bit.",
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

if args.mxp:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')


def train(config_file):
    with open(config_file) as fin:
        modules = load_hyperpyyaml(fin)
        model = modules['model']
        model.summary()
        train_loader = modules['train_loader']
        dev_loader = modules['dev_loader']
        cmvn_loader = None
        if 'cmvn_loader' in modules:
            cmvn_loader = modules['cmvn_loader']
        trainer = modules['trainer']
        trainer.train(train_loader, dev_loader, cmvn_loader)


if args.devices != '-1' and len(args.devices.split(',')) > 1:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        train(args.config_file)
else:
    train(args.config_file)
