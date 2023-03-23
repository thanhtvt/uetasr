import os
import argparse
import traceback
import numpy as np
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml


def get_args():
    parser = argparse.ArgumentParser(
        description="Average checkpoints",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Name of a yaml-formatted file using the extended YAML syntax",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='checkpoints',
        help="Name of folder or file to load pretrained model in exp_dir.",
    )
    parser.add_argument(
        "--num_avg",
        type=int,
        default=5,
        help="Number of checkpoints to average.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='avg_checkpoint.ckpt',
        help="Name of folder or file to save average model in exp_dir.",
    )

    args = parser.parse_args()
    return args


def average_nbest_models(args):
    with open(args.config_file) as fin:
        modules = load_hyperpyyaml(fin)
        model = modules['model']
        model.summary()

        checkpoints = []
        for checkpoint_path in os.listdir(args.checkpoint_dir):
            if checkpoint_path.endswith('.index'):
                ckpt = os.path.join(args.checkpoint_dir, checkpoint_path[:-6])
                checkpoints.append(ckpt)
        checkpoints = sorted(checkpoints, reverse=True)[:args.num_avg]

        weights = []
        for checkpoint_path in checkpoints:
            model.load_weights(checkpoint_path).expect_partial()
            weights.append(model.get_weights())

        new_weights = list()
        for weights_list_tuple in zip(*weights):
            try:
                new_weights.append(np.array(weights_list_tuple).mean(axis=0))
            except Exception:
                print(weights_list_tuple)
                print(type(weights_list_tuple[0]))
                traceback.print_exc()
                exit()

        model.set_weights(new_weights)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        model.save_weights(args.output_path)
        print("Done. Model saved to", args.output_path)


if __name__ == "__main__":
    tf.get_logger().setLevel("DEBUG")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    args = get_args()
    exp_dir = os.path.dirname(args.config)
    average_nbest_models(args)
