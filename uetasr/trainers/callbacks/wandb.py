try:
    import wandb
    have_wandb = True
except ImportError:
    have_wandb = False

from typing import Union


class WandbLogger(wandb.keras.WandbMetricsLogger):
    """ Inherit Callback to log metrics to Weights & Biases """

    def __init__(self,
                 tb_root_dir: str = "./tb_logs",
                 dir: str = "./wandb",
                 project_name: str = "uetasr",
                 config: dict = {},
                 save_code: bool = True,
                 resume: str = "auto",
                 log_freq: Union[int, str] = "epoch",
                 **kwargs):
        wandb.tensorboard.patch(root_dir=tb_root_dir)
        wandb.init(project=project_name,
                   config=config,
                   save_code=save_code,
                   dir=dir,
                   resume=resume)
        super(WandbLogger, self).__init__(log_freq, **kwargs)
