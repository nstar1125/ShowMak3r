import os
import zipfile
from typing import List, Optional

from pathlib import Path
from omegaconf import OmegaConf


def dump_config(cfg, save_dir: Path):
    """"""
    if save_dir.is_dir():
        save_dir = save_dir / 'config.yaml'

    print(f"[INFO] save cfg in {str(save_dir)}")
    OmegaConf.save(cfg, save_dir)


def archive_code(arc_path: Path, filetypes=['.py', '.yml']):
    """
        Dump the codes in zip file for reproducability 
        (ref: https://github.com/elliottwu/unsup3d/blob/master/unsup3d/utils.py#L52)
    """

    if str(arc_path).endswith(".zip"):
        arc_path.parent.mkdir(exist_ok=True)
    else:
        arc_path.mkdir(exist_ok=True)
        arc_path = arc_path / 'code_archive.zip'
    print(f"Archiving code to {str(arc_path)}")
    
    zipf = zipfile.ZipFile(str(arc_path), 'w', zipfile.ZIP_DEFLATED)

    # Code Must be run in repo directory
    cur_dir = os.getcwd()
    flist = []
    for ftype in filetypes:
        flist.extend(glob.glob(os.path.join(cur_dir, '**', '*'+ftype), recursive=True))
    [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    zipf.close()


def cli_printing_settings(
    torch_print_precision: int=6,
):
    torch.set_printoptions(precision=torch_print_precision)


def print_cli(texts: str, content_type: str="info"):
    valid_content_types = ["info", "debug", "warn" "error"]

    if content_type == "info":
        print(f"\n[INFO] {texts}\n")
    elif content_type == "debug":
        print(f"[DEBUG] {texts}")
    elif content_type == "warn":
        print(f"\n\n!!!!!!!!!!!!!!!!!\n[WARNING] {texts}\n!!!!!!!!!!!!!!!!!\n\n")
    elif content_type == "error":
        line = ""
        for _ in range(30):
            line += "-"
        print(f"\n{line}\n" )
        print(f"[ERROR] {texts}")
        print(f"\n{line}" )
    
    else:
        # just print in cli
        print(texts)





def init_wandb(cfg, people_ids: Optional[List]=None, project_name: str="GTU", exp_name: str='default'):
    """
        Initialize cfg of wandb
    """
    import wandb
    wandb.init(
        project=project_name,
        name=exp_name
    )
    wandb.config.update(cfg)


    # define our custom x axis metric
    wandb.define_metric("loss/step")
    wandb.define_metric("loss/*", step_metric="loss/step")

    # define time axis for metric
    wandb.define_metric("metric/*", step_metric="loss/step")
    wandb.define_metric("infos/*", step_metric="loss/step")
    
    if people_ids is not None:
        for pid in people_ids:
            wandb.define_metric(f"_{pid}/*", step_metric=f"loss/step")
    
