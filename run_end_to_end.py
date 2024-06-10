import os
import time
import sys
import math
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import random
import torch
from torchvision.ops import sigmoid_focal_loss
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from functools import partial

# custom code
from utils.label_dict import (
    protein_to_num_full,
    protein_to_num_single_cells,
    cell_to_num_full,
    cell_to_num_single_cells,
)
from utils.classification_utils import get_classifier, FocalBCELoss, threshold_output, write_to_tensorboard, get_scheduler, get_optimizer
from utils.utils import is_main_process, init_distributed_mode, get_params_groups
import utils.vision_transformer as vits
from utils.yaml_tfms import tfms_from_config
import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib
from utils.file_dataset import (
    ImageFileList,
    AutoBalancedFileList,
    AutoBalancedPrecomputedFeatures,
    default_loader,
    pandas_reader,
    pandas_reader_binary_labels,
    pandas_reader_no_labels,
    scKaggle_df_reader,
)


def save_model(model, path, args):
    if args.parallel_training:
        model_to_save = model.module
    else:
        model_to_save = model
    torch.save(model_to_save.state_dict(), path)
    del model_to_save


class classifier_transformation(object):
    def __init__(self, config, isTrain):
        self.config = config
        self.isTrain = isTrain
        (
            self.global_transfo1,
            self.global_transfo2,
            self.local_transfo,
            self.testing_transfo,
        ) = tfms_from_config(self.config)

    def __call__(self, image):
        train_crop = self.global_transfo1(image)
        test_crop = self.testing_transfo(image)
        if self.isTrain:
            return train_crop
        else:
            return test_crop


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return np.array(x.to("cpu"))
    return np.array(x)


def run_end_to_end(
    config,
    loader=default_loader,
    reader=pandas_reader_binary_labels,
    parallel_training=True,
    **kwargs,
):
    # Populate arg namespace with config parameters
    args = argparse.Namespace()
    kwargs["parallel_training"] = parallel_training
    if config.endswith("yaml"):
        config = yaml.safe_load(open(config, "r"))
    else:
        config = json.load(open(config))
        for k in config["command_lind_args"].keys():
            config["classification"][k] = config["command_lind_args"][k]
    for k in config["classification"].keys():
        setattr(args, k, config["classification"][k])
    # Override config file with command line arguments
    for k in kwargs.keys():
        if k in args.__dict__.keys():
            print(
                f"Command line arguments overridden config file {k} (was {args.__dict__[k]}, is now {kwargs[k]})"
            )
        setattr(args, k, kwargs[k])
    if parallel_training:
        init_distributed_mode(args)

    if type(args.train) == str:
        args.train = args.train == "True"
    if args.targets == "protein_localization": #(!)
        args.train_protein = "True"
    if args.targets == "cell_type": #(!)
        args.train_cell_type = "True"
    if type(args.test) == str:
        args.test = args.test == "True"
    if type(args.balance) == str:
        args.balance = args.balance == "True"
    if type(args.use_pretrained_features) == str:
        args.use_pretrained_features = args.use_pretrained_features == "True"
    if type(args.whole_images) == str:
        args.whole_images = args.whole_images == "True"
    if type(args.train_feature_extractor) == str:
        args.train_feature_extractor = args.train_feature_extractor == "True"
    if type(args.train_classifier_head) == str:
        args.train_classifier_head = args.train_classifier_head == "True"
    if type(args.load_classifier_head) == str:
        args.load_classifier_head = args.load_classifier_head == "True"
    if "skip" in args and type(args.skip) == str:
        args.skip = args.skip == "True"

    print(f"Output_dir: {args.output_dir}, output prefix: {args.output_prefix}")
    save_dir = f"{args.output_dir}/{args.output_prefix}/"
    if Path(save_dir).exists() and args.overwrite == False:
        print(
            f"\n\nError: Folder {args.output_dir}/{args.output_prefix} exists; Please change experiment name to prevent loss of information\n\n"
        )
        quit()
    # wait for all processes to check for existing folder
    time.sleep(1)

    if is_main_process():
        Path(save_dir).mkdir(exist_ok=True)
        log_params(
            config,
            kwargs,
            log_path=f"{save_dir}/experiment_params.json",
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create train/val subsets if they don't exist yet:
    # def subset_features(fts, IDs):
    #     idxs = pd.Series(fts[3]).isin(IDs)
    #     features = fts[0][idxs]
    #     prot_lbls = torch.stack(fts[1])[idxs] 
    #     prot_lbls = [prot_lbls[i,:] for i in range(prot_lbls.shape[0])] # converting back to list is v dumb. Fix?
    #     cell_lbls = list(np.array(fts[2])[idxs])
    #     IDs = list(np.array(fts[3])[idxs])
    #     return [features, prot_lbls, cell_lbls, IDs]

    def subset_features(fts, IDs):
        idxs = pd.Series(fts[3]).isin(IDs)
        return [np.array(i)[idxs] for i in fts]

    train_path = f"{args.features.split('.')[0]}_{args.targets}_train.pth"
    valid_path = f"{args.features.split('.')[0]}_{args.targets}_valid.pth"
    if not os.path.isfile(train_path):
        print(f"Preparing data subsets based on train and valid IDs:")
        features = torch.load(args.features)
        # remove rows in with no HPA_FOV protein_localization labels
        print(f"Removing feature-rows with NO positive HPA_FOV (28) protein localization labels ")
        idxs = np.array(features[1]).sum(axis=1) == 0
        features = [np.array(i)[~idxs] for i in features]
        for i in [train_path, valid_path]:
            torch.save(subset_features(features, torch.load(args.train_ids)), i)
            print(f"Saved: {i}")
    else: 
        for i in [train_path, valid_path]: print(f"Using {i}")
    
    # setup dataloader
    if args.use_pretrained_features:
        train_transform = None
        test_transform = None
    else:
        train_transform = classifier_transformation(config, isTrain=True)
        test_transform = classifier_transformation(config, isTrain=False)

    # setup feature_extractor
    if args.use_pretrained_features == False:
        feature_extractor = vits.__dict__[config["model"]["arch"]](
            img_size=[config["embedding"]["image_size"]],
            patch_size=config["model"]["patch_size"],
            drop_path_rate=0.1,  # stochastic depth
            in_chans=config["model"]["num_channels"],
        )

    # Load feature extractor weights, if needed
    if args.train_feature_extractor == False:
        if type(args.feature_extractor_state_dict) != type(None):
            state_dict = torch.load(
                args.feature_extractor_state_dict, map_location="cpu"
            )
            teacher = state_dict
            if "teacher" in state_dict.keys():
                teacher = state_dict["teacher"]
            teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
            teacher = {k.replace("backbone.", ""): v for k, v in teacher.items()}
            msg = feature_extractor.load_state_dict(teacher, strict=False)
            feature_extractor = feature_extractor.to(device)
            print(
                "Pretrained weights for feature extractor found at {} and loaded with msg: {}".format(
                    args.feature_extractor_state_dict, msg
                )
            )
            del teacher
            del state_dict

        if args.use_pretrained_features == False:
            for p in feature_extractor.parameters():
                p.requires_grad = False
            feature_extractor = feature_extractor.eval()

    if args.use_pretrained_features == False:
        feature_extractor.to(device)
        dataset_function = ImageFileList
    else:
        feature_extractor = None
        if args.targets == 'protein_localization':
            print("Training on protein localization labels")
            args.train_protein = True
            dataset_function = partial(
                AutoBalancedPrecomputedFeatures, target_column="proteins")
        elif args.targets == 'cell_type':
            print("Training on cell type labels")
            args.train_cell_type = True
            dataset_function = partial(
                AutoBalancedPrecomputedFeatures, target_column="cells")
        else: 
            print(f"{args.targets} is not implemented.")
        
        args.train_path = train_path
        args.valid_path = valid_path 
        
    # setup classifier head
    if args.use_pretrained_features == False:
        embed_dim = feature_extractor.embed_dim
    else:
        embed_dim = torch.load(args.train_path)[0].shape[1]
    classifier = get_classifier(args, embed_dim)

    # If training, assume we train classifier from scratch
    if (type(args.classifier_state_dict) != type(None)) and args.load_classifier_head:
        msg = classifier.load_state_dict(
            torch.load(args.classifier_state_dict, map_location="cpu")
        )
        print(
            "Pretrained weights for classifier found at {} and loaded with msg: {}".format(
                args.classifier_state_dict, msg
            )
        )
    classifier.to(device)

    if args.train_protein:
        if args.whole_images:
            target_labels = sorted(list(protein_to_num_full.keys()))
        else:
            target_labels = sorted(list(protein_to_num_single_cells.keys()))
        task = "protein"
    elif args.train_cell_type:
        if args.whole_images:
            target_labels = sorted(list(cell_to_num_full.keys()))
        else:
            target_labels = sorted(list(cell_to_num_single_cells.keys()))
        task = "cell_type"

    if args.parallel_training:
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[args.gpu]
        )
    if args.train_feature_extractor:
        if args.parallel_training:
            feature_extractor = torch.nn.parallel.DistributedDataParallel(
                feature_extractor, device_ids=[args.gpu]
            )
        models_parameters = list(classifier.parameters()) + list(
            feature_extractor.parameters()
        )
    else:
        models_parameters = list(classifier.parameters())
    train_ds = dataset_function(
        args.train_path,
        transform=train_transform,
        flist_reader=reader,
        # balance=False,
        balance=args.balance,
        loader=loader,
        training=True,
        with_labels=True,
        root=config["model"]["root"],
        # The target labels are the column names of the protein localizationsm
        # used to create the multilabel target matrix
        target_labels=target_labels,
    )
    valid_ds = dataset_function(
        args.valid_path,
        transform=test_transform,
        balance=False,
        # balance=args.balance,
        flist_reader=reader,
        loader=loader,
        training=True,
        with_labels=True,
        root=config["model"]["root"],
        # The target labels are the column names of the protein localizationsm
        # used to create the multilabel target matrix
        target_labels=target_labels,
    )
    train_ds.scale_features("find_statistics")
    valid_ds.scale_features(train_ds.scaler)
    torch.save(train_ds.scaler, f"{args.output_dir}/{args.output_prefix}/scaler.pth")

    # The balancing action is done online via the AutoBalancedFileList class,
    # so no weigting is needed in the sampler
    if parallel_training:
        train_sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True)
        valid_sampler = torch.utils.data.DistributedSampler(valid_ds, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
        valid_sampler = torch.utils.data.SequentialSampler(valid_ds)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.batch_size_per_gpu),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=int(args.batch_size_per_gpu),
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # setup training
    optim = get_optimizer(models_parameters, args)
    total_steps = int(args.epochs) * len(train_dl)
    scheduler, lr_schedule, wd_schedule = get_scheduler(
        optim, int(total_steps), len(train_dl), args
    )
    
    # Setup loss function
    if args.loss == "BCEWithLogitsLoss":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss == "BCELoss":
        criterion = torch.nn.BCELoss()
    elif args.loss == "focal_loss":
        criterion = partial(sigmoid_focal_loss, reduction="mean")
    if args.train:
        feature_extractor, classifier = run_training(
            feature_extractor,
            classifier,
            train_dl,
            valid_dl,
            optim,
            criterion,
            task,
            scheduler,
            lr_schedule,
            wd_schedule,
            device,
            args,
        )
    if args.test:
        train_loss, train_score = test_model(
            feature_extractor, classifier, train_dl, 0, device, criterion, args
        )
        valid_loss, valid_score = test_model(
            feature_extractor, classifier, valid_dl, 0, device, criterion, args
        )
        print(f"train score: {train_score}, valid score: {valid_score}")
        print(f"train loss: {train_loss}, valid loss: {valid_loss}")


def get_f1_score(all_targets, all_predictions):
    targets = to_np(torch.vstack(all_targets).detach().cpu())
    outputs = (
        threshold_output(torch.vstack(all_predictions).detach().cpu(), use_sigmoid=True)
        .int()
        .float()
    )
    targets = targets.reshape(targets.shape[0], targets.shape[-1])
    outputs = outputs.reshape(outputs.shape[0], outputs.shape[-1])
    indices = np.where(targets.sum(axis=0) > 0)[0]
    targets = targets[:,indices]
    outputs = outputs[:,indices]
    average_score = f1_score(
        targets,
        outputs,
        average='macro',
        zero_division=0,
    )
    full_score = f1_score(
        targets,
        outputs,
        average=None,
        zero_division=0,
    )
    print(full_score)
    return average_score


def test_model(feature_extractor, classifier, dl, epoch, device, criterion, args):
    classifier = classifier.eval()
    with torch.no_grad():
        predictions, all_targets, losses, all_features = [], [], [], []
        batch_pbar = tqdm(
            enumerate(dl),
            total=len(dl),
            unit=" test batches",
            position=1,
            leave=False,
        )
        for index, (images, targets) in batch_pbar:
            images = images.to(device)
            targets = targets.to(device)
            if args.use_pretrained_features:
                features = images
            else:
                features = feature_extractor(images)
            outputs = classifier(features)
            loss = criterion(
                outputs.reshape(targets.shape[0], 1, targets.shape[-1]),
                targets.float().reshape(targets.shape[0], 1, targets.shape[-1]),
            )
            losses.append(loss.item())
            predictions.extend(outputs.detach().cpu())
            # all_features.extend(features)
            all_targets.extend(targets.detach().cpu())
    save_dir = f"{args.output_dir}/{args.output_prefix}"
    torch.save(all_targets, f"{save_dir}/all_targets.pth")
    torch.save(predictions, f"{save_dir}/predictions.pth")
    # torch.save(all_features, f"{save_dir}/penultimate_features.pth")
    score = get_f1_score(all_targets, predictions)
    batch_pbar.close()
    classifier = classifier.train()
    return np.mean(losses), score


def run_training(
    feature_extractor,
    classifier,
    train_dl,
    valid_dl,
    optim,
    criterion,
    task,
    scheduler,
    lr_schedule,
    wd_schedule,
    device,
    args,
):
    print("running training...")
    writer = SummaryWriter(f"{args.output_dir}/logs/{args.output_prefix}/")
    epochs_to_run = int(args.epochs)
    epoch_pbar = tqdm(
        range(epochs_to_run),
        total=epochs_to_run,
        unit=" epochs",
        position=0,
        leave=True,
    )
    for epoch in epoch_pbar:
        if args.train_feature_extractor:
            feature_extractor = feature_extractor.train()
        classifier = classifier.train()
        train_predictions, train_targets, train_loss = [], [], []
        batch_pbar = tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            unit=" batches",
            position=1,
            leave=False,
        )

        for index, (images, targets) in batch_pbar:
            print(f"batch {index}")
            iteration = len(train_dl) * epoch + index  # global training iteration
            if args.schedule == "DINO_cosine":
                if args.schedule == "default":
                    for i, param_group in enumerate(optim.param_groups):
                        param_group["lr"] = lr_schedule[iteration]
                        if i == 0:  # only the first group is regularized
                            param_group["weight_decay"] = wd_schedule[iteration]

            images = images.to(device)
            targets = targets.to(device)
            if args.use_pretrained_features:
                features = images
            else:
                features = feature_extractor(images)

            outputs = classifier(features)
            targets = targets.float().reshape(targets.shape[0], 1, targets.shape[-1])
            outputs = outputs.float().reshape(outputs.shape[0], 1, outputs.shape[-1])
            loss = criterion(
                outputs, targets
            )

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            optim.zero_grad()
            param_norms = None
            loss.backward()
            optim.step()
            if args.schedule != "Flat":
                scheduler.step()

            # logging
            torch.cuda.synchronize()
            train_predictions.append(outputs)
            train_targets.append(targets)
            train_loss.append(loss.item())

            if type(scheduler) is not type(None):
                lr = scheduler.get_last_lr()[0]
            elif args.schedule == "DINO_cosine":
                lr = lr_schedule[index]
            else:
                lr = args.lr
            log = {
                "step": epoch * len(train_dl) + index,
                "LR": lr,
                "train loss per step": loss.item(),
            }

            if args.check_batch is not None:
                # A single epoch on the sc-data is huge and hence we might train for only 1 epoch.
                # Control how often we log validation losses via the check_batch and checksize paramters.
                if index > 0 and index % args.check_batch == 0:
                    with torch.no_grad():
                        vloss = []
                        for vindex, vbatch in enumerate(valid_dl):
                            vimages, vtargets = [i.to(device) for i in vbatch]
                            if args.use_pretrained_features:
                                features = vimages
                            else:
                                features = feature_extractor(vimages)
                            voutputs = classifier(features)
                            vloss.append(criterion(voutputs, vtargets.float()).item())
                            if vindex == args.checksize:
                                break
                    log["valid loss check"] = np.mean(vloss)

            write_to_tensorboard(writer, log, t="step")
            batch_pbar.set_description(
                ", ".join([f"{k}: {float(v):.4}" for k, v in log.items()])
            )
            # if index % 100000 == 0:
            #     if is_main_process():
            #         save_model(
            #             classifier,
            #             f"{args.output_dir}/{args.output_prefix}/classifier_checkpoint_{epoch}_{index}.pth",
            #             args,
            #         )

        log = {
            "epoch": epoch,
            "train loss per epoch": loss.item(),
            "train f1 per epoch:": get_f1_score(train_targets, train_predictions),
        }
        write_to_tensorboard(writer, log, t="epoch")
        # Each checkpoint_frq epochs, save models and log results
        batch_pbar.close()
        log = {}
        if epoch != 0 and epoch % args.checkpoint_frq == 0:
            if args.use_pretrained_features == False:
                feature_extractor = feature_extractor.eval()
            classifier = classifier.eval()

            if args.train_feature_extractor:
                save_model(
                    feature_extractor,
                    f"{args.output_dir}/{args.output_prefix}/feature_extractor_checkpoint_{epoch}.pth",
                    args,
                )
            save_model(
                classifier,
                f"{args.output_dir}/{args.output_prefix}/classifier_checkpoint_{epoch}.pth",
                args,
            )
            train_loss, train_score = test_model(
                feature_extractor, classifier, train_dl, epoch, device, criterion, args
            )
            valid_loss, valid_score = test_model(
                feature_extractor, classifier, valid_dl, epoch, device, criterion, args
            )

            log = {
                "epoch": epoch,
                "training-set loss": train_loss,
                "validation-set loss": valid_loss,
                "training-set f1": train_score,
                "validation-set f1": valid_score,
            }

            write_to_tensorboard(writer, log)
        log["epoch"] = epoch
        log["train loss per epoch"] = loss.item()
        write_to_tensorboard(writer, log, t="epoch")
        epoch_pbar.set_description(
            ", ".join([f"{k}: {float(v):.3}" for k, v in log.items()])
        )

    if args.test_last:
        train_loss, train_score = test_model(
            feature_extractor, classifier, train_dl, epoch, device, criterion, args
        )
        valid_loss, valid_score = test_model(
            feature_extractor, classifier, valid_dl, epoch, device, criterion, args
        )

        log = {
            "epoch": epoch,
            "training-set loss": train_loss,
            "validation-set loss": valid_loss,
            "training-set f1": train_score,
            "validation-set f1": valid_score,
        }
        write_to_tensorboard(writer, log)
        epoch_pbar.set_description(
            ", ".join([f"{k}: {float(v):.3}" for k, v in log.items()])
        )

    if is_main_process():
        if args.train_feature_extractor:
            save_model(
                feature_extractor,
                f"{args.output_dir}/{args.output_prefix}/feature_extractor_final.pth",
                args,
            )
        save_model(
            classifier,
            f"{args.output_dir}/{args.output_prefix}/classifier_final_{task}{'_whole' if args.whole_images else ''}.pth",
            args,
        )

    print(f"Final models saved to {args.output_dir}/{args.output_prefix}/")
    return feature_extractor, classifier


def log_params(config, args, log_path):
    config["command_lind_args"] = args
    with open(log_path, "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO")
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--targets", default="protein_localization", type=str, choices=["protein_localization", "cell_type"])
    parser.add_argument("--checkpoint_frq", default=50, type=int)
    parser.add_argument("--checksize", default=3, type=int)
    parser.add_argument("--check_batch", default=None, type=bool)
    parser.add_argument("--test_last", default=False, type=bool)
    parser.add_argument("--train", default="True", type=str)
    parser.add_argument("--test", default="True", type=str)
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    # parse unrecognized parameters
    args, unknown = parser.parse_known_args()
    print(args)
    # print(unknown)
    # if "--local_rank" in unknown[0]:
    #    keys = unknown[1::2]
    #    values = unknown[2::2]
    #else:
    #    keys = unknown[0::2]
    #    values = unknown[1::2]
   # for k, v in zip(keys, values):
   #     setattr(args, k.replace("--", ""), v)
    run_end_to_end(
        args.config,
        loader=default_loader,
        reader=pandas_reader_binary_labels,
        parallel_training=True if "WORLD_SIZE" in os.environ.keys() else False,
        **{k: args.__dict__[k] for k in args.__dict__.keys() if "config" not in k},
    )
