import argparse
import yaml
import logging
import os
from datetime import datetime
from losses import l1_loss

import numpy as np
import torch
from backbones import get_model
from Auxis.AuxiModel import AuxiRecon
from dataset import get_dataloader
from losses import CombinedMarginLoss, AdaFaceLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config, load_yaml
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

USE_AUXI = True


def main(args):
    # get config
    cfg = get_config(args.config)
    cfg_auxi=None
    if USE_AUXI:
        cfg_auxi = load_yaml(args.configauxi)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    if cfg.pretrained is not None:
       backbone.load_state_dict(torch.load(cfg.pretrained, map_location=torch.device('cuda')))
    backbone = torch.nn.parallel.DistributedDataParallel(module=backbone, broadcast_buffers=False, device_ids=[local_rank],
                                                                          bucket_cap_mb=16, find_unused_parameters=False)
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.train()
    # for p in backbone.parameters():
    #     p.requires_grad = False
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    if USE_AUXI:
       Auximodel = AuxiRecon(cfg_auxi).cuda()
       Auximodel.load_state_dict(torch.load("./pretrained/auxi.pt", map_location=torch.device('cuda')))
       Auximodel = torch.nn.parallel.DistributedDataParallel(module=Auximodel, broadcast_buffers=False, device_ids=[local_rank],
                                                                   bucket_cap_mb=16,find_unused_parameters=False)
       Auximodel.train()
       auxiopt = torch.optim.Adam(
           params=[{"params": Auximodel.module.netL.parameters()},
                   {"params": Auximodel.module.netV.parameters()},
                   {"params": Auximodel.module.netD.parameters()},
                   {"params": Auximodel.module.netA.parameters()}],
                   lr=cfg_auxi.get('lr', 0.0001), betas=(0.9, 0.999), weight_decay=5e-4)



    # margin_loss = AdaFaceLoss(64,
    #                           m=0.4,
    #                           h=0.333,
    #                           t_alpha=0.01,
    #                           interclass_filtering_threshold=cfg.interclass_filtering_threshold
    # )

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        if rank == 0:
           module_partial_fc.load_state_dict(torch.load("./pretrained/partial_fc_gpu0.pt", map_location=lambda storage, loc:storage.cuda(0)))
        if rank == 1:
           module_partial_fc.load_state_dict(torch.load("./pretrained/partial_fc_gpu1.pt", map_location=lambda storage, loc:storage.cuda(1)))
        # for p in module_partial_fc.parameters():
        #     p.requires_grad = False
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        if rank == 0:
           module_partial_fc.load_state_dict(torch.load("./pretrained/partial_fc_gpu0.pt", map_location=lambda storage, loc:storage.cuda(0)))
        if rank == 1:
           module_partial_fc.load_state_dict(torch.load("./pretrained/partial_fc_gpu1.pt", map_location=lambda storage, loc:storage.cuda(1)))
        # for p in module_partial_fc.parameters():
        #     p.requires_grad = False
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],  #{"params": None if cfg.backbonefrz else backbone.parameters()}, {"params": module_partial_fc.parameters()}
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size #* cfg.gradient_acc
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step, #4
        total_iters=cfg.total_step)   #

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    #loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            if USE_AUXI:
               auxiopt.zero_grad()
            global_step += 1
            local_embeddings, f1, f2 = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            if USE_AUXI:
                reconim, reconin, reconmask, recondepth, reconalbedo = Auximodel(f1, f2)
                reconloss: torch.Tensor = l1_loss(reconim, reconin, reconmask)
            # loss = loss / cfg.gradient_acc
            if cfg.fp16:
                if USE_AUXI:
                   loss_total = amp.scale(loss)+reconloss
                   loss_total.backward()
                else:
                    amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
                    if USE_AUXI:
                        auxiopt.step()

            else:
                if USE_AUXI:
                   loss_total = loss+reconloss
                   loss_total.backward()
                else:
                   loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
                    if USE_AUXI:
                        auxiopt.step()
            lr_scheduler.step()
            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss.item(),
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                if USE_AUXI:
                    callback_logging(cfg.gradient_acc, global_step, loss.item(), reconloss.item(), epoch, cfg.fp16, lr_scheduler.get_last_lr()[0],
                                     amp, reconim, recondepth, reconalbedo)
                # else:
                #     callback_logging(cfg.gradient_acc, global_step, loss.item(), epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp, None)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)
            partial_fc_path = os.path.join(cfg.output, "partial_fc_gpu0.pt")
            torch.save(module_partial_fc.state_dict(), partial_fc_path)

            if USE_AUXI:
                path_auxi = os.path.join(cfg.output, "auxi.pt")
                torch.save(Auximodel.module.state_dict(), path_auxi)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if rank == 1:
            partial_fc_path = os.path.join(cfg.output, "partial_fc_gpu1.pt")
            torch.save(module_partial_fc.state_dict(), partial_fc_path)
                
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        partial_fc_path = os.path.join(cfg.output, "partial_fc_gpu0.pt")
        torch.save(module_partial_fc.state_dict(), partial_fc_path)

        if USE_AUXI:
            path_auxi = os.path.join(cfg.output, "auxi.pt")
            torch.save(Auximodel.module.state_dict(), path_auxi)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)
    if rank == 1:
        partial_fc_path = os.path.join(cfg.output, "partial_fc_gpu1.pt")
        torch.save(module_partial_fc.state_dict(), partial_fc_path)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("configauxi", type=str, help="yaml config file")
    main(parser.parse_args())
