import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.multi_organ_aug import MultiOrganRandomGenerator
from datasets.region_mix import TrainDonorPool, OrganRegionMixer, RegionMixDataset
from datasets.region_mix_refinement import REFINEMENTS, BALANCE_RULE, SEAM_RULE
from engine_synapse import train_one_epoch, val_one_epoch
from innovation_audit import ModuleAudit
from split_guard import synapse_split_manifest
import argparse
import hashlib
import json
from pathlib import Path

from models.vmunet.vmunet import VMUNet
from models.vmunet.arg_vss import arg_spec, required_arg_modules
from arg_protocol import validate_protocol

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting_synapse import setting_config

import warnings
warnings.filterwarnings("ignore")





def main(config):
    frozen_protocol = validate_protocol(config)
    config.arg_spec = arg_spec(config.arg_experiment)
    if (config.aug_preset != 'v4' or config.mix_refinement != 'balance' or config.mix_strategy != 'protected'
            or config.mix_prob != .2 or config.mix_max_attempts != 8 or config.disable_aug or not config.preserve_organs):
        raise ValueError('E1/E2 require the frozen V4.1 balance online augmentation')
    if config.network != 'vmunet' or config.num_classes != 9 or (config.input_size_h, config.input_size_w) != (224, 224):
        raise ValueError('These E1/E2 packages target VM-UNet Small on Synapse 224x224, 9 classes')
    if not config.model_config.get('load_ckpt_path') or not os.path.isfile(config.model_config['load_ckpt_path']):
        raise FileNotFoundError('Both experiments must start from the same original VM-UNet Small pretrained file')
    config.required_modules = list(dict.fromkeys(config.required_modules + required_arg_modules(config.arg_experiment)))
    config.work_dir = os.path.join(config.work_dir, "")
    if config.epochs <= 0 or config.batch_size <= 0 or config.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative")
    if config.distributed:
        raise ValueError("This audited package supports the single-process training used in your log; DDP is not implemented")
    if config.val_interval <= 0:
        raise ValueError("val_interval must be positive")
    if config.aug_preset == 'v4' and (config.datasets_name != 'synapse' or config.num_classes != 9):
        raise ValueError('V4 region mixing is implemented for Synapse 2D slices with 9 labels')
    config.audit_dir = os.path.join(config.work_dir, "audit")
    os.makedirs(config.audit_dir, exist_ok=True)


    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)
    code_paths = [Path(__file__), Path(__file__).parent / "engine_synapse.py",
                  Path(__file__).parent / "innovation_audit.py",
                  Path(__file__).parent / "split_guard.py",
                  Path(__file__).parent / "multi_organ_metrics.py",
                  Path(__file__).parent / "arg_losses.py",
                  Path(__file__).parent / "arg_audit.py",
                  Path(__file__).parent / "arg_protocol.py",
                  Path(__file__).parent / "utils.py",
                  Path(__file__).parent / "configs" / "config_setting_synapse.py",
                  Path(__file__).parent / "datasets" / "dataset.py",
                  Path(__file__).parent / "datasets" / "multi_organ_aug.py",
                  Path(__file__).parent / "datasets" / "region_mix.py",
                  Path(__file__).parent / "datasets" / "region_mix_refinement.py",
                  Path(__file__).parent / "datasets" / "multi_organ_aug_v1.py"]
    code_paths += [p for p in (Path(__file__).parent / "models" / "vmunet").glob("*.py")]
    split_manifest = synapse_split_manifest(config.list_dir)
    logger.info("UNCHANGED_SYNapse_LISTS " + json.dumps(split_manifest))
    donor_pool = None
    if config.aug_preset == 'v4':
        donor_pool = TrainDonorPool(config.data_path, config.list_dir,
                                   split_manifest['train']['cases'],
                                   output_size=(config.input_size_h, config.input_size_w),
                                   cache_path=os.path.join(config.audit_dir, 'region_mix_train_index.json'),
                                   logger=logger)
    protocol_keys = ('seed', 'batch_size', 'epochs', 'lr', 'opt', 'sch', 'T_max', 'eta_min',
                     'loss_weight', 'input_size_h', 'input_size_w', 'num_classes', 'model_config')
    pretrained_sha256 = hashlib.sha256(Path(config.model_config['load_ckpt_path']).read_bytes()).hexdigest()
    signature = dict(arg=config.arg_spec, frozen_protocol=frozen_protocol, pretrained_sha256=pretrained_sha256,
                     preset=config.aug_preset, disabled=config.disable_aug,
                     dataset_lists=split_manifest,
                     region_mix=(dict(probability=config.mix_prob, strategy=config.mix_strategy,
                                      max_attempts=config.mix_max_attempts,
                                      refinement=config.mix_refinement,
                                      balance_rule=BALANCE_RULE, seam_rule=SEAM_RULE,
                                      training_index=donor_pool.fingerprint) if donor_pool else None),
                     protocol={key: repr(getattr(config, key, None)) for key in protocol_keys},
                     preserve_organs=config.preserve_organs, required_modules=config.required_modules,
                     hashes={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in code_paths})
    logger.info("ACTIVE_AUGMENTATION " + json.dumps(signature))
    previous_config = os.path.join(config.audit_dir, "run_config.json")
    if os.path.exists(previous_config):
        with open(previous_config) as handle:
            if json.load(handle).get("signature") != signature:
                raise RuntimeError("This work_dir belongs to a different augmentation experiment")
    with open(os.path.join(config.audit_dir, "run_config.json"), "w") as handle:
        json.dump(dict(signature=signature, seed=config.seed, model_config=config.model_config,
                       strict_aug_audit=config.strict_aug_audit, metric_spacing=config.metric_spacing,
                       surface_tolerance=config.surface_tolerance,
                       effective_config={key: repr(value) for key, value in vars(config).items() if not key.startswith('_')}),
                  handle, indent=2)






    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    gpus_type, gpus_num = torch.cuda.get_device_name(), torch.cuda.device_count()
    if config.distributed:
        print('#----------Start DDP----------#')
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(config.seed)
        config.local_rank = torch.distributed.get_rank()





    print('#----------Preparing dataset----------#')
    augmentation = MultiOrganRandomGenerator(
        output_size=[config.input_size_h, config.input_size_w],
        preset=config.aug_preset, disable=config.disable_aug, min_retention=.8,
        preserve_organs=config.preserve_organs,
        probabilities=({'region_mix': (0. if 'region_mix' in config.disable_aug else config.mix_prob)}
                       if config.aug_preset == 'v4' else None))
    augmentation.num_classes = config.num_classes
    train_dataset = config.datasets(
        base_dir=config.data_path,
        list_dir=config.list_dir,
        split="train",
        transform=None if config.aug_preset == 'v4' else augmentation,
    )
    if donor_pool is not None:
        mixer = OrganRegionMixer(donor_pool, probability=augmentation.probabilities['region_mix'],
                                 strategy=config.mix_strategy, max_attempts=config.mix_max_attempts,
                                 refinement=config.mix_refinement)
        train_dataset = RegionMixDataset(train_dataset, augmentation, mixer)
    logger.info("Training transform: %s probabilities=%s", type(train_dataset.transform).__name__,
                train_dataset.transform.probabilities)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size//gpus_num if config.distributed else config.batch_size, 
                                shuffle=(train_sampler is None),
                                pin_memory=True,
                                num_workers=config.num_workers,
                                persistent_workers=False,
                                sampler=train_sampler)

    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                                batch_size=1, # if config.distributed else config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers, 
                                sampler=val_sampler,
                                drop_last=True)

    
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
        installed = model.enable_arg(config.arg_experiment, seed=config.seed)
        if installed != config.arg_spec:
            raise RuntimeError('Installed ARG configuration differs from recorded experiment')
        logger.info('ACTIVE_ARG '+json.dumps(installed))
    else: raise('Please prepare a right net!')

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    # Register parameter hooks AFTER device placement, so CUDA cannot replace them.
    module_audit = ModuleAudit(model.module, config.required_modules)
    logger.info("Required modules: %s", config.required_modules)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    if config.only_test_and_save_figs:
        config.eval_stage = "inference"
        config.eval_weight_epoch = -1
        config.eval_checkpoint = config.best_ckpt_path
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint)
        epoch = 0
        outputs = config.img_save_path
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        mean_dice, mean_hd95 = val_one_epoch(
                val_dataset,
                val_loader,
                model,
                epoch,
                logger,
                config,
                test_save_path=outputs,
                val_or_test=True
            )
        print(mean_dice, mean_hd95)
        return



    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        if checkpoint.get("augmentation_signature") != signature:
            raise RuntimeError("Resume signature differs: use a fresh work_dir for a changed augmentation experiment")
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        if donor_pool is not None:
            mixer.load_state_dict(checkpoint.get('mix_refinement_state'), saved_epoch)
        if saved_epoch > config.epochs:
            raise ValueError("Saved checkpoint is beyond the configured training budget")
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)





    print('#----------Training----------#')
    epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch) if config.distributed else None
        if isinstance(train_dataset, RegionMixDataset):
            train_dataset.set_mix_epoch(epoch)

        loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler,
            module_audit=module_audit,
        )

        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        if epoch % config.val_interval == 0:
            config.eval_stage = "periodic"
            config.eval_weight_epoch = epoch
            config.eval_checkpoint = "current"
            mean_dice, mean_hd95 = val_one_epoch(
                    val_dataset,
                    val_loader,
                    model,
                    epoch,
                    logger,
                    config,
                    test_save_path=outputs,
                    val_or_test=False
                )

        torch.save(
            {
                'augmentation_signature': signature,
                'mix_refinement_state': mixer.state_dict() if donor_pool is not None else None,
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    # Keep the last weights independently of the official best-loss selection.
    torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'final.pth'))
    with open(os.path.join(checkpoint_dir, 'final_metadata.json'), 'w') as handle:
        json.dump(dict(weight_epoch=epoch, selection='last_epoch',
                       augmentation_signature=signature), handle, indent=2)

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        config.eval_stage = "best_train_loss"
        config.eval_weight_epoch = min_epoch
        config.eval_checkpoint = "best.pth"
        mean_dice, mean_hd95 = val_one_epoch(
                val_dataset,
                val_loader,
                model,
                epoch,
                logger,
                config,
                test_save_path=outputs,
                val_or_test=True
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, 
                f'best-epoch{min_epoch}-mean_dice{mean_dice:.4f}-mean_hd95{mean_hd95:.4f}.pth')
        )      


if __name__ == '__main__':
    defaults_path = Path(__file__).with_name('experiment_default.json')
    defaults = json.loads(defaults_path.read_text()) if defaults_path.exists() else {'experiment': 'E1'}
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg-experiment', choices=['E1', 'E2'], default=defaults['experiment'])
    parser.add_argument('--aug-preset', choices=['baseline', 'v1', 'v3', 'v4'], default='v4')
    parser.add_argument('--disable-aug', nargs='*', default=[],
                        choices=['crop', 'rotate', 'intensity', 'adaptive_crop', 'local_intensity', 'region_mix'])
    parser.add_argument('--mix-prob', type=float, default=.2)
    parser.add_argument('--mix-strategy', choices=['protected', 'organ', 'cutmix'], default='protected')
    parser.add_argument('--mix-max-attempts', type=int, default=8)
    parser.add_argument('--mix-refinement', choices=REFINEMENTS, default='balance',
                        help='V4.1 ablations: training-only balance, image seam treatment, or both; none reproduces V4')
    parser.add_argument('--no-organ-protection', action='store_true', help='explicit ablation only')
    parser.add_argument('--allow-zero-aug', action='store_true', help='diagnostics on tiny datasets; disables zero-use failure only')
    parser.add_argument('--required-modules', nargs='+', default=['vmunet'], help='exact model.named_modules() names BEFORE DataParallel')
    parser.add_argument('--metric-spacing', nargs=3, type=float, metavar=('Z_MM','Y_MM','X_MM'))
    parser.add_argument('--surface-tolerance', type=float, default=1.)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--val-interval', type=int)
    parser.add_argument('--work-dir')
    parser.add_argument('--pretrained')
    args = parser.parse_args()
    if args.aug_preset == 'baseline' and (args.disable_aug or args.no_organ_protection):
        parser.error('baseline uses the official transform; augmentation ablations require v1/v3/v4')
    if not np.isfinite(args.mix_prob) or not 0 <= args.mix_prob <= 1 or args.mix_max_attempts < 1:
        parser.error('mix-prob must be in [0,1] and mix-max-attempts must be positive')
    if args.aug_preset != 'v4' and (args.mix_prob != .2 or args.mix_strategy != 'protected' or args.mix_max_attempts != 8):
        parser.error('mix-prob/strategy/max-attempts options require --aug-preset v4')
    if args.mix_refinement != 'none' and (args.aug_preset != 'v4' or args.mix_strategy != 'protected' or
                                           args.mix_prob == 0 or 'region_mix' in args.disable_aug):
        parser.error('mix-refinement requires enabled V4 protected mixing with positive probability')
    if args.aug_preset == 'v4' and args.no_organ_protection:
        parser.error('Use --mix-strategy organ for the region foreground-protection ablation; keep V1 protection fixed')
    config = setting_config
    config.arg_experiment = args.arg_experiment
    config.work_dir = args.work_dir or ('results/arg_'+args.arg_experiment.lower()+'_balance/')
    config.aug_preset = args.aug_preset
    config.disable_aug = args.disable_aug
    config.mix_prob = args.mix_prob
    config.mix_strategy = args.mix_strategy
    config.mix_max_attempts = args.mix_max_attempts
    config.mix_refinement = args.mix_refinement
    config.preserve_organs = not args.no_organ_protection
    config.strict_aug_audit = not args.allow_zero_aug
    config.required_modules = args.required_modules
    config.metric_spacing = tuple(args.metric_spacing) if args.metric_spacing else None
    config.surface_tolerance = args.surface_tolerance
    if not np.isfinite(args.surface_tolerance) or args.surface_tolerance <= 0 or (args.metric_spacing and any(s <= 0 or not np.isfinite(s) for s in args.metric_spacing)):
        parser.error('spacing and tolerance must be finite and positive')
    for key in ('epochs', 'num_workers', 'batch_size', 'lr', 'seed', 'val_interval', 'work_dir'):
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    if args.pretrained:
        config.model_config = dict(config.model_config, load_ckpt_path=args.pretrained)
    main(config)       
