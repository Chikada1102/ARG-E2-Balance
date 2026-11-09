from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def enable_arg(self, mode, seed=42):
        """Call AFTER load_from(), BEFORE CUDA/DataParallel/optimizer creation."""
        from .arg_vss import AnatomicalRelation, ARG_SITE, arg_spec
        stage, block = ARG_SITE
        layer = self.vmunet.layers[stage]
        if block != len(layer.blocks)-1 or layer.use_checkpoint:
            raise ValueError('ARG expects Stage 3 last block (2,1), no gradient checkpointing')
        attention = layer.blocks[block].self_attention
        if hasattr(attention, 'arg'):
            raise RuntimeError('ARG already installed')
        if attention.d_model != 384 or attention.d_inner != 768 or attention.d_state != 16:
            raise ValueError('This ARG experiment requires the audited VM-UNet Small dimensions')
        if next(self.parameters()).device.type != 'cpu':
            raise RuntimeError('Attach ARG on CPU before device placement')
        spec = arg_spec(mode)
        # Original/pretrained weights and caller CPU/CUDA RNG streams are untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            attention.arg = AnatomicalRelation(attention.d_inner, self.num_classes,
                                               spec['feature_dim'], attention.d_state, mode, spec['alpha'])
        self.arg_experiment = mode
        return spec

    def forward(self, x, return_aux=False):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x, return_aux=return_aux)
        if return_aux:
            return logits
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
