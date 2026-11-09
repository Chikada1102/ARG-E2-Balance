"""Fail early if a comparison accidentally changes the frozen VM-UNet setup."""
import math


def validate_protocol(config):
    expected = dict(seed=42, batch_size=32, epochs=300, lr=.001, opt='AdamW',
                    betas=(.9, .999), eps=1e-8, weight_decay=.01, amsgrad=False,
                    sch='CosineAnnealingLR', T_max=100, eta_min=1e-5, last_epoch=-1,
                    amp=False, loss_weight=[1, 1], val_interval=100)
    for key, want in expected.items():
        got = getattr(config, key, None)
        equal = math.isclose(got, want, rel_tol=1e-10, abs_tol=0.) if isinstance(want, float) and isinstance(got, (int, float)) else got == want
        if not equal:
            raise ValueError(f'Frozen E1/E2 protocol: {key} must be {want!r}, received {got!r}')
    cfg = config.model_config
    for key, want in dict(num_classes=9, input_channels=3, depths=[2, 2, 2, 2],
                          depths_decoder=[2, 2, 2, 1], drop_path_rate=.2).items():
        if cfg.get(key) != want:
            raise ValueError(f'Frozen VM-UNet Small model: {key} must be {want!r}')
    if type(config.criterion).__name__ != 'CeDiceLoss':
        raise ValueError('Keep the original CeDiceLoss as the MAIN segmentation criterion')
    if (getattr(config.criterion, 'loss_weight', None) != [1, 1]
            or getattr(getattr(config.criterion, 'diceloss', None), 'n_classes', None) != 9):
        raise ValueError('The actual main criterion must use CeDiceLoss(9, [1, 1])')
    return {key: list(value) if isinstance(value, tuple) else value for key, value in expected.items()}
