

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['l', 'low-quality', 'input-only']:
        from data.dataset_l import DatasetL as D

    # -----------------------------------------
    # denoising
    # -----------------------------------------
    elif dataset_type in ['dncnn', 'denoising']:
        from data.dataset_dncnn import DatasetDnCNN as D

    elif dataset_type in ['dnpatch']:
        from data.dataset_dnpatch import DatasetDnPatch as D

    elif dataset_type in ['ffdnet', 'denoising-noiselevel']:
        from data.dataset_ffdnet import DatasetFFDNet as D

    elif dataset_type in ['fdncnn', 'denoising-noiselevelmap']:
        from data.dataset_fdncnn import DatasetFDnCNN as D

    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    elif dataset_type in ['sr', 'super-resolution']:
        from data.dataset_sr import DatasetSR as D

    elif dataset_type in ['srmd']:
        from data.dataset_srmd import DatasetSRMD as D

    elif dataset_type in ['dpsr', 'dnsr']:
        from data.dataset_dpsr import DatasetDPSR as D

    elif dataset_type in ['usrnet', 'usrgan']:
        from data.dataset_usrnet import DatasetUSRNet as D

    elif dataset_type in ['bsrnet', 'bsrgan', 'blindsr']:
        from data.dataset_blindsr import DatasetBlindSR as D

    # -------------------------------------------------
    # JPEG compression artifact reduction (deblocking)
    # -------------------------------------------------
    elif dataset_type in ['jpeg']:
        from data.dataset_jpeg import DatasetJPEG as D
    # -----------------------------------------
    # LOE Low-light image enhancement
    # -----------------------------------------
    elif dataset_type in ['loe']:
        from data.dataset_loe import DatasetLOE as D
        
    # -----------------------------------------
    # MEF Multi-exposure image Fusion
    # -----------------------------------------
    elif dataset_type in ['mef_GT', 'mff_GT']:
        from data.dataset_mef import DatasetMEF as D
    elif dataset_type in ['mef', 'vif', 'mff', 'nir', 'med']:
        from data.dataset_wogt import Dataset as D

    # -----------------------------------------
    # common
    # -----------------------------------------
    elif dataset_type in ['plain']:
        from data.dataset_plain import DatasetPlain as D

    elif dataset_type in ['plainpatch']:
        from data.dataset_plainpatch import DatasetPlainPatch as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
