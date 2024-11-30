from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    CropForegroundd,
    Spacingd,
    Lambdad,
    EnsureTyped,
    Invertd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    SpatialPadd,
    Transform,
    DeleteItemsd,
    MapTransform,
    RandCropByLabelClassesd,
    RandFlipd
)
import os, json, tempfile, glob, tqdm
from monai.data import Dataset, PersistentDataset, CacheDataset, DataLoader, decollate_batch
import torch
from torch.utils.data import RandomSampler

class Masking(Transform):
    def __init__(
        self,
        image_keys: list,
        mask_key: str,
        mask_value: float = None,
    ):
        self.image_keys = image_keys
        self.mask_key = mask_key
        self.mask_value = mask_value
        self.fill_min = True if mask_value is None else False
    def __call__(self, x: dict):
        if self.mask_key not in x.keys():
            return x
        for key in self.image_keys:
            if self.fill_min:
                fill_val = x[key][~(x[self.mask_key].bool())]
            else:
                fill_val = self.mask_value
            x[key][~(x[self.mask_key].bool())] = fill_val
        return x

class ForgroundMaskingd(MapTransform):
    def __init__(
        self,
        keys: list,
        threshold: float = 0,
        prefix: str = 'foreground_',
        allow_missing_keys: bool = True
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold
        self.prefix = prefix
    def __call__(self, x: dict):
        for key in self.keys:
            x[f"{self.prefix}{key}"] = (x[key] > self.threshold).bool()
        return x

class UnionMask(Transform):
    def __init__(
        self,
        keys: list,
        output_key: str = 'foreground'
    ):
        self.keys = keys
        self.output_key = output_key
    def __call__(self, x:dict):
        mask = x[self.keys[0]].bool()
        for key in self.keys[1:]:
            mask |= x[key].bool()
        x[self.output_key] = mask
        return x

class IntersectionMask(Transform):
    def __init__(
        self,
        keys: list,
        output_key: str = 'foreground'
    ):
        self.keys = keys
        self.output_key = output_key
    def __call__(self, x:dict):
        mask = x[self.keys[0]].bool()
        for key in self.keys[1:]:
            mask &= x[key].bool()
        x[self.output_key] = mask
        return x

class ZscoreNormalization(MapTransform):
    def __init__(
        self,
        keys: dict,
        allow_missing_keys: bool = False,
        apply_mask: bool = True,
        mask_key: str = 'mask',
        subtrahend: float = None,
        divisor: float = None,
        threshold: float = None
    ):
        super().__init__(keys, allow_missing_keys)
        self.apply_mask = apply_mask
        self.mask_key = mask_key
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.threshold = threshold
    def __call__(self, x:dict):
        for key in self.keys:
            if (self.subtrahend is not None) and (self.divisor is not None):
                mean = self.subtrahend
                std = self.divisor
            else:
                if self.apply_mask:
                    mean = x[key][x[self.mask_key].bool()].mean()
                    std = x[key][x[self.mask_key].bool()].std()
                elif self.threshold is not None:
                    mean = x[key][x[key]>self.threshold].mean()
                    std = x[key][x[key]>self.threshold].std()
                else:
                    mean = x[key].mean()
                    std = x[key].std()
            x[key] = (x[key] - mean) / std # assume no 0 std
            meta_key = f"{key}_meta_dict"
            if meta_key not in x.keys():
                x[meta_key] = {}
            x[meta_key]['zscore'] = {
                'mean': mean,
                'std': std
            }
        return x

def define_preprocessing(processing_type, params, keys: list = []):
    if processing_type == 'zscore_norm':
        return ZscoreNormalization(keys = keys, **params)
    elif processing_type == 'scale_intensity':
        return ScaleIntensityRanged(keys = keys, **params)
    elif processing_type == 'minmax':
        return ScaleIntensityd(keys = keys, **params)

def define_postprocessing(processing_type, params, keys: list = []):
    if processing_type == 'zscore_norm':
        return InverseZscoreNormalization(keys = keys, **params)
    elif processing_type == 'scale_intensity':
        a_min = params['a_min']
        a_max = params['a_max']
        b_min = 0
        b_max = 1
        if 'b_min' in params:
            b_min = params['b_min']
            b_max = params['b_max']
        return InverseIntensityScaleRanged(keys = keys, a_min = a_min, a_max = a_max, b_min = b_min, b_max = b_max)#Lambdad(keys = keys, func = lambda x: ((x + b_min) / (b_max - b_min)) * (a_max - a_min) + a_min, overwrite = False)
    elif processing_type == 'minmax':
        return InverseMinmax(keys = keys, **params)

def define_trans(opts):
    source_name = opts['data_opt']['source']
    # get configurations options
    data_opt = opts['data_opt']
    train_opt = opts['train_opt']

    source_key = train_opt['source_key']
    target_key = train_opt['target_key']
    keys = [source_key, target_key]
    if 'synthrad23' in source_name:
        keys += ['mask']
    pixdim = data_opt['pixdim']
    patch_size = train_opt['patch_size']
    num_patch = train_opt['num_patch']
    # get intensity transform
    source_processing_config = data_opt['preprocessing_config'][source_key]
    target_processing_config = data_opt['preprocessing_config'][target_key]
    trans_source = define_preprocessing(keys = [source_key], processing_type = source_processing_config['processing_type'], params = source_processing_config['params'])
    trans_target = define_preprocessing(keys = [target_key], processing_type = target_processing_config['processing_type'], params = target_processing_config['params'])
    # set threshold for foreground masking
    source_threshold = 0.1 if source_key in ('ct', 'cbct') else 0.1
    target_threshold = 0.1 if target_key in ('ct', 'cbct') else 0.1
    # set mask fill value
    source_mask_val = None if source_key in ('ct', 'cbct') else 0
    target_mask_val = None if target_key in ('ct', 'cbct') else 0
    # get list of transforms
    list_base_trans = [
        # loader
        LoadImaged(keys = keys),
        # ensure channel dimension
        EnsureChannelFirstd(keys = keys),
        # orientation
        # Orientationd(keys = keys, axcodes = 'RAS'),
        # foreground cropping
        CropForegroundd(keys = keys, source_key = source_key),
        CropForegroundd(keys = keys, source_key = target_key),
        # masking - if mask is given (for synthrad23)
        Masking(image_keys = [source_key], mask_key = 'mask', mask_value = source_mask_val),
        Masking(image_keys = [target_key], mask_key = 'mask', mask_value = target_mask_val),
        # spatial transform
        Spacingd(keys = keys, pixdim = pixdim, mode = 'trilinear'),
        trans_source,
        trans_target,
        # ScaleIntensityd(keys = keys, minv = 0, maxv = 1),
        SpatialPadd(keys = keys, spatial_size = patch_size),
        ForgroundMaskingd(keys = [source_key], threshold = source_threshold, prefix = 'foreground_'),
        ForgroundMaskingd(keys = [target_key], threshold = target_threshold, prefix = 'foreground_'),
        IntersectionMask(keys = [f'foreground_{source_key}', f'foreground_{target_key}'], output_key = 'foreground'),
        DeleteItemsd(keys = [f'foreground_{source_key}', f'foreground_{target_key}']),
    ]
    list_train_trans = list_base_trans + [
        # dummy random transform
        RandFlipd(keys = keys + ['foreground', 'mask'], prob = 0, spatial_axis = [0], allow_missing_keys = True),
        SpatialPadd(keys = keys + ['foreground', 'mask'], spatial_size = patch_size, allow_missing_keys = True),
        RandCropByLabelClassesd(keys = keys + ['foreground', 'mask'], label_key = 'foreground', ratios = [0, 1], num_classes = 2, num_samples = num_patch, spatial_size = patch_size, allow_missing_keys = True)
    ]
    list_val_trans = list_base_trans
    # define transforms
    trans_train = Compose(list_train_trans)
    trans_eval = Compose(list_val_trans)
    trans_cache = Compose(list_base_trans)
    return trans_train, trans_eval, trans_cache

def define_dataloaders(data_dir, opts, args):
    # load configurations
    data_opt = opts['data_opt']
    dataset_name = data_opt['source']
    fold = args.fold
    cv = opts['train_opt']['cv']
    num_workers = args.num_workers
    memory_cache = args.memory_cache
    persistent_cache = args.persistent
    max_num_val = opts['train_opt']['max_num_val']
    source_key = opts['train_opt']['source_key']
    target_key = opts['train_opt']['target_key']
    batch_size = opts['train_opt']['batch_size']
    num_patch = opts['train_opt']['num_patch']
    step_size = opts['train_opt']['step_size']

    cache_dir_base = os.path.join(data_dir, 'persistent')
    location_map = {}
    if os.path.exists(os.path.join(cache_dir_base, 'location_map.json')):
        location_map = json.load(open(os.path.join(cache_dir_base, 'location_map.json'), 'r'))
    
    # load file list
    meta = json.load(open(os.path.join(data_dir, 'meta', f'{dataset_name}.json')))
    list_files = [{
        f'fileloc_{target_key}': fdict[target_key], 
        f'fileloc_{source_key}': fdict[source_key], 
        **{key: os.path.join(data_dir, val) for key,val in fdict.items()}
    } for fdict in meta['files']]
    # split files
    list_files_dev = [file for idx, file in enumerate(list_files) if idx % cv != fold]
    n_val = min(int(len(list_files_dev) * 0.1), max_num_val)
    list_files_train = list_files_dev[n_val:]
    list_files_val = list_files_dev[:n_val]
    list_files_test = [file for idx, file in enumerate(list_files) if idx % cv == fold]
    # define transforms
    trans_train, trans_eval, trans_cache = define_trans(opts)
    # load datasets
    if memory_cache:
        datasets = {
            'train': CacheDataset(list_files_train, transform = trans_train, num_workers = num_workers),
            'val': CacheDataset(list_files_val, transform = trans_eval, num_workers = num_workers),
            'test': CacheDataset(list_files_test, transform = trans_eval, num_workers = num_workers),
        }
    elif persistent_cache:
        cache_dir = os.path.join(data_dir, 'persistent', dataset_name)
        source_key = opts['train_opt']['source_key']
        target_key = opts['train_opt']['target_key']
        if dataset_name in ('hcp1200', 'dhcp'):
            hash_func = lambda x: x[source_key].split('/')[-3].encode()
            # .strip('/').strip('.nii.gz').replace('/', '@').encode()
        elif 'synthrad23' in dataset_name:
            hash_func = lambda x: x[source_key].split('/')[-2].encode()
        elif 'brats21':
            hash_func = lambda x: x[source_key].split('/')[-2].encode()
        elif 'brainmetshare':
            hash_func = lambda x: x[source_key].split('/')[-2].encode()

        persistent_exist = False
        for dirname, ref_opt in location_map.items():
            if data_opt == ref_opt:
                persistent_exist = True
                cache_dir = os.path.join(cache_dir_base, dirname)
        if not persistent_exist:
            while True:
                dirname = os.path.basename(tempfile.mkdtemp())
                cache_dir = os.path.join(cache_dir_base, dirname)
                if not os.path.exists(cache_dir):
                    break
                print(f'Persistent files do not exist, making one in {cache_dir}')
        else:
            print(f'Persistent files exist in {cache_dir}')
        # make persistent cache files if 
        list_files_persistent = glob.glob(os.path.join(cache_dir, '*.pt'))
        if len(list_files_persistent) != len(list_files):
            location_map[dirname] = data_opt
            print('Making persistent cache files...')
            torch.multiprocessing.set_sharing_strategy('file_system')
            ds = PersistentDataset(list_files, transform = trans_cache, cache_dir=cache_dir, hash_func=hash_func)
            dl = DataLoader(ds, num_workers = num_workers)
            pbar = tqdm.tqdm(total = len(dl), position = 0)
            for idx, d in enumerate(dl):
                pbar.update(1)
            pbar.close()
            json.dump(location_map, open(os.path.join(cache_dir_base, 'location_map.json'), 'w'))
        datasets = {
            'train': PersistentDataset(list_files_train, transform = trans_train, cache_dir = cache_dir, hash_func = hash_func),
            'val': PersistentDataset(list_files_val, transform = trans_eval, cache_dir = cache_dir, hash_func = hash_func),
            'test': PersistentDataset(list_files_test, transform = trans_eval, cache_dir = cache_dir, hash_func = hash_func),
        }
    else:
        datasets = {
            'train': Dataset(list_files_train, transform = trans_train),
            'val': Dataset(list_files_val, transform = trans_eval),
            'test': Dataset(list_files_test, transform = trans_eval),
        }
    # get dataloader
    sampler = RandomSampler(datasets['train'], replacement = True, num_samples = step_size * batch_size)
    dataloaders = {
        'train': DataLoader(datasets['train'], num_workers = num_workers, sampler = sampler, batch_size = batch_size),
        'val': DataLoader(datasets['val'], num_workers = num_workers, batch_size = 1),
        'test': DataLoader(datasets['test'], num_workers = num_workers, batch_size = 1),
    }
    return dataloaders
