import argparse
import logging
import os
import random
from pydoc import locate
import sys
from datasets.dataset_multiphase import Multiphase_dataset
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import importlib
from torch.utils.data import DataLoader
from datasets.dataset_synapse import Synapse_dataset
from trainer.trainer_phase import  inference

#todo: 动态导入测试数据集，动态导入测试模型

parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/",
    help="root dir for validation volume data",
)  # for acdc volume_path=root_dir
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=3, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--is_savenii", action="store_true", help="whether to save results during inference")
parser.add_argument("--test_save_dir", type=str, default="./model_out/test", help="saving prediction as nii!")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=42, help="random seed")
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs="+",
)

parser.add_argument("--list_dir", type=str, default="./lists/lists_liver", help="list dir")
parser.add_argument("--inference",default="inference",help="inference style, default is random_crop")
parser.add_argument("--gpu",default="1",help="gpu id")
parser.add_argument("--crop_inference",default="random_crop_inference",\
                    help="style of inference: resize_inference, standard_crop_inference, random_crop_inference")
parser.add_argument(
    "--test_path",
    type=str,
    default="/data/3DUNET_Github/multi_phase/processed/test_source",
    help="root dir for test data",
)
parser.add_argument(
    "--checkpoint",default='/data/3DUNET_Github/multi_phase/model_out/Fusion_Multiphase_epoch_4_lr_0.00011.pth' ,help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--module", help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)
# parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
# parser.add_argument("--resume", help="resume from checkpoint")
# parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")

def dynamic_import(module_name, function_name):
    # 导入指定的模块
    module = importlib.import_module(module_name)
    # 获取模块中的函数
    function = getattr(module, function_name)
    return function



args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "z_spacing": 1,
        },
        "Multiphase": {
            "Dataset": Multiphase_dataset,
            "z_spacing": 1,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    net = locate(args.module)
    net = net(args.num_classes, 1, 64).cuda()

    snapshot = os.path.join(args.output_dir, args.checkpoint)

    if args.checkpoint:
        print("use checkpoint:", args.checkpoint)
        state_dict = torch.load(snapshot)['state_dict']
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        msg=net.load_state_dict(new_state_dict)
        print('have loaded:',msg)

    snapshot_name = snapshot.split("/")[-1]

    log_folder = args.output_dir+ "/test_log"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        filemode='w'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "test")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    assert args.crop_inference is not None
    # print(f'inference_single_volume_style: {args.crop_inference}')
    logging.info('inference_single_volume_style:%s',args.crop_inference)
    args.crop_inference = dynamic_import("utils.utils_multiphase", args.crop_inference)

    # laod dataset and inference

    db_test = Multiphase_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir,
                                 img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    inference(net, testloader, args, test_save_path)
