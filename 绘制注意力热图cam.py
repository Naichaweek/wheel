import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from  vgg_2iput import BCNN,SE_block


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='attention_test/intermediate.jpg',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam++',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='atten',
                        help='Output directory to save the images')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python 绘制注意力热图cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }
    # 这里是获取有几个目标层的列表
    # model = models.resnet50(pretrained=True)
    # for name, layer in model.named_modules():
    #     print(name)
    # print("--------------------------------------------")
    # target_layers = [model.layer4]
    # print(target_layers)

    # 这里是获取一个目标层
    # model = models.mnasnet1_0(pretrained=True)
    # print(model)
    # print("--------------------------------------------")
    # target_layers = [model.layers[-3]]
    # print(target_layers)

    # 这里是获取一个目标层
    # model = models.vgg11(pretrained=True)
    # print(model)
    # print("--------------------------------------------")
    # target_layers = [model.features[-3]]
    # print(target_layers)

    # 这里是获取一个目标层
    model = BCNN()
    print(model)
    print("--------------------------------------------")
    target_layers = [model.se1]
    print(target_layers)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer你可以打印模型来帮助选择图层
    # You can pass a list with several target layers,你可以传递一个有几个目标层的列表，
    # in that case the CAMs will be computed per layer and then aggregated.在这种情况下，cam将按层计算，然后聚合。
    # You can also try selecting all layers of a certain type, with e.g:你也可以尝试选择某种类型的所有图层，例如:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # 我们必须指定要为之生成类激活映射的目标。
    # 如果targets为None，则将使用评分最高的类别(对于批处理中的每个成员)。
    # You can target specific categories by
    # e.g ：targets = [ClassifierOutputTarget(281)]
    targets = None

    # 选择cam算法，使用with语句确保上下文被释放，并且可以在循环中重新创建不同的CAM对象。
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:


        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)
