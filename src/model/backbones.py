import torch
import torch.nn as nn
import torchvision
import timm
from torchvision.models._utils import IntermediateLayerGetter


class ResNetBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    
    def __init__(self, name='resnet50', pretrained=True, trainable_layers=3):
        super().__init__()
        
        # Get the specified ResNet model
        if name == 'resnet18':
            backbone = torchvision.models.resnet18(weights='DEFAULT' if pretrained else None)
            out_channels = 512
        elif name == 'resnet34':
            backbone = torchvision.models.resnet34(weights='DEFAULT' if pretrained else None)
            out_channels = 512
        elif name == 'resnet50':
            backbone = torchvision.models.resnet50(weights='DEFAULT' if pretrained else None)
            out_channels = 2048
        elif name == 'resnet101':
            backbone = torchvision.models.resnet101(weights='DEFAULT' if pretrained else None)
            out_channels = 2048
        else:
            raise ValueError(f"Invalid ResNet model name: {name}")
            
        # Freeze BatchNorm layers
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                    
        # Freeze layers based on trainable_layers parameter
        assert 0 <= trainable_layers <= 5
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        
        # Freeze all layers first
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad_(False)
            
        # Then unfreeze selected layers
        for layer_name in layers_to_train:
            for name, parameter in backbone.named_parameters():
                if layer_name in name:
                    parameter.requires_grad_(True)
        
        # Define which layers to return
        return_layers = {'layer4': 'feat'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.out_channels = out_channels
        
    def forward(self, x):
        return self.backbone(x)


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone."""
    
    def __init__(self, name='efficientnet_b3', pretrained=True, trainable_layers=3):
        super().__init__()
        
        # Get the specified EfficientNet model
        if name == 'efficientnet_b0':
            backbone = torchvision.models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            out_channels = 1280
        elif name == 'efficientnet_b1':
            backbone = torchvision.models.efficientnet_b1(weights='DEFAULT' if pretrained else None)
            out_channels = 1280
        elif name == 'efficientnet_b2':
            backbone = torchvision.models.efficientnet_b2(weights='DEFAULT' if pretrained else None)
            out_channels = 1408
        elif name == 'efficientnet_b3':
            backbone = torchvision.models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
            out_channels = 1536
        else:
            raise ValueError(f"Invalid EfficientNet model name: {name}")
        
        # Remove classifier head
        self.features = backbone.features
        self.out_channels = out_channels
        
        # Control which layers to train
        trainable_params = list(self.features.parameters())
        total_layers = len(trainable_params)
        freeze_layers = total_layers - (total_layers // 5 * trainable_layers)
        
        # Freeze early layers
        for i, param in enumerate(trainable_params):
            if i < freeze_layers:
                param.requires_grad = False
                
    def forward(self, x):
        x = self.features(x)
        # Return a dict to match ResNet output format
        return {'feat': x}


class SwinTransformerBackbone(nn.Module):
    """Swin Transformer backbone."""
    
    def __init__(self, name='swin_tiny_patch4_window7_224', pretrained=True, trainable_layers=3):
        super().__init__()
        
        # Load Swin Transformer from timm
        self.model = timm.create_model(
            name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(3,)  # Last stage output
        )
        
        # Set output channels
        if name == 'swin_tiny_patch4_window7_224':
            self.out_channels = 768
        elif name == 'swin_small_patch4_window7_224':
            self.out_channels = 768
        elif name == 'swin_base_patch4_window7_224':
            self.out_channels = 1024
        else:
            raise ValueError(f"Invalid Swin Transformer name: {name}")
        
        # Control trainable layers
        param_list = list(self.model.parameters())
        total_layers = len(param_list)
        layers_to_freeze = max(0, total_layers - (total_layers // 4 * trainable_layers))
        
        for i, param in enumerate(param_list):
            if i < layers_to_freeze:
                param.requires_grad = False
                
    def forward(self, x):
        x = self.model(x)[0]  # Get the last stage output
        return {'feat': x}


def build_backbone(config):
    """Build backbone based on config."""
    backbone_name = config["backbone"]
    
    if backbone_name.startswith('resnet'):
        return ResNetBackbone(name=backbone_name, pretrained=True, trainable_layers=3)
    elif backbone_name.startswith('efficientnet'):
        return EfficientNetBackbone(name=backbone_name, pretrained=True, trainable_layers=3)
    elif backbone_name.startswith('swin'):
        return SwinTransformerBackbone(name=backbone_name, pretrained=True, trainable_layers=3)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")