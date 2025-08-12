#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.models.utae_paps_models.utae import UTAE
import torchvision.models as models

resnet = models.resnet34(pretrained=True)

utae = UTAE(
            input_dim=7,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
        )


# In[2]:


utae


# In[3]:


resnet


# In[4]:


print("ResNet layer1 conv1 shape:", resnet.layer1[0].conv1.weight.shape)
print("UTAE down_blocks[0] conv1 shape:", utae.down_blocks[0].conv1.conv[0].weight.shape)


# In[5]:


import torch
import torch.nn as nn

def load_resnet_weights_to_utae(resnet, utae_model):
    loaded_layers = 0
    loaded_weights = []

    # Mapping resnet layers to UTAE down_blocks
    resnet_layer1 = resnet.layer1  # Equivalent to down_blocks[0] and down_blocks[1] in UTAE
    resnet_layer2 = resnet.layer2  # Equivalent to down_blocks[2] in UTAE

    # Transfer weights from ResNet's layer1 to UTAE down_blocks[0] and down_blocks[1]
    for i in range(2):  # down_blocks[0] and down_blocks[1]
        res_block = resnet_layer1[i]  # ResNet BasicBlock
        utae_block = utae_model.down_blocks[i]  # UTAE DownConvBlock

        # Skip the 'down' ConvLayer in UTAE since it's not equivalent
        # Load weights for conv1
        if res_block.conv1.weight.shape == utae_block.conv1.conv[0].weight.shape:
            utae_block.conv1.conv[0].weight.data = res_block.conv1.weight.data
            print(f"ResNet layer: layer1[{i}].conv1 --> UTAE layer: down_blocks[{i}].conv1")
            loaded_weights.append(f'down_blocks[{i}].conv1')
            loaded_layers += 1
            if hasattr(res_block.conv1, "bias") and res_block.conv1.bias is not None:
                utae_block.conv1.conv[0].bias.data = res_block.conv1.bias.data

        # Load weights for conv2
        if res_block.conv2.weight.shape == utae_block.conv2.conv[0].weight.shape:
            utae_block.conv2.conv[0].weight.data = res_block.conv2.weight.data
            print(f"ResNet layer: layer1[{i}].conv2 --> UTAE layer: down_blocks[{i}].conv2")
            loaded_weights.append(f'down_blocks[{i}].conv2')
            loaded_layers += 1
            if hasattr(res_block.conv2, "bias") and res_block.conv2.bias is not None:
                utae_block.conv2.conv[0].bias.data = res_block.conv2.bias.data

    # Transfer weights from ResNet's layer2 to UTAE down_blocks[2]
    res_block = resnet_layer2[0]  # First block in layer2
    utae_block = utae_model.down_blocks[2]  # UTAE DownConvBlock for layer2 equivalent

    # Skip the 'down' ConvLayer in UTAE since it has different stride and kernel size
    # Load weights for conv1 (64 -> 128)
    if res_block.conv1.weight.shape == utae_block.conv1.conv[0].weight.shape:
        utae_block.conv1.conv[0].weight.data = res_block.conv1.weight.data
        print(f"ResNet layer: layer2[0].conv1 --> UTAE layer: down_blocks[2].conv1")
        loaded_weights.append(f'down_blocks[2].conv1')
        loaded_layers += 1
        if hasattr(res_block.conv1, "bias") and res_block.conv1.bias is not None:
            utae_block.conv1.conv[0].bias.data = res_block.conv1.bias.data

    # Load weights for conv2 (128 -> 128)
    if res_block.conv2.weight.shape == utae_block.conv2.conv[0].weight.shape:
        utae_block.conv2.conv[0].weight.data = res_block.conv2.weight.data
        print(f"ResNet layer: layer2[0].conv2 --> UTAE layer: down_blocks[2].conv2")
        loaded_weights.append(f'down_blocks[2].conv2')
        loaded_layers += 1
        if hasattr(res_block.conv2, "bias") and res_block.conv2.bias is not None:
            utae_block.conv2.conv[0].bias.data = res_block.conv2.bias.data

    print(f"Total layers loaded: {loaded_layers}")
    print("Loaded weights:")
    for weight in loaded_weights:
        print(weight)

# Example usage:
# resnet = torchvision.models.resnet34(pretrained=True)
# load_resnet_weights_to_utae(resnet, utae_model)


# In[11]:


# Initialize UTAE model
utae_model = UTAE(input_dim=7)

# Load pretrained ResNet
resnet = models.resnet34(pretrained=True)

# load weights:
load_resnet_weights_to_utae(resnet, utae)


# In[9]:


import torch
random_input = torch.randn(1, 5, 7, 128, 128)  
batch_positions = torch.arange(5).unsqueeze(0).repeat(1, 1) 

utae_model.eval()  
with torch.no_grad(): 
    output = utae_model(random_input, batch_positions)

print("Output shape:", output.shape)


# In[13]:


def calculate_loaded_percentage_in_encoder(resnet, utae_model):
    total_params = 0
    loaded_params = 0

    # Mapping resnet layers to UTAE down_blocks
    resnet_layer1 = resnet.layer1  # Equivalent to down_blocks[0] and down_blocks[1] in UTAE
    resnet_layer2 = resnet.layer2  # Equivalent to down_blocks[2] in UTAE

    # For down_blocks[0] and down_blocks[1], matching ResNet's layer1
    for i in range(2):  # down_blocks[0] and down_blocks[1]
        res_block = resnet_layer1[i]  # ResNet BasicBlock
        utae_block = utae_model.down_blocks[i]  # UTAE DownConvBlock

        # Total parameters in UTAE conv1 and conv2
        total_params += utae_block.conv1.conv[0].weight.numel()
        total_params += utae_block.conv2.conv[0].weight.numel()

        # Check if ResNet weights can be loaded
        if res_block.conv1.weight.shape == utae_block.conv1.conv[0].weight.shape:
            loaded_params += res_block.conv1.weight.numel()
        if res_block.conv2.weight.shape == utae_block.conv2.conv[0].weight.shape:
            loaded_params += res_block.conv2.weight.numel()

    # For down_blocks[2], matching ResNet's layer2
    res_block = resnet_layer2[0]  # First block in layer2
    utae_block = utae_model.down_blocks[2]  # UTAE DownConvBlock for layer2 equivalent

    # Total parameters in UTAE conv1 and conv2
    total_params += utae_block.conv1.conv[0].weight.numel()
    total_params += utae_block.conv2.conv[0].weight.numel()

    # Check if ResNet weights can be loaded
    if res_block.conv1.weight.shape == utae_block.conv1.conv[0].weight.shape:
        loaded_params += res_block.conv1.weight.numel()
    if res_block.conv2.weight.shape == utae_block.conv2.conv[0].weight.shape:
        loaded_params += res_block.conv2.weight.numel()

    # Compute the percentage of weights loaded
    percentage_loaded = (loaded_params / total_params) * 100
    print(f"Total weights in encoder: {total_params}")
    print(f"Loaded weights in encoder: {loaded_params}")
    print(f"Percentage of weights loaded in encoder: {percentage_loaded:.2f}%")

# Example usage:
# resnet = torchvision.models.resnet34(pretrained=True)
# calculate_loaded_percentage_in_encoder(resnet, utae_model)


# In[15]:


calculate_loaded_percentage_in_encoder(resnet, utae_model)


# In[ ]:




