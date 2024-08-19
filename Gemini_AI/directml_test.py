import torch
import torch_directml

# Set the device to the default DirectML device
device = torch_directml.device(torch_directml.default_device())

# Create a simple tensor
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Move the tensor to the DirectML device (GPU)
tensor_on_device = tensor.to(device)

# Print the device of the tensor to verify where it's located
print(f"Tensor is on device: {tensor_on_device.device}")

# Check if the tensor is on a DirectML device (which is GPU)
if "privateuseone" in str(tensor_on_device.device):
    print("The tensor is using the GPU (DirectML).")
else:
    print("The tensor is not using the GPU.")
