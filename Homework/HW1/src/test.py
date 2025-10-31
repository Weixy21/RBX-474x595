
import importlib
import torch
import torch.nn as nn

import networks as net
import pytest

def test_part1_linear():
    print("\nLinear")
    u = torch.rand((1, 10))
    customLayer = net.CustomLinear(10, 2)
    inbuiltLayer = nn.Linear(in_features=10, out_features=2)

    inbuiltLayer.weight.data.copy_(customLayer.weight.data)
    inbuiltLayer.bias.data.copy_(customLayer.bias.data)

    y_custom = customLayer(u)
    y_inbuilt = inbuiltLayer(u)

    assert torch.allclose(y_custom, y_inbuilt, rtol=1e-4)


def test_part1_linear_grad():
    print("\nLinear")
    u = torch.rand((1, 10))
    customLayer = net.CustomLinear(10, 2)
    inbuiltLayer = nn.Linear(in_features=10, out_features=2)

    inbuiltLayer.weight.data.copy_(customLayer.weight.data)
    inbuiltLayer.bias.data.copy_(customLayer.bias.data)

    y_custom = customLayer(u)
    y_inbuilt = inbuiltLayer(u)

    lossFunc = nn.MSELoss()

    loss_custom = lossFunc(y_custom, torch.zeros_like(y_custom))
    loss_in = lossFunc(y_inbuilt, torch.zeros_like(y_inbuilt))

    loss_custom.backward()
    loss_in.backward()
        
    assert torch.allclose(customLayer.bias.grad, inbuiltLayer.bias.grad, rtol=1e-4)
    assert torch.allclose(customLayer.weight.grad, inbuiltLayer.weight.grad, rtol=1e-4)


def test_part1_relu():
    # RELU
    print("\nRELU")
    u1 = torch.rand((1, 10), requires_grad=True)
    u2 = u1.detach().clone()
    u2.requires_grad_()

    customLayer = net.CustomReLU()
    inbuiltLayer = nn.ReLU()

    y_custom = customLayer(u1)
    y_inbuilt = inbuiltLayer(u2)

    assert torch.allclose(y_inbuilt, y_custom, rtol=1e-4)

    lossFunc = nn.MSELoss()
    loss_custom = lossFunc(y_custom, torch.zeros_like(y_custom))
    loss_in = lossFunc(y_inbuilt, torch.zeros_like(y_inbuilt))

    loss_custom.backward()
    loss_in.backward()

    assert torch.allclose(u1.grad, u2.grad, rtol=1e-4)

def test_part1_sm():
    # SOFTMAX
    print("\n SoftMax")

    u1 = torch.rand((1, 3), requires_grad=True)
    u2 = u1.detach().clone()
    u2.requires_grad_()
    customLayer = net.CustomSoftmax(1)
    inbuiltLayer = nn.Softmax()

    y_custom = customLayer(u1)
    y_inbuilt = inbuiltLayer(u2)

    lossFunc = nn.MSELoss()

    loss_custom = lossFunc(y_custom, torch.zeros_like(y_custom))
    loss_in = lossFunc(y_inbuilt, torch.zeros_like(y_inbuilt))

    loss_custom.backward()
    loss_in.backward()

    assert torch.allclose(u1.grad, u2.grad, rtol=1e-4)

def test_part2_mlp_val():
    pipeline = net.Pipeline()
    model = net.CustomMLP().to("cpu")
    model.load_state_dict(torch.load("./mlp.pth", map_location=torch.device('cpu')))

    val_acc = pipeline.val_step(model)

    assert val_acc > 40

def test_part3_cnn_torch_val():
    pipeline = net.Pipeline()
    model = net.RefCNN().to(pipeline.device)
    model.load_state_dict(torch.load("./cnn_inbuilt.pth", map_location=torch.device('cpu')))

    val_acc = pipeline.val_step(model)

    assert val_acc > 53

def test_part4_conv_inference():
    inbuiltLayer = nn.Conv2d(2, 3, 3, stride=2, padding='valid')
    customLayer = net.CustomConv2d(2, 3, 3, 2)

    inbuiltLayer.weight.data.copy_(customLayer.weight.data)
    inbuiltLayer.bias.data.copy_(customLayer.bias.data)

    u1 = torch.rand((1, 2, 5, 5), requires_grad=True)
    u2 = u1.detach().clone()
    u2.requires_grad_()

    y1 = inbuiltLayer(u1)
    y2 = customLayer(u2)

    assert torch.allclose(y1, y2, rtol=1e-4)


def test_part4_conv_gradient():
    inbuiltLayer = nn.Conv2d(2, 3, 3, stride=2, padding='valid')
    customLayer = net.CustomConv2d(2, 3, 3, 2)

    inbuiltLayer.weight.data.copy_(customLayer.weight.data)
    inbuiltLayer.bias.data.copy_(customLayer.bias.data)

    u1 = torch.rand((1, 2, 5, 5), requires_grad=True)
    u2 = u1.detach().clone()
    u2.requires_grad_()

    y1 = inbuiltLayer(u1)
    y2 = customLayer(u2)

    lossFunc = nn.MSELoss()
    loss_custom = lossFunc(y2, torch.zeros_like(y2))
    loss_in = lossFunc(y1, torch.zeros_like(y1))

    loss_in.backward()
    loss_custom.backward()

    assert torch.allclose(inbuiltLayer.weight.grad, customLayer.weight.grad, rtol=1e-4)
    assert torch.allclose(inbuiltLayer.bias.grad, customLayer.bias.grad, rtol=1e-4)
    assert torch.allclose(u1.grad, u2.grad, rtol=1e-4)

# pip install pytest-timeout
@pytest.mark.timeout(60) 
def test_part5_cnn_custom_val():
    pipeline = net.Pipeline()
    model = net.CustomCNN().to(pipeline.device)
    model.load_state_dict(torch.load("./cnn_custom.pth", map_location=torch.device('cpu')))

    val_acc = pipeline.val_step(model)

    assert val_acc > 53

pytest.main()




