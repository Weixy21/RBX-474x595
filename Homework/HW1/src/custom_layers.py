import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias):
        # weight shape - output x input dimension
        # bias shape - output dimension

        # implement y = x (mult) w_transpose + b

        # YOUR IMPLEMENTATION HERE!
        # output=1 is a placeholder
        output = 1

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        # save the tensors required for back pass
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Shapes.
        # grad_output - batch x output_count
        # grad_input  - batch x input
        # grad_weight - output x input
        # grad_bias   - output shape

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # YOUR IMPLEMENTATION HERE!

        # use either print or logger to print its outputs.
        # make sure you disable before submitting
        # print(grad_input)
        # logger.info("grad_output: %s", grad_bias.shape)

        return grad_input, grad_weight, grad_bias
    
class CustomReLULayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # YOUR IMPLEMENTATION HERE!
        output = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = grad_output.clone()
        # YOUR IMPLEMENTATION HERE!
        
        return grad_input


class CustomSoftmaxLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

        # YOUR IMPLEMENTATION HERE!
        softmax_output = 1

        ctx.save_for_backward(softmax_output)
        ctx.dim = dim

        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, = ctx.saved_tensors
        dim = ctx.dim

        # YOUR IMPLEMENTATION HERE!
        grad_input = 1

        # print(grad_input.shape)
        return grad_input, None

class CustomConvLayer(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias, stride, kernel_size):
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # implement the cross correlation filter
        
        # weight shape - out_ch x in_ch x kernel_width x kernel_height
        # bias shape - out_ch
        # input shape - batch x ch x width x height
        # out shape - batch x out_ch x width //stride x height //stride
        
        # You can assume the following,
        #  no padding
        #  kernel width == kernel height
        #  stride is identical along both axes

        out_ch = weight.shape[0]
        in_ch = weight.shape[1]

        kernel = kernel_size
        
        batch, _, height, width = input.shape

        # YOUR IMPLEMENTATION HERE!
        output = 1


        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, stride, kernel_size = inputs
        # save the tensors required for back pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride

    @staticmethod
    def backward(ctx, grad_output):
        # grad output shape - batch x out_dim x out_width x out_height (strided)
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        grad_input = grad_weight = grad_bias = None

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        out_ch = weight.shape[0]
        in_ch = weight.shape[1]

        kernel = weight.shape[2]
        
        batch, _, height, width = input.shape

        # YOUR IMPLEMENTATION HERE!

        return grad_input, grad_weight, grad_bias, None, None