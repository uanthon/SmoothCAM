import torch
from einops.layers.torch import Reduce, Rearrange
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as T


class SmoothCAM:


    def __init__(self, model, attention_matrix_layer='before_softmax', attention_grad_layer='after_softmax',
                 head_fusion='sum', layer_fusion='sum', smooth_grad_n=10, smooth_grad_sigma=0.1, magnitude=False):

        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.smooth_grad_n = smooth_grad_n
        self.smooth_grad_sigma = smooth_grad_sigma
        self.magnitude = magnitude

        for name, module in self.model.named_modules():
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)

    def get_attn_matrix(self, module, input, output):
        if not hasattr(self, 'attn_matrix'):
            self.attn_matrix = []
        self.attn_matrix.append(output[:, :, 0:1, :])

        b, h, n, d = self.attn_matrix[0].shape
        self.head = h
        self.width = int((d - 1) ** 0.5)

    def get_grad_attn(self, module, grad_input, grad_output):
        if not hasattr(self, 'grad_attn'):
            self.grad_attn = []
        self.grad_attn.append(grad_output[0][:, :, 0:1, :])

    def generate(self, input_tensor, cls_idx=None):
        input_tensor = input_tensor.repeat(self.smooth_grad_n, 1, 1, 1)

        x = input_tensor.data.cpu().numpy()
        stdev = self.smooth_grad_sigma * (np.max(x) - np.min(x))
        noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
        x_plus_noise = x + noise
        x_plus_noise[0] = input_tensor[0].cpu()
        x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(input_tensor.device), requires_grad=True)

        outputs = self.model(x_plus_noise)

        one_hot = torch.zeros(self.smooth_grad_n, outputs.size()[-1], device=outputs.device)
        one_hot.scatter_(1, outputs.argmax(dim=1).view(-1, 1), 1)

        one_hot = one_hot.requires_grad_(True)
        loss = torch.sum(one_hot * outputs, dim=1)
        loss.backward(torch.ones_like(loss))

        attn_matrices = self.attn_matrix
        grad_attns = self.grad_attn
        self.attn_matrix = []
        self.grad_attn = []
        mask_list=[]

        for i in range(len(attn_matrices[0])):
            attn_matrices.reverse()
            attn_per_sample = []
            grad_per_sample = []
            for j in range(len(attn_matrices)):
                attn_per_sample.append(attn_matrices[j][i])
                grad_per_sample.append(grad_attns[j][i])
            attn = torch.stack(attn_per_sample, dim=0)
            gradient = torch.stack(grad_per_sample, dim=0)
            gradient = torch.nn.functional.relu(gradient)
            attn = torch.sigmoid(attn)
            mask = gradient * attn
            mask = mask[:, :, :, 1:].unsqueeze(0)

            mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
            mask = Reduce('b l z p -> b z p', reduction=self.layer_fusion)(mask)
            mask = Rearrange('b z (h w) -> b z h w', h=self.width, w=self.width)(mask)

            mask_list.append(mask)

        all_masks = torch.cat(mask_list, dim=0)
        mean_mask = torch.mean(all_masks, dim=0)

        return outputs.argmax(dim=1)[0], mean_mask