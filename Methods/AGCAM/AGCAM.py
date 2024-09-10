import torch
from einops.layers.torch import Reduce, Rearrange

class AGCAM:

    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum'):

        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)
                
    def get_attn_matrix(self, module, input, output):
        self.attn_matrix.append(output[:, :, 0:1, :])
        


    def get_grad_attn(self, module, grad_input, grad_output):
        self.grad_attn.append(grad_output[0][:, :, 0:1, :])
        
    
    def generate(self, input_tensor, cls_idx=None):
        self.attn_matrix = []
        self.grad_attn = []

        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        self.prediction = prediction  
        if cls_idx==None:
            loss = output[0, prediction[0]]
        else:
            loss = output[0, cls_idx]
        loss.backward()

        b, h, n, d = self.attn_matrix[0].shape
        self.head=h
        self.width = int((d-1)**0.5)

        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        gradient = self.grad_attn[0]
        for i in range(1, len(self.attn_matrix)):
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        gradient = torch.nn.functional.relu(gradient)
        attn = torch.sigmoid(attn)
        mask = gradient * attn

        mask = mask[:, :, :, 1:].unsqueeze(0)
        mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
        mask = Reduce('b l z p -> b z p', reduction=self.layer_fusion)(mask)
        mask = Rearrange('b z (h w) -> b z h w', h=self.width, w=self.width)(mask)
        return prediction, mask

