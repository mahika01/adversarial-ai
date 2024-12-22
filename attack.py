from random import random
from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy
from torch.autograd import Variable
import torchvision.models as models

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        epsilon = 0.1,
        min_val = 0,
        max_val = 1
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            epsilon: magnitude of perturbation that is added
        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilon = 0.1
        self.min_val = 0
        self.max_val = 1


    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        original_images = original_images.to(self.device)
        # original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images.clone().detach()
       
        alpha = 2/255
        steps = 50
        perturbed_image = perturbed_image + (torch.rand(perturbed_image.size(), dtype=perturbed_image.dtype, device=perturbed_image.device)-0.5)*2*self.epsilon
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        
        for i in range(steps):
            grad = self.vm.get_batch_input_gradient(perturbed_image.data, target_labels)
            grad = grad.sign()
            perturbed_image = perturbed_image - alpha*grad
        
        perturbed_image = original_images + torch.clamp(perturbed_image-original_images, -self.epsilon, self.epsilon)
        perturbed_image = perturbed_image.detach()
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
    
        adv_outputs, detected_output  = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return perturbed_image.cpu().detach().numpy(), correct
