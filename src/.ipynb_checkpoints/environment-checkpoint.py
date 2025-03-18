
import torch
############ For Environment ###############
speed_of_light = 343

class linear_array():

    def __init__(self,M,d):
        # Initializes the geometry
        self.c = speed_of_light
        self.M = M # n of sensors
        self.d = d # Spacing bw each
        self.pos = torch.Tensor([i*d for i in range(M)]).view(1,-1) #(1,M)
        self.thetas = []
        self.signals = []

    def eval_svec(self,theta):
        

