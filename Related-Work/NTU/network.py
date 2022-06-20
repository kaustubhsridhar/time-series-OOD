import numpy as np
import torch
import torch.nn as nn

class Bi3DOF(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(Bi3DOF, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        self.nz = args.nz
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def encode(self, x, nsamples=1):
        return self.encoder.encode(x, nsamples)

    def loss(self, x, kl_weight, nsamples=1):
        z, (kl_grp1, kl_grp2)  = self.encode(x, nsamples)        
        reconstruct_err = self.decoder.reconstruct_error(x, z)
        return 0.5*reconstruct_err + kl_weight*(kl_grp1+kl_grp2), reconstruct_err, (kl_grp1, kl_grp2)


class GaussianEncoderBase(nn.Module):
    def __init__(self, args):
        super(GaussianEncoderBase, self).__init__()

        self.device = args.device
        self.latentprior = args.latentprior
        self.mu1 = args.mu1
        self.mu2 = args.mu2
        self.var1 = args.var1 * np.ones(args.nz)
        self.var1 = torch.from_numpy(self.var1).float().to(args.device) 
        self.var2 = args.var2 * np.ones(args.nz)
        self.var2 = torch.from_numpy(self.var2).float().to(args.device)

    def forward(self, x):
        raise NotImplementedError

    def encode(self, input, nsamples):
        # grp1 for horizontal and grp2 for vertical 
        (mu_grp1, logvar_grp1), (mu_grp2, logvar_grp2) = self.forward(input)
        z_grp1 = self.reparameterize(mu_grp1, logvar_grp1, nsamples)
        z_grp2 = self.reparameterize(mu_grp2, logvar_grp2, nsamples)     
        if self.latentprior == "optimal":
            priorvar_grp1 = self.var1 * torch.ones(logvar_grp1.size()).to(self.device)
            d_grp1 = ( (mu_grp1-self.mu1).pow(2) + logvar_grp1.exp() + priorvar_grp1 - \
                            2.*torch.sqrt(logvar_grp1.exp()*priorvar_grp1) ).sum(dim=1)
            priorvar_grp2 = self.var2 * torch.ones(logvar_grp2.size()).to(self.device)
            d_grp2 = ( (mu_grp2-self.mu2).pow(2) + logvar_grp2.exp() + priorvar_grp2 - \
                            2.*torch.sqrt(logvar_grp2.exp()*priorvar_grp2) ).sum(dim=1)
        else:
            d_grp1 = 0.5 * (mu_grp1.pow(2) + logvar_grp1.exp() - logvar_grp1 - 1).sum(dim=1)
            d_grp2 = 0.5 * (mu_grp2.pow(2) + logvar_grp2.exp() - logvar_grp2 - 1).sum(dim=1)

        return (z_grp1, z_grp2), (d_grp1, d_grp2)


    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)


class Encoder(GaussianEncoderBase):

    def __init__(self, args):
        super(Encoder, self).__init__(args) 
        self.group = 2  
        self.nz = args.nz
        self.nd = args.nd
        self.hiddenunits = 512
        args.tpu = False

        self.grp1_conv1 = nn.Conv2d(self.nd, 32, (5,5), stride=(3,3), padding=0, bias=False)  
        self.grp1_conv1_bn = nn.BatchNorm2d(32)      
        self.grp1_conv1_ac = nn.ReLU() if args.tpu else nn.ELU()
        self.grp1_conv2 = nn.Conv2d(32, 64, (5,5), stride=(3,3), padding=(2,0), bias=False)
        self.grp1_conv2_bn = nn.BatchNorm2d(64)         
        self.grp1_conv2_ac = nn.ReLU() if args.tpu else nn.ELU()
        self.grp1_conv3 = nn.Conv2d(64, 128, (5,5), stride=(3,3), padding=(2,2), bias=False)
        self.grp1_conv3_bn = nn.BatchNorm2d(128) 
        self.grp1_conv3_ac = nn.ReLU() if args.tpu else nn.ELU()
        self.grp1_conv4 = nn.Conv2d(128, 256, (5,5), stride=(1,1), padding=(0,0), bias=False)
        self.grp1_conv4_bn = nn.BatchNorm2d(256) 
        self.grp1_conv4_ac = nn.ReLU() if args.tpu else nn.ELU()   
        self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.nz) 
           
        self.grp2_conv1 = nn.Conv2d(self.nd, 32, (5,5), stride=(3,3), padding=0,  bias=False)  
        self.grp2_conv1_bn = nn.BatchNorm2d(32)       
        self.grp2_conv1_ac = nn.ReLU() if args.tpu else nn.ELU()           
        self.grp2_conv2 = nn.Conv2d(32, 64, (5,5), stride=(3,3), padding=(2,0), bias=False)
        self.grp2_conv2_bn = nn.BatchNorm2d(64)
        self.grp2_conv2_ac = nn.ReLU() if args.tpu else nn.ELU()
        self.grp2_conv3 = nn.Conv2d(64, 128, (5,5), stride=(3,3), padding=(2,2), bias=False)
        self.grp2_conv3_bn = nn.BatchNorm2d(128)
        self.grp2_conv3_ac = nn.ReLU() if args.tpu else nn.ELU()
        self.grp2_conv4 = nn.Conv2d(128, 256, (5,5), stride=(1,1), padding=(0,0), bias=False)
        self.grp2_conv4_bn = nn.BatchNorm2d(256)
        self.grp2_conv4_ac = nn.ReLU() if args.tpu else nn.ELU()  
        self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.nz)     

        self.init_parameters()


    def init_parameters(self):
        nn.init.normal_(self.grp1_conv1.weight, 0, np.sqrt(2./(5*5*32)) )
        nn.init.normal_(self.grp1_conv2.weight, 0, np.sqrt(2./(5*5*64)) )
        nn.init.normal_(self.grp1_conv3.weight, 0, np.sqrt(2./(5*5*128)) )
        nn.init.normal_(self.grp1_conv4.weight, 0, np.sqrt(2./(5*5*256)) )
        nn.init.xavier_uniform_(self.grp1_linear.weight)
        nn.init.constant_(self.grp1_linear.bias, 0.0)

        nn.init.normal_(self.grp2_conv1.weight, 0, np.sqrt(2./(5*5*32)) )
        nn.init.normal_(self.grp2_conv2.weight, 0, np.sqrt(2./(5*5*64)) )
        nn.init.normal_(self.grp2_conv3.weight, 0, np.sqrt(2./(5*5*128)) )
        nn.init.normal_(self.grp2_conv4.weight, 0, np.sqrt(2./(5*5*256)) )
        nn.init.xavier_uniform_(self.grp2_linear.weight)
        nn.init.constant_(self.grp2_linear.bias, 0.0)


    def forward(self, x):
        x_grp1 = x[:,0,:,:,:]
        x_grp2 = x[:,1,:,:,:]

        output_grp1 = self.grp1_conv1(x_grp1)    
        output_grp1 = self.grp1_conv1_bn(output_grp1)
        output_grp1 = self.grp1_conv1_ac(output_grp1)
        output_grp1 = self.grp1_conv2(output_grp1)   
        output_grp1 = self.grp1_conv2_bn(output_grp1)             
        output_grp1 = self.grp1_conv2_ac(output_grp1)       
        output_grp1 = self.grp1_conv3(output_grp1)        
        output_grp1 = self.grp1_conv3_bn(output_grp1)        
        output_grp1 = self.grp1_conv3_ac(output_grp1)
        output_grp1 = self.grp1_conv4(output_grp1)  
        output_grp1 = self.grp1_conv4_bn(output_grp1)        
        output_grp1 = self.grp1_conv4_ac(output_grp1)
        output_grp1 = output_grp1.view(output_grp1.size(0), -1)
        output_grp1 = self.grp1_linear(output_grp1)  

        output_grp2 = self.grp2_conv1(x_grp2)
        output_grp2 = self.grp2_conv1_bn(output_grp2)               
        output_grp2 = self.grp2_conv1_ac(output_grp2)         
        output_grp2 = self.grp2_conv2(output_grp2)   
        output_grp2 = self.grp2_conv2_bn(output_grp2)          
        output_grp2 = self.grp2_conv2_ac(output_grp2)
        output_grp2 = self.grp2_conv3(output_grp2)   
        output_grp2 = self.grp2_conv3_bn(output_grp2)          
        output_grp2 = self.grp2_conv3_ac(output_grp2)
        output_grp2 = self.grp2_conv4(output_grp2)  
        output_grp2 = self.grp2_conv4_bn(output_grp2)
        output_grp2 = self.grp2_conv4_ac(output_grp2)
        output_grp2 = output_grp2.view(output_grp2.size(0), -1)        
        output_grp2 = self.grp2_linear(output_grp2)  

        return output_grp1.chunk(2, 1), output_grp2.chunk(2, 1)

class Decoder(nn.Module):
    def __init__(self, args, ngpu=0):
        super(Decoder, self).__init__()
        self.group = args.group
        self.nz = args.nz
        self.nd = args.nd      
        self.H = args.transform_size[0]
        self.W = args.transform_size[1] 
        self.hiddenunits = 512        

        self.grp1_linear = nn.Linear(self.nz, self.hiddenunits)      
        self.grp1_deconv4 = nn.ConvTranspose2d(256, 128, (5,5), stride=(1,1), padding=(0,0), bias=False)
        self.grp1_deconv4_bn = nn.BatchNorm2d(128)
        self.grp1_deconv4_ac = nn.ReLU() if args.tpu else nn.ELU()
        self.grp1_deconv3 = nn.ConvTranspose2d(128, 64, (5,5), stride=(3,3), padding=(2,2), bias=False) 
        self.grp1_deconv3_bn = nn.BatchNorm2d(64)        
        self.grp1_deconv3_ac = nn.ReLU() if args.tpu else nn.ELU()       
        self.grp1_deconv2 = nn.ConvTranspose2d(64, 32, (5,5), stride=(3,3), padding=(2,0), bias=False)       
        self.grp1_deconv2_bn = nn.BatchNorm2d(32)        
        self.grp1_deconv2_ac = nn.ReLU() if args.tpu else nn.ELU()        
        self.grp1_deconv1 = nn.ConvTranspose2d(32, self.nd, (5,5), stride=(3,3), padding=(0,0), bias=False)
        self.grp1_deconv1_bn = nn.BatchNorm2d(self.nd)   
        self.grp1_deconv1_ac = nn.ReLU() if args.tpu else nn.ELU()     
        self.grp1_sigmoid = nn.Sigmoid()     

        self.grp2_linear = nn.Linear(self.nz, self.hiddenunits)      
        self.grp2_deconv4 = nn.ConvTranspose2d(256, 128, (5,5), stride=(1,1), padding=(0,0), bias=False)
        self.grp2_deconv4_bn = nn.BatchNorm2d(128)        
        self.grp2_deconv4_ac = nn.ReLU() if args.tpu else nn.ELU() 
        self.grp2_deconv3 = nn.ConvTranspose2d(128, 64, (5,5), stride=(3,3), padding=(2,2), bias=False) 
        self.grp2_deconv3_bn = nn.BatchNorm2d(64)        
        self.grp2_deconv3_ac = nn.ReLU() if args.tpu else nn.ELU()       
        self.grp2_deconv2 = nn.ConvTranspose2d(64, 32, (5,5), stride=(3,3), padding=(2,0), bias=False)  
        self.grp2_deconv2_bn = nn.BatchNorm2d(32)             
        self.grp2_deconv2_ac = nn.ReLU() if args.tpu else nn.ELU()       
        self.grp2_deconv1 = nn.ConvTranspose2d(32, self.nd, (5,5), stride=(3,3), padding=(0,0), bias=False)
        self.grp2_deconv1_bn = nn.BatchNorm2d(self.nd)        
        self.grp2_deconv1_ac = nn.ReLU() if args.tpu else nn.ELU()
        self.grp2_sigmoid = nn.Sigmoid()      
        self.loss = nn.MSELoss(reduction='none')   
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.grp1_linear.weight)
        nn.init.constant_(self.grp1_linear.bias, 0.0)
        nn.init.normal_(self.grp1_deconv4.weight, 0, np.sqrt(2./(5*5*256)) )
        nn.init.normal_(self.grp1_deconv3.weight, 0, np.sqrt(2./(5*5*128)) )
        nn.init.normal_(self.grp1_deconv2.weight, 0, np.sqrt(2./(5*5*64)) )
        nn.init.normal_(self.grp1_deconv1.weight, 0, np.sqrt(2./(5*5*32)) )
        nn.init.xavier_uniform_(self.grp2_linear.weight)
        nn.init.constant_(self.grp2_linear.bias, 0.0)
        nn.init.normal_(self.grp2_deconv4.weight, 0, np.sqrt(2./(5*5*256)) )
        nn.init.normal_(self.grp2_deconv3.weight, 0, np.sqrt(2./(5*5*128)) )
        nn.init.normal_(self.grp2_deconv2.weight, 0, np.sqrt(2./(5*5*64)) )
        nn.init.normal_(self.grp2_deconv1.weight, 0, np.sqrt(2./(5*5*32)) )

    def forward(self, z):
        grp1_z = torch.squeeze(z[0])
        grp2_z = torch.squeeze(z[1])
        grp1_output = self.grp1_linear(grp1_z)
        grp1_output = grp1_output.view(grp1_output.size()[0], int(self.hiddenunits/2),1,2)
        grp1_output = self.grp1_deconv4(grp1_output)  
        grp1_output = self.grp1_deconv4_bn(grp1_output)                       
        grp1_output = self.grp1_deconv4_ac(grp1_output)  
        grp1_output = self.grp1_deconv3(grp1_output)   
        grp1_output = self.grp1_deconv3_bn(grp1_output)                  
        grp1_output = self.grp1_deconv3_ac(grp1_output)  
        grp1_output = self.grp1_deconv2(grp1_output)                        
        grp1_output = self.grp1_deconv2_bn(grp1_output)          
        grp1_output = self.grp1_deconv2_ac(grp1_output)   
        grp1_output = self.grp1_deconv1(grp1_output)                     
        grp1_output = self.grp1_deconv1_bn(grp1_output)          
        grp1_output = self.grp1_deconv1_ac(grp1_output)            
        grp2_output = self.grp2_linear(grp2_z)
        grp2_output = grp2_output.view(grp2_output.size()[0], int(self.hiddenunits/2),1,2)
        grp2_output = self.grp2_deconv4(grp2_output)
        grp2_output = self.grp2_deconv4_bn(grp2_output)                          
        grp2_output = self.grp2_deconv4_ac(grp2_output)  
        grp2_output = self.grp2_deconv3(grp2_output) 
        grp2_output = self.grp2_deconv3_bn(grp2_output)                  
        grp2_output = self.grp2_deconv3_ac(grp2_output)  
        grp2_output = self.grp2_deconv2(grp2_output)
        grp2_output = self.grp2_deconv2_bn(grp2_output)                                
        grp2_output = self.grp2_deconv2_ac(grp2_output)   
        grp2_output = self.grp2_deconv1(grp2_output)   
        grp2_output = self.grp2_deconv1_bn(grp2_output)                          
        grp2_output = self.grp2_deconv1_ac(grp2_output)  
        grp1_output = self.grp1_sigmoid(grp1_output)
        grp2_output = self.grp2_sigmoid(grp2_output)     
        grp1_output = grp1_output.view(grp1_output.size()[0], 1, self.nd, self.H, self.W )
        grp2_output = grp2_output.view(grp2_output.size()[0], 1, self.nd, self.H, self.W )
        output = torch.cat((grp1_output, grp2_output), 1)  

        return output

    def reconstruct_error(self, x, z):
        batch_size,_,_ = z[0].size()
        recon_x_flat = self.forward(z).view(batch_size, -1)
        x_flat = x.view(batch_size, -1)
        rec_error = self.loss(recon_x_flat, x_flat)  
        rec_error = rec_error.mean(1)  
        return rec_error

