import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'attention_local': decoder.AttentionDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, encoder_hand=None, encoder_img=None, encoder_t2d=None, device=None):
        super().__init__()
        
        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None
        
        if encoder_hand is not None:
            self.encoder_hand = encoder_hand.to(device)
        else:
            self.encoder_hand = None

        if encoder_img is not None:
            self.encoder_img = encoder_img.to(device)
        else:
            self.encoder_img = None
            
        if encoder_t2d is not None:
            self.encoder_t2d = encoder_t2d.to(device)
        else:
            self.encoder_t2d = None
            
        self._device = device

    def forward(self, p, inputs, imgs=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        c_hand = self.encode_hand_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r
            

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c
    
    def encode_hand_inputs(self, inputs):
        ''' Encodes to the parameters of hand

        Args:
            input (tensor): the input
        '''

        if self.encoder_hand is not None:
            c = self.encoder_hand(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def encode_hand_mano(self, inputs):
        ''' Encode the hand with mano layer

        Args:
            input (tensor): the input
        '''

        c = self.encoder_hand.forward_mano(inputs)
        return c

    
    def encode_img_inputs(self, imgs):
        ''' Encodes to the feature of tactile images

        Args:
            input (tensor): the input
        '''

        if self.encoder_img is not None:
            B, F, C, H, W = imgs.size()
            c_list = []
            
            for b_idx in range(B):
                imgs_in = imgs[b_idx, :, :, :, :].reshape(F, C, H, W)
                c_t = self.encoder_img(imgs_in).reshape(1, F, -1)
                c_list.append(c_t)
            
            c = torch.cat(c_list, dim=0)

        else:
            c = torch.empty(imgs.size(0), 0)
        
        return c

    def encode_t2d(self, inputs, imgs):
        ''' Encodes to the depth of tactile images and digit poses

        Args:
            input (tensor): the input
        '''
        pred_depth = self.encoder_t2d.encode_img_inputs(imgs)
        c_hand = self.encoder_t2d.encode_hand_inputs(inputs)
        
        return pred_depth, c_hand
        
            

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r
    
    def decode_img(self, p, p_img, c, c_img=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
            c_img (tensor): latent code from tactile signals
        '''
        
        logits = self.decoder.forward_img(p, p_img, c, c_img, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r
    
    def decode_contact(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
            c_img (tensor): latent code from tactile signals
        '''
        
        logits, pred_contact = self.decoder.forward_contact(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r, pred_contact

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
