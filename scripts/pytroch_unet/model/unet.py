#!/usr/bin/python3

import torch
import torch.nn as nn

# Reference Soruce : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class UNET(nn.Module):

    def __init__(self, in_channel=1, pooling='max', hiddenactivation='GELU', encoder_ksize=(3, 3), decoder_ksize=(3, 3),
                 encoder_padding=1, decoder_padding=1,updownsample_ksize=(2,2), updownsample_padding=0, dropout=0, device=None, outputActivate=None):
        super().__init__()
        self.device = None
        self.in_channel = in_channel
        self.hiddenactivation = hiddenactivation
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.poolinglayer = nn.MaxPool2d((2,2)) if (pooling == 'max') else nn.AvgPool2d((2,2))
        self.unet_encoder1 = self._encoder_convBlock(self.in_channel, 32, encoder_ksize, encoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder2 = self._encoder_convBlock(32, 64, encoder_ksize, encoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder3 = self._encoder_convBlock(64, 128, encoder_ksize, encoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder4 = self._encoder_convBlock(128, 256, encoder_ksize, encoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder5 = self._encoder_convBlock(256, 512, encoder_ksize, encoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)

        self.unet_decoder1 = self._decoder_convBlock(512, 256, decoder_ksize, decoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder2 = self._decoder_convBlock(256, 128, decoder_ksize, decoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder3 = self._decoder_convBlock(128, 64, decoder_ksize, decoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder4 = self._decoder_convBlock(64, 32, decoder_ksize, decoder_padding, dropout_p=dropout, hiddenactivation=self.hiddenactivation).to(self.device)
        self.output = nn.Conv2d(32, self.in_channel, kernel_size=1,padding=0).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoder1 = self.unet_encoder1(x)
        poolLayer = self.poolinglayer(encoder1)
        encoder2 = self.unet_encoder2(poolLayer)
        poolLayer = self.poolinglayer(encoder2)
        encoder3 = self.unet_encoder3(poolLayer)
        poolLayer = self.poolinglayer(encoder3)
        encoder4 = self.unet_encoder4(poolLayer)
        poolLayer = self.poolinglayer(encoder4)
        encoder5 = self.unet_encoder5(poolLayer)

        upscale = self.unet_decoder1(encoder5, encoder4)
        upscale = self.unet_decoder2(upscale, encoder3)
        upscale = self.unet_decoder3(upscale, encoder2)
        upscale = self.unet_decoder4(upscale, encoder1)
        upscale = self.output(upscale)
        return upscale

    class _encoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, ksize=(3, 3), padding=1 , dropout_p=0, hiddenactivation='GELU'):
            super().__init__()
            hiddenactivation = getattr(nn, hiddenactivation)
            activation = hiddenactivation()
            if (dropout_p > 0):
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=padding),
                    activation,
                    nn.Dropout2d(dropout_p),
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    activation,
                )
            else:
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=padding),
                    activation,
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    activation,
                )

        def forward(self, x):
            return self.conv2d(x)

    class _decoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, ksize=(3,3), padding=1, dropout_p=0, hiddenactivation='GELU'):
            super().__init__()
            hiddenactivation = getattr(nn,hiddenactivation)
            activation =  hiddenactivation()
            self.upscale = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=2),
            )
            if (dropout_p > 0):
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=padding),
                    activation,
                    nn.Dropout2d(dropout_p),
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    activation,
                )
            else:
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=padding),
                    activation,
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    activation,
                )

        def forward(self, x, y):
            x = self.upscale(x)
            x = self.conv2d(torch.cat([x, y], 1))
            return x

