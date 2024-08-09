import torch
import torch.nn as nn
from criterion import Criterion

class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False, xavier=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)




    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64, xavier=False):
        super().__init__()
        self.n_classes = output_c
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True, xavier=xavier)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True, xavier=xavier)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block, xavier=xavier)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True, xavier=xavier)
        self.softmax = nn.Softmax(dim=1)

        if xavier:
            # module for module in model.modules() if not isinstance(module, nn.Sequential)
            for module in self.modules():
                if not isinstance(module, nn.Sequential):
                    if type(module) in [nn.Conv2d, nn.Linear]:
                        torch.nn.init.xavier_uniform_(module.weight)


    def forward(self, x, con=None):
        x = self.model(x)
        return self.softmax(x)



# model = Unet(input_c=1, output_c=2)
# criterion = Criterion(gamma=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# input = torch.rand((2, 1, 512, 512))
# target = torch.zeros((2, 2, 512, 512))
# target[:, 0, :, :] = 1
#
# output = model(input)
# ø = criterion(output, target)
# a = 2