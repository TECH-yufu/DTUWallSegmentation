import torch
import torch.nn as nn
from criterion import Criterion

class ContextualUnet(nn.Module):
    def __init__(self, input_c=1, output_c=2, context_c=2, num_filters=64, xavier=False):
        super().__init__()

        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        dropout = nn.Dropout(0.5)

        self.n_classes = output_c
        self.softmax = nn.Softmax(dim=1)

        self.contextdown = nn.Sequential(nn.Conv2d(context_c, num_filters, kernel_size=4,
                                                  stride=2, padding=1, bias=False)
                                        )

        self.downblock1 = nn.Sequential(nn.Conv2d(input_c, num_filters, kernel_size=4,
                                                  stride=2, padding=1, bias=False)
                                        )
        self.downblock2 = nn.Sequential(downrelu,
                                        nn.Conv2d(num_filters, num_filters * 2, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(num_filters * 2)
                                        )
        self.downblock3 = nn.Sequential(downrelu,
                                        nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(num_filters * 4)
                                        )
        self.downblock4 = nn.Sequential(downrelu,
                                        nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(num_filters * 8)
                                        )
        self.down = nn.Sequential(downrelu,
                                        nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(num_filters * 8)
                                        )
        self.bottleneck = nn.Sequential(downrelu,
                                        nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                                        uprelu,
                                        nn.ConvTranspose2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                           stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(num_filters * 8)
                                        )
        self.bottleneck_down = nn.Sequential(downrelu,
                                        nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                  stride=2, padding=1, bias=False)
                                        )
        self.bottleneck_up = nn.Sequential(uprelu,
                                        nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, kernel_size=4,
                                                           stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(num_filters * 8)
                                        )


        self.up = nn.Sequential(uprelu,
                                nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_filters * 8),
                                dropout
                                )
        self.upblock4 = nn.Sequential(uprelu,
                                nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 4, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_filters * 4),
                                )

        self.upblock3 = nn.Sequential(uprelu,
                                nn.ConvTranspose2d(num_filters * 8, num_filters * 2, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_filters * 2),
                                )

        self.upblock2 = nn.Sequential(uprelu,
                                nn.ConvTranspose2d(num_filters * 4, num_filters, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_filters),
                                )

        self.upblock1 = nn.Sequential(uprelu,
                                nn.ConvTranspose2d(num_filters * 2, output_c, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
                                nn.Tanh(),
                                )
        if xavier:
            # module for module in model.modules() if not isinstance(module, nn.Sequential)
            for module in self.modules():
                if not isinstance(module, nn.Sequential):
                    if type(module) in [nn.Conv2d, nn.Linear]:
                        torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x, y_context):
        # print("Input dim:", x.shape)
        d1 = self.downblock1(x)
        # print(d1.shape)

        d2 = self.downblock2(d1)
        # print(d2.shape)
        d3 = self.downblock3(d2)
        # print(d3.shape)
        d4 = self.downblock4(d3)
        # print(d4.shape)

        d5 = self.down(d4)
        # print(d5.shape)
        d6 = self.down(d5)
        # print(d6.shape)
        d7 = self.down(d6)
        # print(d7.shape)

        bottleneck1 = self.bottleneck_down(d7)
        # print("bottleneck:", bottleneck1.shape)

        c = self.context(y_context)


        up7 = self.bottleneck_up(torch.cat([bottleneck1, c], 1))
        # print(up7.shape)
        ## TORCH.CAT BECAUSE OF SKIP CONNECTIONS
        up6 = self.up(torch.cat([up7,d7], 1))
        # print(up6.shape)
        up5 = self.up(torch.cat([up6, d6], 1))
        # print(up5.shape)
        up4 = self.up(torch.cat([up5, d5], 1))
        # print(up4.shape)
        up3 = self.upblock4(torch.cat([up4, d4], 1))
        # print(up3.shape)
        up2 = self.upblock3(torch.cat([up3, d3], 1))
        # print(up2.shape)
        up1 = self.upblock2(torch.cat([up2, d2], 1))
        # print(up1.shape)
        x = self.upblock1(torch.cat([up1,d1], 1))

        # print("Output dim:", x.shape)

        return self.softmax(x)

    def context(self,y_context):

        d1 = self.contextdown(y_context)

        d2 = self.downblock2(d1)

        d3 = self.downblock3(d2)

        d4 = self.downblock4(d3)


        d5 = self.down(d4)

        d6 = self.down(d5)

        d7 = self.down(d6)

        context = self.bottleneck_down(d7)

        return context


# model = ContextualUnet(input_c=1, output_c=2)
# # summary(model, (1, 256, 256))
# criterion = Criterion(lambda_=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# label = torch.rand((2, 1, 256,256))
# prev_label = torch.rand((2, 1, 256,256))
# prev_slice = torch.rand((2, 1, 256,256))
#
# context = torch.cat([prev_label, prev_slice], 1)
#
#
# # target = torch.zeros((2, 2, 512, 512))
# # target[:, 0, :, :] = 1
# # input, target = input.cuda(), target.cuda()
#
# output = model(label, context)
# print(output.shape)
# # ø = criterion(output, target)
# a = 2