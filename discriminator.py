
from constants import * 

dim_d = 64

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = nn.Conv2d(nc * 2, dim_d, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(dim_d)
        self.c2 = nn.Conv2d(dim_d, dim_d*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(dim_d*2)
        self.c3 = nn.Conv2d(dim_d*2, dim_d*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(dim_d*4)
        self.c4 = nn.Conv2d(dim_d*4, dim_d*8, 4, 1, 1)
        self.bn4 = nn.BatchNorm2d(dim_d*8)
        self.c5 = nn.Conv2d(dim_d*8, 1, 4, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        xy = F.leaky_relu(self.bn1(self.c1(xy)), 0.2)
        xy = F.leaky_relu(self.bn2(self.c2(xy)), 0.2)
        xy = F.leaky_relu(self.bn3(self.c3(xy)), 0.2)
        xy = F.leaky_relu(self.bn4(self.c4(xy)), 0.2)
        xy = self.c5(xy)

        return self.sigmoid(xy)
