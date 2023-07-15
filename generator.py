
from constants import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n1 = nn.Conv2d(nc, ngf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.n2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.n3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        self.n4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ngf*8)
        self.n5 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ngf*8)
        self.n6 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8)

        self.m9 = nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1)
        self.bn9 = nn.BatchNorm2d(ngf*8)
        self.m0 = nn.ConvTranspose2d(ngf*8*2, ngf*8, 4, 2, 1)
        self.bn0 = nn.BatchNorm2d(ngf*8)
        self.m1 = nn.ConvTranspose2d(ngf*8*2, ngf*4, 4, 2, 1)
        self.bn1_ = nn.BatchNorm2d(ngf*4)
        self.m2 = nn.ConvTranspose2d(ngf*4*2, ngf*2, 4, 2, 1)
        self.bn2_ = nn.BatchNorm2d(ngf*2)
        self.m3 = nn.ConvTranspose2d(ngf*2*2, ngf*1, 4, 2, 1)
        self.bn3_ = nn.BatchNorm2d(ngf)
        self.m4 = nn.ConvTranspose2d(ngf*1*2, nc, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        n1 = self.bn1(self.n1(x))
        n2 = self.bn2(self.n2(F.leaky_relu(n1, 0.2)))
        n3 = self.bn3(self.n3(F.leaky_relu(n2, 0.2)))
        n4 = self.bn4(self.n4(F.leaky_relu(n3, 0.2)))
        n5 = self.bn5(self.n5(F.leaky_relu(n4, 0.2)))
        n6 = self.bn6(self.n6(F.leaky_relu(n5, 0.2)))

        m9 = torch.cat([self.bn9(self.m9(F.relu(n6))), n5], 1)
        m0 = torch.cat([self.bn0(self.m0(F.relu(m9))), n4], 1)
        m1 = torch.cat([self.bn1_(self.m1(F.relu(m0))), n3], 1)
        m2 = torch.cat([self.bn2_(self.m2(F.relu(m1))), n2], 1)
        m3 = torch.cat([self.bn3_(self.m3(F.relu(m2))), n1], 1)
        m4 = self.m4(F.relu(m3))

        return self.tanh(m4)
