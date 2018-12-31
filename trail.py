import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision
import pathlib
import torch.nn.functional as F

path=pathlib.Path('/home/rahul/Desktop/CIS581/680/celeba')
device=torch.device('cuda')
dataset=torchvision.datasets.ImageFolder(root=path,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
train_loader=torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=False)


class vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(3,3,kernel_size=2,padding=1)
        self.conv2=torch.nn.Conv2d(3,32,kernel_size=2,padding=1)
        self.conv3=torch.nn.Conv2d(32,64,kernel_size=2,padding=1)
        self.conv4=torch.nn.Conv2d(64,32,kernel_size=2,padding=1)
        self.fc1=torch.nn.Linear(512,1024)
        self.fc2=torch.nn.Linear(512,1024)
        self.deconv1=torch.nn.ConvTranspose2d(1024,128,kernel_size=5,stride=2)
        self.deconv2=torch.nn.ConvTranspose2d(128,64,kernel_size=5,stride=2)
        self.deconv3=torch.nn.ConvTranspose2d(64,32,kernel_size=6,stride=2)
        self.deconv4=torch.nn.ConvTranspose2d(32,3,kernel_size=6,stride=2)


    def encoder(self,x):
        out=F.max_pool2d(F.relu(self.conv1(x)),2)
        out=F.max_pool2d(F.relu(self.conv2(out)),2)
        out=F.max_pool2d(F.relu(self.conv3(out)),2)
        out=F.max_pool2d(F.relu(self.conv4(out)),2)
        out=out.view(out.size(0),-1)
        mean=F.relu(self.fc1(out))
        log_var=F.relu(self.fc2(out))
        return mean,log_var

    def decoder(self,x):
        out=F.relu(self.deconv1(x))
        out=F.relu(self.deconv2(out))
        out=F.relu(self.deconv3(out))
        out=F.sigmoid(self.deconv4(out))
        return out

    def forward(self,x):
        mean,log_var=self.encoder(x)
        std=log_var.mul(0.5).exp_()
        esp=torch.randn(mean.size()).to(device)
        z=mean+std*esp
        z=z.view(-1,1024,1,1).to(device)
        deco_img=self.decoder(z)
        return deco_img

model_vae=vae().to(device)
criterion=torch.nn.BCELoss()
vae_optimizer=torch.optim.Adam(model_vae.parameters())

epochs=500
img_list=[]
for i in range(epochs):
    total_loss=0
    for idx,(images,_) in enumerate(train_loader):
        model_vae.train()
        vae_optimizer.zero_grad()
        images=images.to(device)
        dec_img=model_vae(images)

        loss=criterion(dec_img,images)
        loss.backward()
        vae_optimizer.step()
    print('The Loss is {}'.format(loss))
    dec_img=dec_img.detach().cpu()
    img_list.append(vutils.make_grid(dec_img,padding=2,normalize=True))
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
