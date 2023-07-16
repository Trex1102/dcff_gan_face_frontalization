from constants import *
from dataset import *
from discriminator import *
from generator import *
from utils import *

class Trainer:
    def __init__(self):
        return None
    
    def train(self):
        # Create the dataset
        #create_dataset()

        dataset = PFImageDataset(profile_path,
                         frontal_path,
                         transform=transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.Resize(image_size),
                             transforms.CenterCrop(image_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,),(0.5,))
                         ]))
        print("Dataset size:", len(dataset))

        image, image2 = dataset[2000]  # Assuming the dataset returns a tuple of image and label

        # Converting the image tensor to a numpy array and removing the normalization
        image = (image * 0.5) + 0.5  # Denormalize the image
        image = image.permute(1, 2, 0).numpy()  # Reorder dimensions and convert to numpy array

        # Displaying the image using matplotlib
        plt.imshow(image)
        plt.axis('off')
        plt.show()



        image2 = (image2 * 0.5) + 0.5  # Denormalize the image
        image2 = image2.permute(1, 2, 0).numpy()  # Reorder dimensions and convert to numpy array

        # Displaying the image using matplotlib
        plt.imshow(image2)
        plt.axis('off')
        plt.show()



        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size
                                                )

        test_dataloader = torch.utils.data.DataLoader(test_dataset
                                                    )
            
        train_features, train_labels = next(iter(train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")

        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training-Label Pair Images")
        plt.imshow(np.transpose(vutils.make_grid([train_features[2].to(device), train_labels[2].to(device)], padding=2, normalize=True).cpu(), (1,2,0)))

        # Create the generator
        netG = Generator().to(device)

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        netG.apply(weights_init)

        # Print the model
        #print(netG)

        # Create the Discriminator
        netD = Discriminator().to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu >= 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(weights_init)

        # Print the model
        #print(netD)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        BCE = nn.BCELoss()
        L1 = nn.L1Loss()

        fix_con = next(iter(test_dataloader))
        print(len(fix_con))
        fix_X, fix_y = fix_con
        print(type(fix_X))
        fix_X = fix_X.to(device)
        fix_y = fix_y.to(device)

        img_list = []
        netG_losses = []
        netD_losses = []
        netG_GAN_losses = []
        netG_L1_losses = []
        num_epochs = 10
        iter_per_plot = 1

        iters = 0
        L1_lambda = 100.0

        test_num = 7


        for epoch in tqdm(range(num_epochs)):

            for i, (x, y) in enumerate(train_dataloader):

                size = x[0].shape[0]
                x, y = x.to(device), y.to(device)

                netD.zero_grad()


                r_patch = netD(y, x)
                r_masks = torch.ones(r_patch.shape).to(device)
                f_masks = torch.zeros(r_patch.shape).to(device)
                r_gan_loss=BCE(r_patch, r_masks)

                fake = netG(x)


                #fake_patch
                f_patch = netD(fake.detach(),x)
                f_gan_loss=BCE(f_patch,f_masks)
                netD_loss = r_gan_loss + f_gan_loss
                netD_loss.backward()
                optimizerD.step()
                # netG
                netG.zero_grad()
                f_patch = netD(fake,x)
                f_gan_loss=BCE(f_patch,r_masks)
                L1_loss = L1(fake,y)
                netG_loss = f_gan_loss + L1_lambda*L1_loss
                netG_loss.backward()

                optimizerG.step()
                iters += 1
                if (iters + 1) % iter_per_plot == 0 :

                    print('Epoch [{}/{}], Step [{}/{}], netD_loss: {:.4f}, netG_loss: {:.4f},netD(real): {:.2f}, netD(fake):{:.2f}, netG_loss_gan:{:.4f}, netG_loss_L1:{:.4f}'.format(epoch, num_epochs, i+1, len(train_dataloader), netD_loss.item(), netG_loss.item(), r_patch.mean(), f_patch.mean(), f_gan_loss.item(), L1_loss.item()))

                    netG_losses.append(netG_loss.item())
                    netD_losses.append(netD_loss.item())
                    netG_GAN_losses.append(f_gan_loss.item())
                    netG_L1_losses.append(L1_loss.item())
                    with torch.no_grad():
                        netG.eval()
                        fake = netG(fix_X).detach().cpu()
                        netG.train()
                    figs=plt.figure(figsize=(10,10))
                    plt.subplot(1,3,1)
                    plt.axis("off")
                    plt.title("input image")
                    plt.imshow(np.transpose(vutils.make_grid(fix_X, nrow=1, padding=5,
                    normalize=True).cpu(), (1,2,0)))
                    plt.subplot(1,3,2)
                    plt.axis("off")
                    plt.title("netGerated image")
                    plt.imshow(np.transpose(vutils.make_grid(fake, nrow=1, padding=5,
                    normalize=True).cpu(), (1,2,0)))

                    plt.subplot(1,3,3)
                    plt.axis("off")
                    plt.title("ground truth")
                    plt.imshow(np.transpose(vutils.make_grid(fix_y, nrow=1, padding=5,
                    normalize=True).cpu(), (1,2,0)))

                    save_dir = 'logs/test' + str(test_num)
                    os.makedirs(save_dir, exist_ok=True)

                    plt.savefig(os.path.join(save_dir, 'Test-{}-{}.png'.format(epoch, test_num)))
                    plt.close()
                    img_list.append(figs)
            save_dir = 'logs/saved_model' + str(test_num)
            os.makedirs(save_dir, exist_ok=True)

            torch.save(netG.state_dict(), os.path.join(save_dir, 'netG-{}-{}.pth'.format(epoch, test_num)))


