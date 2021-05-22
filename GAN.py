import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

import numpy as np
import pytorch_lightning as pl
from utils import compute_FID, compute_IS, create_dir_from_tensors

class GAN(pl.LightningModule):
    def __init__(
            self,
            generator_class,
            discriminator_class,
            no_validation_images: int = 10,
            lr_gen: float = 1E-3,
            lr_dis: float = 1E-2,
            batch_size: int = 32,
            b1: float = 0.0,
            b2: float = 0.9,
            dataset: str = "MNIST",
            FID_step: int = 10,
            FID_dim: int = 2048,
            fid_max_data: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters('lr_gen', 'lr_dis', 'batch_size', 'b1', 'b2', 'FID_step', 'FID_dim', 'fid_max_data')

        self.generator = generator_class
        self.discriminator = discriminator_class

        self.validation_z = torch.randn(no_validation_images, self.generator.latent_dim)

        # this is used for tracing, I think to ensure that your dimensions are as expected
        # TODO figure out proper dim
        # self.example_input_array = torch.zeros()

        self.dataset = dataset

    def forward(self, z):
        return self.generator(z)

    def WGAN_GP_loss(self, real_imgs, fake_imgs, real_validity, fake_validity, lambda_gp=10):

        def compute_gradient_penalty(D, real_samples, fake_samples):
            """Calculates the gradient penalty loss for WGAN GP"""
            # Random weight term for interpolation between real and fake samples
            alpha = Variable(torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))).to(self.device)
            # Get random interpolation between real and fake samples
            interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
            d_interpolates = D(interpolates)
            fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
            # Get gradient w.r.t. interpolates
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty

        gradient_penalty = compute_gradient_penalty(self.discriminator, real_imgs, fake_imgs)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        return d_loss

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """

        :param batch:
        :param batch_idx:
        :param optimizer_idx: which optimizer to use for the training, step
        we need this since we have two one for the generator and another for the disciminator
        :return:
        """

        real_imgs, _ = batch  # we do not need the actual class

        z = torch.randn(real_imgs.shape[0], self.generator.latent_dim).type_as(real_imgs)
        real = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)

        # train generator
        if optimizer_idx == 0:
            gen_imgs = self(z) # this calls the forward pass
            D_fake = self.discriminator(gen_imgs)
            # g_loss = -torch.mean(D_fake)
            g_loss = self.adversarial_loss(D_fake, real)
            # g_loss = -torch.mean(D_fake)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:

            gen_imgs = self(z) # this calls the forward pass

            # Real images
            real_validity = self.discriminator(real_imgs)
            # Fake images
            fake_validity = self.discriminator(gen_imgs)

            dis_loss = self.WGAN_GP_loss(real_imgs, gen_imgs, real_validity, fake_validity)

            return dis_loss

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#use-multiple-optimizers-like-gans
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_gen,
                                 betas=(self.hparams.b1, self.hparams.b2))
        dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_dis,
                                 betas=(self.hparams.b1, self.hparams.b2))
        # gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_gen)
        # dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_dis)
        # return (
        #     {'optimizer': gen_opt, 'frequency': 1},
        #     {'optimizer': dis_opt, 'frequency': 5}
        # )
        return gen_opt, dis_opt

    def on_epoch_end(self):
        """
        at the end of the epoch runs this function
        :return:
        """
        z = self.validation_z.to('cuda' if torch.cuda.is_available() else 'cpu')
        # TODO minibatch

        train_dat = torch.utils.data.TensorDataset(z)  # assume train_in is a tensor
        dataloader_train = torch.utils.data.DataLoader(train_dat, batch_size=self.hparams.batch_size)

        offset = 0
        for z_mini in dataloader_train:
            gen_imgs = self(z_mini[0])
            fake_path = create_dir_from_tensors(gen_imgs, offset=offset, already_created=False)
            if (self.current_epoch + 1) % self.hparams.FID_step != 0:
                break
            offset += self.hparams.batch_size

        grid = torchvision.utils.make_grid(gen_imgs)
        self.logger.experiment.add_image('generated_image_epoch_{}'.format(self.current_epoch), grid,
                                         self.current_epoch)
        #

        if (self.current_epoch + 1) % self.hparams.FID_step == 0:
            FID = compute_FID(fake_path, self.dataset, self.hparams.batch_size,
                              self.device, self.hparams.FID_dim, self.hparams.fid_max_data)
            self.log('FID', FID)
            if self.dataset not in ["MNIST", "FashionMNIST", "MNIST_128"]:
                IS = compute_IS(fake_path, already_created=True)
                self.log('IS', IS[0])