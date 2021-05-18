import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from utils import compute_FID

class GAN(pl.LightningModule):
    def __init__(
            self,
            generator_class,
            discriminator_class,
            no_validation_images: int = 10,
            lr_gen: float = 1E-3,
            lr_dis: float = 1E-3,
            batch_size: int = 32,
            b1: float = 0.5,
            b2: float = 0.999,
            dataset: str = "MNIST",
            FID_dim: int = 2048
    ):
        super().__init__()
        self.save_hyperparameters('lr_gen', 'lr_dis', 'batch_size', 'b1', 'b2', 'FID_dim')

        self.generator = generator_class
        self.discriminator = discriminator_class

        self.validation_z = torch.randn(no_validation_images, self.generator.latent_dim)

        # this is used for tracing, I think to ensure that your dimensions are as expected
        # TODO figure out proper dim
        # self.example_input_array = torch.zeros()

        self.dataset = dataset

    def forward(self, z):
        return self.generator(z)

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
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # for p in self.discriminator.parameters():
            #     p.data.clamp_(-0.1, 0.1)
            fake = torch.zeros(real_imgs.size(0), 1).type_as(real_imgs)

            gen_imgs = self(z) # this calls the forward pass
            real_loss = self.adversarial_loss(self.discriminator(real_imgs), real)
            fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
            # D_real = self.discriminator(real_imgs)
            # D_fake = self.discriminator(gen_imgs.detach())
            # D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
            dis_loss = (real_loss + fake_loss) / 2
            # return D_loss
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
        z = self.validation_z.type_as(self.generator.linear_layer.weight)
        gen_imgs = self(z)
        # gen_imgs = self(self.validation_z)
        grid = torchvision.utils.make_grid(gen_imgs)
        # write generated images to tensorboard using the manual logger of pl
        self.logger.experiment.add_image('generated_image_epoch_{}'.format(self.current_epoch), grid, self.current_epoch)

        FID = compute_FID(gen_imgs, self.dataset, self.hparams.batch_size, self.device, self.hparams.FID_dim)
        self.log('FID', FID)
        print('FID', FID)