import torch
import itertools
from models import Generator, Discriminator
from utils import init_weights

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Add arguments to the CycleGAN script.")
    parser.add_argument(
        '--epochs',
        help="Number of epochs of the training",
        type=int,
        default=300,
        required=False)
    parser.add_argument(
        '--dataset',
        help="Name of the dataset",
        type=str,
        default="monet2photo",
        required=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imSize = 256 if torch.cuda.is_available() else 128

    # Define the two generators and discriminators
    GAB = Generator((3, imSize, imSize)).to(device)
    GBA = Generator((3, imSize, imSize)).to(device)

    DA = Discriminator().to(device) # Discriminate A images
    DB = Discriminator().to(device)

    # Initialize the weights of the networks as described in the paper
    GAB.apply(init_weights)
    GBA.apply(init_weights)
    DA.apply(init_weights)
    DB.apply(init_weights)

    #TODO: We could also import them and load them from previously trained models.

    # Select the different losses 
    LGan = torch.nn.MSELoss().to(device)
    LCyc = torch.nn.L1Loss().to(device)

    # Create the optimizers
    # We have to chain because the losses make use of both networks
    optimGen = torch.optim.Adam(itertools.chain(GAB.parameters(), GBA.parameters()), lr = 0.0002, betas=(0.5,0.999))
    optimDis = torch.optim.Adam(itertools.chain(DA.parameters(), DB.parameters()), lr=0.0001, betas=(0.5,0.999))

    # Create the datasets
    datasetTrain = DataLoader(load_datasest(imSize, 'datasets/train'), batch_size=5, shuffle=True)
    datasetTest = DataLoader(load_datasest(imSize, 'datasets/test'), batch_size=5, shuffle=True)
    
    # Create the pools of previously created images for discriminator training
    poolA = []
    poolB = []

    # Start the training
    for epoch in range(args.epochs):
        for index, batch in enumerate(datasetTrain):
            # Load true, real images
            trueA = batch["A"]
            trueB = batch["B"]
            # Create the ground truth of true and false images
            tr = torch.Tensor(np.ones((trueA.size()[0], GBA.output_shape), requires_grad=False)
            fa = torch.Tensor(np.zeros((trueA.size()[0], GBA.output_shape), requires_grad=False)

            ## Train the generators

            fakeA = GBA(batch["B"])
            fakeB = GAB(batch["A"])
            
            # Cycle losses (going back and forth to a domain)
            CLoss1 = LCyc(GBA(GAB(trueA)), trueA)
            CLoss2 = LCyc(GAB(GBA(trueB)), trueB)
            CLoss = (CLoss1 + CLoss2)/2

            # Adversarial losses
            LGan1 = LGan(DA(fakeA).requires_grad(False), tr) # Not sure about the requires_grad
            LGan2 = LGan(DB(fakeB).requires_grad(False), tr)
            LG = (LGan1 + LGan2)/2

            # Identity losses not necessary for image -> painting, useful in the other way

            LossG = LG + CLoss
            optimGen.zero_grad()
            LossG.backward()
            optimGen.step()

            ## Train the discriminators

            ## A

            # Update the pools
            fakeA = update_pool(poolA, fakeA)
            fakeB = update_pool(poolB, fakeB)

            # Compute the losses
            realLoss = LGan(DA(trueA), True)
            fakeLoss = LGan(DA(fakeA), False)
            lossDA = (realLoss + fakeLoss)/2

            ## B

            # Compute the losses
            realLoss = LGan(DB(trueB), True)
            fakeLoss = LGan(DB(fakeB), False)
            lossDB = (realLoss + fakeLoss)/2

            lossD = lossDA + lossDB
            optimDis.zero_grad()
            lossD.backward()
            optimDis.step()
            

    # Set the learning rates accordingly