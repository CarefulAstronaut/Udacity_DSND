## Use of the argparse tool was referenced from github user corochann and their work with MNIST data found below.  I liked the addition of "Print ArgParse variables and used it here"
## https://github.com/corochann/deep-learning-tutorial-with-chainer/blob/master/src/02_mnist_mlp/train_mnist_4_trainer.py
## I think it's covered under the MIT License

import argparse
import image_prep
import functions
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser(description='Train a Neural Network to categorize images of flowers.')
    parser.add_argument('--save_dir', '-dir', 
                    help='Select a save directory for the model')
    parser.add_argument('--arch', '-arch', type=str, default=vgg11,
                    help='Choose an architecture for the model to be based on')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.003,
                    help='Set a specific learning rate')
    parser.add_argument('--hidden_units', '-hu', type=int, default=256,
                    help='Set the number of hidden nodes in the network')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                    help='Set how many training epochs the model should go through')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='Use a specific GPU for training if available')
    arg = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('Model Architecture: {}'.format(args.arch))
    print('Learning Rate: {}'.format(args.learning_rate))
    print('Hidden Units: {}'.format(args.hidden_units))
    print('Epochs: {}'.format(args.epochs))
    
    ## Building Classifier
    # Load Pre-trained Network
    model = models.(args.arch, pretrained=True)
    
    # Freeze parameters against back propogation
    for param in model.parameters():
        param.requires_grad = False
        
    # Build netowrk with specified number of hidden units
    model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units) # Use the specified number of hidden units
                                     nn.ReLU(), 
                                     nn.Dropout(0.05), 
                                     nn.Linear(args.hidden_units, 102), # Matches the initial layer of hidden units
                                     nn.LogSoftMax(dim=1))
    
    criterion = nn.NLLLoss()
    
    # Optimizer with specified learning rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Use specific GPU
    if args.gpu >=0:
        cuda.get_device(args.gpu).use() # Use a specific GPU
        model.to_gpu() 
        
    # Train the model with specified number of epochs
    epochs = args.epochs
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
    
    running_loss = 0
    
    print(f'Epoch {epoch+1}/{epochs}..')
    
    for inputs, labels in train_loader: 
        
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad()
    
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        
    else:
        valid_loss = 0
        accuracy = 0
    
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                valid_loss = criterion(logps, labels)
                
                # Track the loss and accuracy on the validation set to determine the best hyperparameters
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(*equals.type(torch.FloatTensor)).item()
                
        train_losses.append(running_loss/len(train_loader))
        valid_losses.append(valid_loss/len(valid_loader))

        # Print results of each training epoch
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
        running_loss = 0