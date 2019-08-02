data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transformations of images
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                                    
test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(244),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Datasets 
train_set = datasets.ImageFolder(train_dir, transform=train_transform)
valid_set = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_set = datasets.ImageFolder(test_dir, transform=test_transform)

# Define dataloaders
train_loader = torch.utils.data.DataLoader(train_set)
valid_loader = torch.utils.data.DataLoader(valid_set)
test_loader = torch.utils.data.DataLoader(test_set)