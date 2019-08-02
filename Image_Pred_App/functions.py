import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from PIL import image

## Processes image and retruns as numpy
def process_image(image):
    
    mean = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]
    
    image.thumbnail(256)
    image.crop(244)
    np_image = np.array(image)
    norm_image = (np_image - mean) / sd
    np.transpose(norm)image, (2, 0, 1)
    
    return 
    

## Predict class from image file
def predict(image_path, model, topk=5):

    model.eval()
    
    dataiter = iter(test_loader)
    images, label = dataiter.next()
    img = images[0]
    
    img.img.view(1, 784)
    
    with torch.no_grad():
        output = model.forward(img)
        # Need feedback on code from part 1
        probs = 
        classes = 
        
## Viewing image and it's predicted calsses
def view_classify(img, ps):
    
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arrange(5), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arrange(5))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    
    plt.tight_layout()
    
## Saving the model in a checkpoint
def model_save():
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    torch.save(model.state_dict(), args.save_dir + '/checkpoint.pth')
    
## Loading the model from checkpoint
def model_load():
    state_dict = torch.load(args.save_dir + '/checkpoint.pth')
    model.load_state_dict(state_dict)
    
## Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)