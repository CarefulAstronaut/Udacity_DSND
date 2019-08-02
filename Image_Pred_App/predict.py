## Use of the argparse tool was referenced from github user corochann and their work with MNIST data found below.  I liked the addition of "Print ArgParse variables and used it here"
## https://github.com/corochann/deep-learning-tutorial-with-chainer/blob/master/src/02_mnist_mlp/train_mnist_4_trainer.py
## I think it's covered under the MIT License

import argparse
import image_prep
import functions

def main():
    parser = argparse.ArgumentParser(description='Train a Neural Network to categorize images of flowers.')
    parser.add_argument('--top_k', '-k', type=int, default=5 
                    help='Define how many possible categories to return')
    parser.add_argument('--names', '-n')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='Use a specific GPU for inference if available')
    arg = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('Top K Categories: {}'.format(args.top_k))
    
    
    ### Will need to use a corrected inference program