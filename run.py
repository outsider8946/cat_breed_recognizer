import os, sys,getopt
import torch, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from template_model import CatModel

label_map = {0:'bengal', 1:'maine_coon', 2:'ragdoll', 3:'oriental_shorthair', 4:'british_shorthair', 5:'siamese'}
def preporcessing_img(filepath:str):
    img_mat = cv2.imread(filepath)
    img_mat = cv2.resize(img_mat,(32,32))
    img_mat = img_mat/255
    return torch.tensor(np.array([img_mat])).float()

def load_model(filepath:str):
    model = CatModel()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main(argv):
    img_path = ''
    opts, args = getopt.getopt(argv,"p:")
    for opt,arg in opts:
        if(opt == '-p'):
            img_path = arg

        if not os.path.isfile(img_path):
            print('This file does not exist! May be path is incorrect.')
            return -1

        model_path = '../model/cat_breed_classifier.pt'
        if not os.path.isfile(model_path):
            print('Model does not exist! Model must be in same folder with run file.')
            return -1

        model = load_model(model_path)
        input = preporcessing_img(img_path)
        output = model(input)
        bounds, labels = output
        bounds = bounds[0].detach().numpy()
        labels = labels[0].detach().numpy()

        img = Image.open(img_path)
        width, height = img.width,img.height

        fig, ax = plt.subplots()
        ax.imshow(img)
        rect = patches.Rectangle((bounds[0]*width,bounds[1]*height),bounds[2]*width,bounds[3]*height,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(bounds[0]*width, bounds[1]*height, label_map[int(labels.argmax())], bbox=dict(fill=True, edgecolor='red', linewidth=2,color='red'))
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])