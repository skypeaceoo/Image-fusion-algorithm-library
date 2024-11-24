from args import args
import utils
from torch.autograd import Variable
from Models import Generator
import torch
import os
from utils import make_floor
from imageio import imsave

def _generate_fusion_image(G_model, ir_img, vis_img):

    f = G_model(ir_img, vis_img)
    return f

def load_model(model_path):
    G_model = Generator()
    G_model.load_state_dict(torch.load(model_path))
    print('# generator parameters:', sum(param.numel() for param in G_model.parameters()))
    G_model.eval()
    G_model.cuda()
    return G_model

def generate(model,ir_path, vis_path, output_path,  index,  mode):
    result = "results"
    ir_img = utils.get_test_images(ir_path, mode=mode)
    vis_img = utils.get_test_images(vis_path, mode=mode)
    ir_img = ir_img.cuda()
    vis_img = vis_img.cuda()
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)



    img_fusion = _generate_fusion_image(model, ir_img, vis_img)

    ############################ multi outputs ##############################################
    file_name = str(index).zfill(2) + '.png'
    output_path = output_path + file_name
    img = img_fusion.cpu().data[0].numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    utils.save_images(output_path, img)
    print(output_path)



