import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import facenet_pytorch
from facenet_pytorch import MTCNN
from PIL import Image
mtcnn = MTCNN(post_process=False)

from data import CelebaInterface, LFWInterface, AdienceInterface
from torchvision import transforms


# celebA image
celeba_imgs_list = []
celeba_start_index = 50
celeba_end_index = 58
for i in range(celeba_start_index, celeba_end_index+1):
    file_name = f"{i:06d}.jpg"
    celeba_img = Image.open(r'E:/datasets/celeba/img_align_celeba/img_align_celeba_mtcnn/'+file_name)
    celeba_imgs_list.append(celeba_img)
print(celeba_imgs_list)



# LFW images
lfw_imgs_list = []

lfw_data_module = LFWInterface(dim_img=224,
                               dataset='LFW_data',
                               data_dir='E:\datasets\lfw\lfw112',
                               sensitive_attr='Male',
                               batch_size=1,
                               num_workers=0,
                               pin_memory=False,
                               identity_nums=5749,
                               sensitive_dim=1,
                               purpose='attr_extract')
lfw_data_module.setup(stage='test')

for i, item in enumerate(lfw_data_module.test_dataloader()):
    x, _, _ = item
    print(x.size())
    x = x.squeeze(0)
    to_img = transforms.ToPILImage()
    lfw_img = to_img(x)
    #img.show()

    lfw_imgs_list.append(lfw_img)
    if i == 8:
        break
#print(lfw_imgs_list)


#image = Image.open('path_to_your_image.jpg')
#faces, _ = mtcnn(image, save_path='path_to_save_cropped_faces/')




# Adience
adience_imgs_list = []

adience_data_module = AdienceInterface(dim_img=224,
                                       dataset='Adience',
                                       data_dir='E:\datasets\Adience',
                                       sensitive_attr='Male',
                                       batch_size=1,
                                       num_workers=0,
                                       pin_memory=False,
                                       identity_nums=5749,
                                       sensitive_dim=1,
                                       purpose='gender_extract')

for i, item in enumerate(adience_data_module.test_dataloader()):
    print('i', i)
    x, _, _ = item
    x = x.squeeze(0)
    to_img = transforms.ToPILImage()
    adience_img = to_img(x)

    if i == 0:
        continue
    elif i <= 9:
        adience_imgs_list.append(adience_img)
    elif i == 10:
        break
print(adience_imgs_list)





fig, axes = plt.subplots(3, 9, figsize=(6.5, 2.5))
#plt.subplots_adjust(hspace=0.2)
for j in range(9):
    axes[0, j].imshow(celeba_imgs_list[j])
    axes[0, j].axis('off')

for j in range(9):
    axes[1, j].imshow(lfw_imgs_list[j])
    axes[1, j].axis('off')

for j in range(9):
    axes[2, j].imshow(adience_imgs_list[j])
    axes[2, j].axis('off')

#for i in range(3):

plt.tight_layout()
plt.show()
plt.close()
plt.axis('off')
plt.xticks([])
plt.yticks([])
fig.savefig('datasets_imgs.pdf', dpi=300, bbox_inches='tight', pad_inches=0)