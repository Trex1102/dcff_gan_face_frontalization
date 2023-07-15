from constants import *

# def create_dataset():
#     img_path = dataroot + 'Data/Images/'
#     for i in range(1, 501):
#     # for i in range(1, 501):
#         folder_num = "".join(['0'] * (3 - len(str(i)))) + str(i)
#         f_src = img_path + folder_num + "/frontal/"
#         f_dst = dataroot + 'frontals/'

#         p_src = img_path + folder_num + "/profile/"
#         p_dst = dataroot + 'profiles/'
#         # for f in os.listdir(f_src):

#         for j in range(1, 41):
#             old_fn = '{:02d}.jpg'.format(j)
#             new_fn_f = '{}_f_{}'.format(i, old_fn)
#             os.rename(f_src + old_fn, f_dst + new_fn_f)
#             new_fn_p = '{}_p_{}'.format(i, old_fn)
#             os.rename(p_src + old_fn, p_dst + new_fn_p)

def create_dataset():
    img_path = dataroot + 'Data/Images/'
    for i in range(1, 501):
        folder_num = "".join(['0'] * (3 - len(str(i)))) + str(i)
        f_src = img_path + folder_num + "/frontal/"
        f_dst = dataroot + 'frontals/'

        p_src = img_path + folder_num + "/profile/"
        p_dst = dataroot + 'profiles/'

        # Create destination directories if they don't exist
        os.makedirs(f_dst, exist_ok=True)
        os.makedirs(p_dst, exist_ok=True)

        for j in range(1, 41):
            old_fn = '{:02d}.jpg'.format(j)
            new_fn_f = '{}_f_{}'.format(i, old_fn)
            os.rename(f_src + old_fn, f_dst + new_fn_f)
            new_fn_p = '{}_p_{}'.format(i, old_fn)
            os.rename(p_src + old_fn, p_dst + new_fn_p)


class PFImageDataset(Dataset):
    def __init__(self, profile_path, frontal_path, transform=None):
        self.profile_path = profile_path
        self.frontal_path = frontal_path
        self.transform = transform

        self.profile_images = os.listdir(profile_path)
        self.frontal_images = os.listdir(frontal_path)
        self.length = min(len(self.profile_images), len(self.frontal_images))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        p_img_file = '{}_p_{:02d}.jpg'.format(idx // 40 + 1, (idx - 1) % 40 + 1)
        p_img_path = os.path.join(self.profile_path, p_img_file)

        f_img_file = '{}_f_{:02d}.jpg'.format(idx // 40 + 1, (idx - 1) % 40 + 1)
        f_img_path = os.path.join(self.frontal_path, f_img_file)
        img_mode = ImageReadMode(1)
        image = read_image(p_img_path, mode=img_mode.RGB)
        label = read_image(f_img_path, mode=img_mode.RGB)
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        return image, label