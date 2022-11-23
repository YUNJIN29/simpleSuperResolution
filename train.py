import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Module import *
from Dataset import *
from utils.trainOptions import *
from utils.imageUtil import ImageSplitter

# load options
opts = TrainOptions().getOpts()

device = torch.device(opts.device)
checkpoint = opts.checkpoint
save_dir = opts.save_dir
train_dataset_dir = opts.train_dir
test_dataset_dir = opts.test_dir
learning_rate = opts.lr
epoch = opts.epoch
seg_size = opts.seq_size
scale_factor = opts.scale
border_pad_size = opts.border
logdir = opts.log_dir
# train times
train_times = 0
# test times
test_times = 0
test_cycle = opts.test_cycle
save_cycle = opts.save_cycle
pic_no = 0
test_offset = train_times - test_times * test_cycle

# dataset
train_dataset = ImgDataset(train_dataset_dir, HR_dir=opts.target_folder, LR_dir=opts.input_folder,
                           prefix=opts.train_prefix, subfix=opts.train_subfix)
test_dataset = ImgDataset(test_dataset_dir, HR_dir=opts.target_folder, LR_dir=opts.input_folder,
                          prefix=opts.test_prefix, subfix=opts.test_subfix)
train_dataset_len = len(train_dataset)
test_dataset_len = len(test_dataset)

# load data
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# load model
model = SRCNN()
model = model.to(device)

# trans datatype
model.float()

# loss Function
loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)

#  optimizer
optimizer = optim.Adam([
    {'params': model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
], lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# load model state
if checkpoint != '':
    state_data = torch.load(checkpoint)
    model.load_state_dict(state_data['model'])
    optimizer.load_state_dict(state_data['optim'])
    train_times = state_data['train_epoch']
    test_times = state_data['eval_epoch']
    pic_no = state_data['pic_no']
    print('载入checkpoint: {}'.format(checkpoint))
    model.eval()

img_splitter = ImageSplitter(seg_size, scale_factor, border_pad_size)

# tensorBoard
writer = SummaryWriter(logdir)


def clac(img, target):
    img = img.to(device)
    target = target.to(device)
    output = model(img)
    return loss_fn(output, target), output


def train(img, target, train_times):
    model.train()
    loss, final = clac(img, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar("train_loss", loss.item(), train_times)
    print('完成第{}次训练，loss: {}'.format(train_times, loss.item()))
    return train_times + 1


def patchTrain(img, target, train_times, pic_no):
    model.train()
    img_parts = img_splitter.split_img_tensor(img)
    target_part = img_splitter.split_img_tensor(target)
    print('第{}张数据，共{}个切片'.format(pic_no, len(img_parts)))
    total_loss = 0
    for i in range(len(img_parts)):
        loss, out = clac(img_parts[i], target_part[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_times = train_times + 1
        total_loss = total_loss + loss
        print('完成第{}次训练，loss: {}'.format(train_times, loss.item()))
    writer.add_scalar("train_avg_loss", total_loss.item(), train_times)
    return train_times, pic_no + 1


def test(test_times):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        flag = True
        for image, expect in test_dataloader:
            loss, final = clac(image, expect)
            total_loss = total_loss + loss
            if flag:
                flag = False
                image = image.to(device)
                con = torch.cat([image, final])
                writer.add_images("test-img", con, test_times)
    writer.add_scalar("test_loss", total_loss, test_times)
    writer.add_scalar("total_test_loss", total_loss / test_dataset_len, test_times)
    print("\n完成第{}次测试，total loss: {}\n".format(test_times, total_loss))
    return test_times + 1


def saveModel(save_no):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    filename = "checkpoint-{}X-{}.pth".format(scale_factor, save_no)
    save_state_data = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'train_epoch': train_times,
        'eval_epoch': test_times,
        'pic_no': pic_no
    }
    torch.save(save_state_data,
               os.path.join(save_dir, filename))
    print("已保存{}".format(filename))


if opts.patchs == 0:
    for i in range(epoch):
        print("----第{}轮学习开始----".format(i))
        for img, target in train_dataloader:
            train_times = train(img, target, train_times)

            # test
            if train_times % test_cycle == 0:
                test_times = test(test_times)

            # save state
            if train_times % save_cycle == 0:
                saveModel(train_times)
        print("----第{}轮学习结束----".format(i))
    if train_times % save_cycle != 0:
        test_times = test(test_times)
        saveModel(train_times)
else:
    for i in range(epoch):
        print("----第{}轮学习开始----".format(i))
        for img, target in train_dataloader:
            train_times, pic_no = patchTrain(img, target, train_times, pic_no)
            if pic_no % test_cycle == 0:
                test_times = test(test_times)
            if pic_no % save_cycle == 0:
                saveModel(pic_no)
        print("----第{}轮学习结束----".format(i))
    if pic_no % save_cycle != 0:
        test_times = test(test_times)
        saveModel(pic_no)

writer.close()
