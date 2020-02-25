import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from collections import OrderedDict


class MIML(nn.Module):

    def __init__(self, L=1024, K=20, base_model='resnet',  fine_tune=False):
        """
        Arguments:
            L (int):
                number of labels
            K (int):
                number of sub categories
        """
        super(MIML, self).__init__()
        self.L = L
        self.K = K
        self.b = base_model
        # pretrained ImageNet VGG
        if base_model == 'vgg':
            # pretrained ImageNet VGG
            base_model = torchvision.models.vgg16(pretrained=True)
            base_model = list(base_model.features)[:-1]
            self.base_model = nn.Sequential(*base_model)
            dim = 512
            map_size = 256
        elif base_model == 'resnet':
            base_model = torchvision.models.resnet101(
                pretrained=True)
            self.base_model = torch.nn.Sequential(OrderedDict([
                ('conv1', base_model.conv1),
                ('bn1', base_model.bn1),
                ('relu', base_model.relu),
                ('maxpool', base_model.maxpool),
                ('layer1', base_model.layer1)]))

            self.intermidate = torch.nn.Sequential(OrderedDict([
                ('layer2', base_model.layer2),
                ('layer3', base_model.layer3)]))

            self.last = torch.nn.Sequential(OrderedDict([
                ('layer4', base_model.layer4)]))

            dim = 2048
            map_size = 64
        self.fine_tune(fine_tune)
        self.sub_concept_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(dim, 512, 1)),
            ('dropout1', nn.Dropout(0.5)),  # (-1,512,14,14)
            ('conv2', nn.Conv2d(512, K*L, 1)),
            # input need reshape to (-1,L,K,H*W)
            ('maxpool1', nn.MaxPool2d((K, 1))),
            # reshape input to (-1,L,H*W), # permute(0,2,1)
            ('softmax1', nn.Softmax(dim=2)),
            # permute(0,2,1) # reshape to (-1,L,1,H*W)
            ('maxpool2', nn.MaxPool2d((1, map_size)))
        ]))

    def forward(self, x):
        # IN:(8,3,224,224)-->OUT:(8,512,14,14)
        if self.b == 'vgg':
            base_out = self.base_model(x)
        elif self.b == 'resnet':
            base_out = self.last(self.intermidate(self.base_model(x)))

        # C,H,W = 512,14,14
        _, C, H, W = base_out.shape
        # OUT:(8,512,14,14)

        conv1_out = self.sub_concept_layer.dropout1(
            self.sub_concept_layer.conv1(base_out))

        # shape -> (n_bags, (L * K), n_instances, 1)
        conv2_out = self.sub_concept_layer.conv2(conv1_out)
        # shape -> (n_bags, L, K, n_instances)
        conv2_out = conv2_out.reshape(-1, self.L, self.K, H*W)
        # shape -> (n_bags, L, 1, n_instances),remove dim: 1
        maxpool1_out = self.sub_concept_layer.maxpool1(conv2_out).squeeze(2)

        # softmax
        permute1 = maxpool1_out.permute(0, 2, 1)
        softmax1 = self.sub_concept_layer.softmax1(permute1)
        permute2 = softmax1.permute(0, 2, 1)
        # reshape = permute2.unsqueeze(2)
        # predictions_instancelevel
        reshape = permute2.reshape(-1, self.L, 1, H*W)
        # shape -> (n_bags, L, 1, 1)
        maxpool2_out = self.sub_concept_layer.maxpool2(reshape)
        out = maxpool2_out.squeeze()

        return out

    def fine_tune(self, fine_tune=True):
        # only fine_tune the last three convs 8.21  微调后6个卷积层,17:  ;  最后三个24:
        if self.b == 'vgg':
            layer = -6
            for p in self.base_model.parameters():
                p.requires_grad = False
            for c in list(self.base_model.children())[24:]:
                for p in c.parameters():
                    p.requires_grad = True
        elif self.b == 'resnet':
            for p in self.base_model.parameters():
                p.requires_grad = False
            for p in list(self.intermidate.parameters()):
                p.requires_grad = fine_tune
            for p in list(self.last.parameters()):
                p.requires_grad = fine_tune


if __name__ == "__main__":
    model = MIML()
    miml_checkpoint = torch.load(
        '/home/lkk/code/self-critical.pytorch/checkpoint_ResNet_epoch_22.pth.tar')
    model.intermidate.load_state_dict(
        miml_checkpoint['intermidate'])
    model.last.load_state_dict(miml_checkpoint['last'])
    model.sub_concept_layer.load_state_dict(
        miml_checkpoint['sub_concept_layer'])

    del miml_checkpoint
    torch.cuda.empty_cache()

    out = model(torch.randn(8, 3, 334, 334))
    # print(out.shape)
    # summary(model.cuda(), (36, 2048, 1), 8)
