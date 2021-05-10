# Imports
import numpy as np
import _pickle as cPickle
import os


# PyTorch Imports
import torch
import torchvision
import torchsummary


# Create PyTorch Models
# Model: DenseNet 121 (Baseline)
class DenseNet121(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(DenseNet121, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.densenet121(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)

        # Sigmoid Activation Layer
        self.fc_sigmoid = torch.nn.Sigmoid()


        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.densenet121(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        # Activation layer
        outputs = self.fc_sigmoid(outputs)


        return outputs



# Model: ResNet 50 (Baseline)
class ResNet50(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(ResNet50, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.resnet50(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)

        # Sigmoid Activation Layer
        self.fc_sigmoid = torch.nn.Sigmoid()



        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.resnet50(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        # Activation layer
        outputs = self.fc_sigmoid(outputs)


        return outputs


# Model: VGG-16 (Baseline)
class VGG16(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(VGG16, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.vgg16(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)

        # Sigmoid Activation Layer
        self.fc_sigmoid = torch.nn.Sigmoid()



        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.vgg16(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        # Activation layer
        outputs = self.fc_sigmoid(outputs)


        return outputs



# Class: PAM Module
class PAM_Module(torch.nn.Module):
    def __init__(self, channels=1024, height=7, width=7):
        super(PAM_Module, self).__init__()

        # Init Variables
        self.channels = channels
        self.height = height
        self.width = width


        # Module Layers
        self.softmax = torch.nn.Softmax(dim=1)


        return

    

    def forward(self, v1, q1, k1):
        # We first reshape the inputs (v1, q1, k1)
        # v = Reshape([w*h,512])(v1)
        v = torch.reshape(v1, (v1.size(0), self.channels, self.height * self.width))
        
        # q = Reshape([w*h,512])(q1)
        q = torch.reshape(q1, (q1.size(0), self.channels, self.height * self.width))

        # k = Reshape([w*h,512])(k1)
        k = torch.reshape(k1, (k1.size(0), self.channels, self.height * self.width))
            
        # att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]), output_shape=(w*h,w*h))([q,k]) # 49*49
        q = torch.transpose(q, 2, 1)
        att = torch.matmul(q, k)

        # Flatten tensor before softmax, so it has shape (batch, height * width * height * width)
        att = torch.reshape(att, (att.size(0), -1))

        # Apply softmax
        # att = Lambda(lambda x:  K.softmax(x), output_shape=(w*h,w*h))(att)
        att = self.softmax(att)

        # Reshape Tensor again so it has the shape (batch, height * width, height * width)
        att = torch.reshape(att, (att.size(0), self.height * self.width, self.height * self.width))


        # out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), output_shape=(w*h,512))([att,v])
        out = torch.matmul(v, att)

        
        # out = Reshape([w,h,512])(out)
        out = torch.reshape(out, (out.size(0), self.channels, self.height, self.width))
        
        # out = Add()([out, v1])
        out = torch.add(out, v1)
        
        
        return  out



# Class: CAM Module
class CAM_Module(torch.nn.Module):
    def __init__(self, channels=1024, height=7, width=7):
        super(CAM_Module, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width


        return
    


    def forward(self, v1, q1, k1):
        # Reshape layers so they have shape (batch, channels, height * width)
        # v = Reshape([w*h,512])(v1)
        v = torch.reshape(v1, (v1.size(0), self.channels, self.height * self.width))

        # q = Reshape([w*h,512])(q1)
        q = torch.reshape(q1, (q1.size(0), self.channels, self.height * self.width))

        # k = Reshape([w*h,512])(k1)
        k = torch.reshape(k1, (k1.size(0), self.channels, self.height * self.width))


        # First transpose k so it has shape (batch, self.height * self.width, self.channels)
        k = torch.transpose(k, 2, 1)
        # att= Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,1]), output_shape=(512,512))([q,k])
        att = torch.matmul(q, k)

        # Matmul att and v so we get a Tensor with shape (batch, channels, height * width)
        # out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), output_shape=(w*h,512))([v,att])
        out = torch.matmul(att, v)

        # Reshape it to the shape (batch, channels, height, width)
        # out = Reshape([w,h,512])(out)
        out = torch.reshape(out, (out.size(0), self.channels, self.height, self.width))
        
        # Add out to v1
        # out = Add()([out, v1])
        out = torch.add(out, v1)
        
        
        return  out



# Multi-Level DAM
class MultiLevelDAM(torch.nn.Module):
    def __init__(self, channels=3, height=224, width=224, nr_classes=1, backbone="densenet121"):
        super(MultiLevelDAM, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes

        # Get the backbone
        # DenseNet121
        if backbone.lower() == "densenet121":
            self.backbone_name = backbone.lower()
            self.backbone = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
            self.att_channels = 1024
            self.att_height = 7
            self.att_width = 7
        
        # VGG-16
        elif backbone.lower() == "vgg16":
            self.backbone_name = backbone.lower()
            self.backbone = torchvision.models.vgg16(pretrained=True).features
            self.att_channels = 512
            self.att_height = 7
            self.att_width = 7
            
        
        # ResNet-50
        elif backbone.lower() == "resnet50":
            self.backbone_name = backbone.lower()
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
            self.att_channels = 2048
            self.att_height = 7
            self.att_width = 7
            


        # Models Layers
        # DenseNet121
        if self.backbone_name == "densenet121":
            # DenseBlock2; shape(batch, 512, 28, 28)
            self.denseblock2 = torch.nn.Sequential(
                # Input
                self.backbone.features.conv0,
                self.backbone.features.norm0,
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2)),
                # DenseBlock1
                self.backbone.features.denseblock1,
                # Transition1
                self.backbone.features.transition1.norm,
                torch.nn.ReLU(),
                self.backbone.features.transition1.conv,
                torch.nn.AvgPool2d(kernel_size=(2, 2)),
                # DenseBlock2
                self.backbone.features.denseblock2
            )
        
            # DenseBlock3; shape(batch, 1024, 14, 14)
            self.denseblock3 = torch.nn.Sequential(
                # Input
                self.backbone.features.conv0,
                self.backbone.features.norm0,
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2)),
                # DenseBlock1
                self.backbone.features.denseblock1,
                # Transition1
                self.backbone.features.transition1.norm,
                torch.nn.ReLU(),
                self.backbone.features.transition1.conv,
                torch.nn.AvgPool2d(kernel_size=(2, 2)),
                # DenseBlock2
                self.backbone.features.denseblock2,
                # Transition2
                self.backbone.features.transition2.norm,
                torch.nn.ReLU(),
                self.backbone.features.transition2.conv,
                torch.nn.AvgPool2d(kernel_size=(2, 2)),
                # DenseBlock3
                self.backbone.features.denseblock3
            )
            
            # DenseBlock 4; shape(batch, 1024, 7, 7) (it is the same as the features block)
            self.denseblock4 = self.backbone.features
        

        # VGG-16
        elif self.backbone_name == "vgg16":
            # Block-1; shape (256, 28, 28)
            self.vggblock1 = torch.nn.Sequential(
                *list(self.backbone.children())[0:17]
            )

            # Block-2; shape (512, 14, 14)
            self.vggblock2 = torch.nn.Sequential(
                *list(self.backbone.children())[0:24]
            )

            # Block-3; shape(512, 7, 7)
            self.vggblock3 = self.backbone



        # ResNet-50
        elif self.backbone_name == "resnet50":
            # Block-1; shape (512, 28, 28)
            self.resnetblock1 = torch.nn.Sequential(
                *(list(self.backbone.children())[0:6])
            )

            # Block-2; shape (1024, 14, 14)
            self.resnetblock2 = torch.nn.Sequential(
                *(list(self.backbone.children())[0:7])
            )


            # Block-3; shape()
            self.resnetblock3 = self.backbone



        # Average Pooling Layers
        # No. 1
        self.avg_pool2d_1 = torch.nn.AvgPool2d(kernel_size=(4, 4))

        # No. 2
        self.avg_pool2d_2 = torch.nn.AvgPool2d(kernel_size=(2, 2))

        # Batch Norm and ReLU No. 1
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=self.att_channels, eps=1e-6, momentum=0.1)
        self.relu1 = torch.nn.ReLU()

        # Batch Norm and ReLU No. 2
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=self.att_channels, eps=1e-6, momentum=0.1)
        self.relu2 = torch.nn.ReLU()


        # PAM and CAM Modules
        # PAM
        self.pam = PAM_Module(channels=self.att_channels, height=self.att_height, width=self.att_width)
        # CAM
        self.cam = CAM_Module(channels=self.att_channels, height=self.att_height, width=self.att_width)


        # FC Layers
        self.fc1 = torch.nn.Linear(in_features=(self.att_channels * self.att_height * self.att_width), out_features=self.att_channels)
        self.fc_relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=self.att_channels, out_features=int(self.att_channels/2))
        self.fc_relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(in_features=int(self.att_channels/2), out_features=self.nr_classes)
        self.fc_sigmoid = torch.nn.Sigmoid()

        # Dropout Layers
        self.drop1 = torch.nn.Dropout(0.25)
        self.drop2 = torch.nn.Dropout(0.25)

        return
    


    def forward(self, inputs):
        # DenseNet121
        if self.backbone_name == "densenet121":
            # k1
            # x1 = vgg_model.get_layer('conv4_3').output
            # x1 = AveragePooling2D()(x1)
            k1 = self.denseblock3(inputs)
            k1 = self.avg_pool2d_2(k1)
            # print(f"k1 size: {k1.size()}")

            # q1
            # x2 = vgg_model.get_layer('conv4_1').output
            # x2 = AveragePooling2D()(x2)
            q1 = self.denseblock2(inputs)
            q1 = self.avg_pool2d_1(q1)
            q1 = torch.cat((q1, q1), dim=1)
            # print(f"q1 size: {q1.size()}")


            # v1
            # x3 = vgg_model.get_layer('conv5_1').output
            v1 = self.denseblock4(inputs)
            # print(f"v1 size: {v1.size()}")



            # Backbone features
            # xlast = vgg_model.get_layer('conv5_3').output
            back_fts = self.backbone.features(inputs)
            # print(f"backbone fts size: {back_fts.size()}")
        

            # Attention Modules
            # PAM
            # att_1 = self.PAM()
            # v1, q1, k1
            # x_1 = att_1([x3,x2,x1])
            # x_1 = Activation('relu')(x_1)
            # x_1 = NormL()(x_1)
            pam_out = self.pam(v1, q1, k1)
            pam_out = self.relu1(pam_out)
            pam_out = self.batch_norm1(pam_out)
            
            
            # CAM
            # att_2= self.CAM()
            # x_2=att_2([xlast,xlast,xlast])
            # x_2 = Activation('relu')(x_2)
            # x_2 = NormL()(x_2)
            cam_out = self.cam(back_fts, back_fts, back_fts)
            cam_out = self.relu2(cam_out)
            cam_out = self.batch_norm2(cam_out)

            # Perform the average of the layers
            # x=Average()([x_1,x_2])
            att_avg = torch.add(pam_out, cam_out)
            att_avg = att_avg / 2

            # Flatten
            # x=Flatten()(x)
            att_avg = torch.reshape(att_avg, (att_avg.size(0), -1))

            # FC Layer 1
            # x = Dense(self.hidden_dim, activation='relu', name='fc6')(x)
            # x=Dropout(0.25)(x)
            outputs = self.fc1(att_avg)
            outputs = self.fc_relu1(outputs)
            outputs = self.drop1(outputs)

            # FC Layer 2
            # x = Dense(self.hidden_dim, activation='relu', name='fc7')(x)
            # x=Dropout(0.25)(x)
            outputs = self.fc2(outputs)
            outputs = self.fc_relu2(outputs)
            outputs = self.drop2(outputs)
            
            
            # Last FC Layer
            # out = Dense(self.nb_class, activation='softmax', name='fc8')(x)
            # model = Model(vgg_model.input, out)
            outputs = self.fc3(outputs)
            outputs = self.fc_sigmoid(outputs)
        

        # VGG-16
        elif self.backbone_name == "vgg16":
            # k1 - Block 2
            k1 = self.vggblock2(inputs)
            # print(f"k1 size: {k1.size()}")
            k1 = self.avg_pool2d_2(k1)
            # print(f"k1 size: {k1.size()}")

            # q1 - Block 1
            q1 = self.vggblock1(inputs)
            # print(f"q1 size: {q1.size()}")
            q1 = self.avg_pool2d_1(q1)
            q1 = torch.cat((q1, q1), dim=1)
            # print(f"q1 size: {q1.size()}")


            # v1 - Block 3
            v1 = self.vggblock3(inputs)
            # print(f"v1 size: {v1.size()}")



            # Backbone features
            back_fts = self.backbone(inputs)
            # print(f"backbone fts size: {back_fts.size()}")
        

            # Attention Modules
            # PAM
            pam_out = self.pam(v1, q1, k1)
            pam_out = self.relu1(pam_out)
            pam_out = self.batch_norm1(pam_out)
            
            
            # CAM
            cam_out = self.cam(back_fts, back_fts, back_fts)
            cam_out = self.relu2(cam_out)
            cam_out = self.batch_norm2(cam_out)

            # Perform the average of the layers
            att_avg = torch.add(pam_out, cam_out)
            att_avg = att_avg / 2

            # Flatten
            att_avg = torch.reshape(att_avg, (att_avg.size(0), -1))

            # FC Layer 1
            outputs = self.fc1(att_avg)
            outputs = self.fc_relu1(outputs)
            outputs = self.drop1(outputs)

            # FC Layer 2
            outputs = self.fc2(outputs)
            outputs = self.fc_relu2(outputs)
            outputs = self.drop2(outputs)
            
            
            # Last FC Layer
            outputs = self.fc3(outputs)
            outputs = self.fc_sigmoid(outputs)
        

        # ResNet-50
        elif self.backbone_name == "resnet50":
            # k1 - Block 2
            k1 = self.resnetblock2(inputs)
            # print(f"k1 size: {k1.size()}")
            k1 = self.avg_pool2d_2(k1)
            k1 = torch.cat((k1, k1), dim=1)
            # print(f"k1 size: {k1.size()}")


            # q1 - Block 1
            q1 = self.resnetblock1(inputs)
            # print(f"q1 size: {q1.size()}")
            q1 = self.avg_pool2d_1(q1)
            q1 = torch.cat((q1, q1, q1, q1), dim=1)
            # print(f"q1 size: {q1.size()}")


            # v1 - Block 3
            v1 = self.resnetblock3(inputs)
            # print(f"v1 size: {v1.size()}")



            # Backbone features
            back_fts = self.backbone(inputs)
            # print(f"backbone fts size: {back_fts.size()}")
        

            # Attention Modules
            # PAM
            pam_out = self.pam(v1, q1, k1)
            pam_out = self.relu1(pam_out)
            pam_out = self.batch_norm1(pam_out)
            
            
            # CAM
            cam_out = self.cam(back_fts, back_fts, back_fts)
            cam_out = self.relu2(cam_out)
            cam_out = self.batch_norm2(cam_out)

            # Perform the average of the layers
            att_avg = torch.add(pam_out, cam_out)
            att_avg = att_avg / 2

            # Flatten
            att_avg = torch.reshape(att_avg, (att_avg.size(0), -1))

            # FC Layer 1
            outputs = self.fc1(att_avg)
            outputs = self.fc_relu1(outputs)
            outputs = self.drop1(outputs)

            # FC Layer 2
            outputs = self.fc2(outputs)
            outputs = self.fc_relu2(outputs)
            outputs = self.drop2(outputs)
            
            
            # Last FC Layer
            outputs = self.fc3(outputs)
            outputs = self.fc_sigmoid(outputs)
        

        return outputs



# TODO: Erase uppon review
# Tests
# mldam = MultiLevelDAM(backbone="resnet50")
# print(mldam)
# torchsummary.summary(mldam, (3, 224, 224))
# m = torchvision.models.resnet50(pretrained=True)
# m = torch.nn.Sequential(*(list(m.children())[:-2]))
# torchsummary.summary(m, (3, 224, 224))
# m = torch.nn.Sequential(*(list(m.children())[0:7]))
# torchsummary.summary(m, (3, 224, 224))