import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset


class TranslinkDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_C_GNN, X_P_GNN, X_T_GNN, X_Ext_GNN, X_C_ST, X_P_ST, X_T_ST, Y_GNN):
        """
        Args:
            datasets = [X_C_GNN, X_P_GNN, X_T_GNN, X_Ext_GNN]
        """
        self._X_C_GNN = X_C_GNN
        self._X_P_GNN = X_P_GNN
        self._X_T_GNN = X_T_GNN
        self._X_Ext_GNN = X_Ext_GNN
        self._X_C_ST = X_C_ST
        self._X_P_ST = X_P_ST
        self._X_T_ST = X_T_ST
        self._Y_GNN = Y_GNN

    def __len__(self):
        return len(self._X_C_GNN)

    def __getitem__(self, idx):

        return self._X_C_GNN[idx], self._X_P_GNN[idx], self._X_T_GNN[idx], self._X_Ext_GNN[idx], self._X_C_ST[idx],\
               self._X_P_ST[idx], self._X_T_ST[idx], self._Y_GNN[idx]

# 519 for whole
class GCModel(nn.Module):
    def __init__(self,  mask_in, mask_out, stop_distribution, stop_number_location, c_conf_ST=(3, 2, 100, 100),
                 c_conf_G=(3, 2, 389, 10), p_conf_ST=(3, 2, 100, 100), p_conf_G=(3, 2, 389, 10),
                 t_conf_ST=(3, 2, 100, 100), t_conf_G=(3, 2, 389, 10), external_dim=11,
                 nb_conv_unit=4, nb_residual_unit=3):
        super(GCModel, self).__init__()

        # mask_in, mask_out, stop_distribution, stop_number_location
        self.mask_in = mask_in
        self.mask_out = mask_out
        self.stop_distribution = stop_distribution
        self.stop_number_location = stop_number_location

        # initialize GNN part
        self.external_dim = external_dim
        self.c_conf_G = c_conf_G
        self.p_conf_G = p_conf_G
        self.t_conf_G = t_conf_G

        self.nb_inputs_G = c_conf_G[0]
        self.nb_flow_G, self.stop_nums_G, self.closest_stop_nums_G = c_conf_G[1], 389, c_conf_G[3]

        self.relu = torch.relu
        self.tanh = torch.tanh

        self.lin_front = c_conf_G[2] - 2 - 2 - 2 - 2 - 1

        # Branch c
        self.c_way_G = self.make_one_way_G(in_channels=self.nb_inputs_G * self.nb_flow_G)
        self.p_way_G = self.make_one_way_G(in_channels=self.nb_inputs_G * self.nb_flow_G)
        self.t_way_G = self.make_one_way_G(in_channels=self.nb_inputs_G * self.nb_flow_G)

        # initialize STCNN part
        self.nb_residual_unit = nb_residual_unit
        self.c_conf_ST = c_conf_ST
        self.p_conf_ST = p_conf_ST
        self.t_conf_ST = t_conf_ST

        self.nb_flow_ST, self.map_height_ST, self.map_width_ST = c_conf_ST[1], c_conf_ST[2], c_conf_ST[3]

        self.c_way_ST = self.make_one_way_ST(in_channels=self.c_conf_ST[0] * self.nb_flow_ST)
        self.p_way_ST = self.make_one_way_ST(in_channels=self.p_conf_ST[0] * self.nb_flow_ST)
        self.t_way_ST = self.make_one_way_ST(in_channels=self.t_conf_ST[0] * self.nb_flow_ST)

        # External Components
        if self.external_dim != None and self.external_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
                ('embd', nn.Linear(self.external_dim, 10, bias=True)),
                ('relu1', nn.ReLU()),
                ('fc', nn.Linear(10, self.nb_flow_G * self.stop_nums_G, bias=True)),
                ('relu2', nn.ReLU()),
            ]))

        self.mo_fusion = self.last_fusion(2, self.stop_nums_G)

    def make_one_way_G(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('convmo1', ConvModule()),
            ('convmo2', ConvModule()),
            ('convmo5', ConvModule()),
            ('convmo6', ConvModule()),
            ('convmo3', nn.Conv2d(in_channels=6, out_channels=2, kernel_size=2, stride=1, padding=0, bias=True)),
            ('squeeze1', _squeeze_layer()),
            ('convmo4', nn.Linear(self.lin_front, self.stop_nums_G, bias=True)),
            ('FusionLayer', TrainableEltwiseLayer(n=2, h=self.stop_nums_G))
        ]))

    def make_one_way_ST(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=in_channels, out_channels=64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter=64, repetations=self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels=64, out_channels=2)),
            ('FusionLayer', TrainableEltwiseLayer_ST(n=self.nb_flow_ST, h=self.map_height_ST, w=self.map_width_ST))
        ]))

    def last_fusion(self, n, h):

        return TrainableEltwiseLayer(n, h)

    def forward(self, data):
        """
        [self._X_C_GNN[idx], self._X_P_GNN[idx], self._X_T_GNN[idx], self._X_Ext_GNN[idx],
                  self._X_C_ST[idx], self._X_P_ST[idx], self._X_T_ST[idx]]
        """
        input_c_G = data[0]
        input_p_G = data[1]
        input_t_G = data[2]
        external_input = data[3]
        input_c_ST = data[4]
        input_p_ST = data[5]
        input_t_ST = data[6]

        # Three-way Convolution
        main_output = torch.zeros(1, 2, self.stop_nums_G)

        # temporal closeness GNN
        main_output_c = torch.zeros(1, 2, self.stop_nums_G)

        input_c_G = input_c_G.view(-1, self.nb_inputs_G * self.nb_flow_G,
                                   self.stop_nums_G, self.closest_stop_nums_G)
        out_c_G = self.c_way_G(input_c_G)
        main_output_c += out_c_G

        # temporal closeness ST
        input_c_ST = input_c_ST.view(-1, self.c_conf_ST[0] * 2, self.map_height_ST, self.map_width_ST)
        out_c_ST = self.c_way_ST(input_c_ST)

        out_c_ST[0][0][self.mask_in] = 0
        out_c_ST[0][1][self.mask_out] = 0

        alpha = 0.8
        for j in range(len(self.stop_distribution)):
            stop = self.stop_distribution[j]
            stop_lat = self.stop_number_location[stop][0]
            stop_lon = self.stop_number_location[stop][1]
            main_output_c[0][0][j] = alpha * main_output_c[0][0][j] + (1 - alpha) * out_c_ST[0][0][stop_lat - 1][
                stop_lon - 1]
            main_output_c[0][1][j] = alpha * main_output_c[0][1][j] + (1 - alpha) * out_c_ST[0][1][stop_lat - 1][
                stop_lon - 1]

        # period GNN
        main_output_p = torch.zeros(1, 2, self.stop_nums_G)

        input_p_G = input_p_G.view(-1, self.nb_inputs_G * self.nb_flow_G,
                                   self.stop_nums_G, self.closest_stop_nums_G)
        out_p_G = self.p_way_G(input_p_G)
        main_output_p += out_p_G

        # period ST
        input_p_ST = input_p_ST.view(-1, self.p_conf_ST[0] * 2, self.map_height_ST, self.map_width_ST)
        out_p_ST = self.p_way_ST(input_p_ST)

        out_p_ST[0][0][self.mask_in] = 0
        out_p_ST[0][1][self.mask_out] = 0

        alpha = 0.8
        for j in range(len(self.stop_distribution)):
            stop = self.stop_distribution[j]
            stop_lat = self.stop_number_location[stop][0]
            stop_lon = self.stop_number_location[stop][1]
            main_output_p[0][0][j] = alpha * main_output_p[0][0][j] + (1 - alpha) * out_p_ST[0][0][stop_lat - 1][
                stop_lon - 1]
            main_output_p[0][1][j] = alpha * main_output_p[0][1][j] + (1 - alpha) * out_p_ST[0][1][stop_lat - 1][
                stop_lon - 1]

        # trend GNN
        main_output_t = torch.zeros(1, 2, self.stop_nums_G)

        input_t_G = input_t_G.view(-1, self.nb_inputs_G * self.nb_flow_G,
                                   self.stop_nums_G, self.closest_stop_nums_G)
        out_t_G = self.c_way_G(input_t_G)
        main_output_t += out_t_G

        # trend ST
        input_t_ST = input_t_ST.view(-1, self.t_conf_ST[0] * 2, self.map_height_ST, self.map_width_ST)
        out_t_ST = self.t_way_ST(input_t_ST)

        out_t_ST[0][0][self.mask_in] = 0
        out_t_ST[0][1][self.mask_out] = 0

        alpha = 0.8
        for j in range(len(self.stop_distribution)):
            stop = self.stop_distribution[j]
            stop_lat = self.stop_number_location[stop][0]
            stop_lon = self.stop_number_location[stop][1]
            main_output_t[0][0][j] = alpha * main_output_p[0][0][j] + (1 - alpha) * out_t_ST[0][0][stop_lat - 1][
                stop_lon - 1]
            main_output_t[0][1][j] = alpha * main_output_p[0][1][j] + (1 - alpha) * out_t_ST[0][1][stop_lat - 1][
                stop_lon - 1]

        # external input
        external_output = self.external_ops(external_input)
        external_output = self.relu(external_output)
        external_output = external_output.view(-1, self.nb_flow_G, self.stop_nums_G)
        main_output += external_output

        main_output = main_output + main_output_c + main_output_p + main_output_t

        main_output = self.mo_fusion(main_output)

        main_output = torch.sigmoid(main_output)

        return main_output


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        # self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        # if self.has_bn:
        #    x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x


class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual  # short cut

        return out


class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x


# Matrix-based fusion
class TrainableEltwiseLayer_ST(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer_ST, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w),
                                    requires_grad=True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        x = x * self.weights  # element-wise multiplication

        return x

class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=2, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = torch.relu
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, input_c):
        x = input_c.view(-1, 6, (input_c.shape)[2], (input_c.shape)[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max1(x)

        return x


class _squeeze_layer(nn.Module):
    def __init__(self):
        super(_squeeze_layer, self).__init__()

    def forward(self, x):
        return x.squeeze()


# Matrix-based fusion
class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h),
                                    requires_grad=True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h
        x = x * self.weights  # element-wise multiplication

        return x
