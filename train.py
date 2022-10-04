import torch
import data_utilize6 as data1
import data_utilize2 as data2
import data_utilize3 as data3
import data_utilize4 as data4
import data_utilize5 as data5
import data_utilize1 as data6
import torch.utils.data as Datas
import Network as Network
import metrics as criterion
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_ids = [0]
device = torch.device("cuda:0")

for j in range (0,400):
    for i in range(1, 7):
        if i==1:
            data = data1.train_data
            net = Network.mainNetwork(5).to(device)
            netfixed = Network.mainNetwork(5).to(device)
        elif i==2:
            data = data2.train_data
            net = Network.mainNetwork(4).to(device)
            netfixed = Network.mainNetwork(4).to(device)
        elif i==3:
            data = data3.train_data
            net = Network.mainNetwork(4).to(device)
            netfixed = Network.mainNetwork(4).to(device)
        elif i==4:
            data = data4.train_data
            net = Network.mainNetwork(4).to(device)
            netfixed = Network.mainNetwork(4).to(device)
        elif i == 5:
            data = data5.train_data
            net = Network.mainNetwork(4).to(device)
            netfixed = Network.mainNetwork(4).to(device)
        else:
            data = data6.train_data
            net = Network.mainNetwork(4).to(device)
            netfixed = Network.mainNetwork(4).to(device)

        dataloder = Datas.DataLoader(dataset=data, batch_size=1, shuffle=True)

        net_global = Network.Encoder().to(device)



        if i == 1:
            flag1 = os.path.exists('./pkl/Client1.pkl')
            flag2 = os.path.exists('./pkl/fixed1.pkl')
        elif i == 2:
            flag1 = os.path.exists('./pkl/Client2.pkl')
            flag2 = os.path.exists('./pkl/fixed2.pkl')
        elif i == 3:
            flag1 = os.path.exists('./pkl/Client3.pkl')
            flag2 = os.path.exists('./pkl/fixed3.pkl')
        elif i == 4:
            flag1 = os.path.exists('./pkl/Client4.pkl')
            flag2 = os.path.exists('./pkl/fixed4.pkl')
        elif i == 5:
            flag1 = os.path.exists('./pkl/Client5.pkl')
            flag2 = os.path.exists('./pkl/fixed5.pkl')
        else:
            flag1 = os.path.exists('./pkl/Client6.pkl')
            flag2 = os.path.exists('./pkl/fixed6.pkl')

        if flag1 == True:
            pretrained_dict = torch.load('./pkl/Client' + str(i) + '.pkl')
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            print("loading-Client"+str(i)+",success")

        if flag2==True:
            pretrained_dict = torch.load('./pkl/fixed' + str(i) + '.pkl')
            model_dict = netfixed.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            netfixed.load_state_dict(model_dict)
            print("loading-fixed" + str(i) + ",success")


        flag = os.path.exists('./pkl/sever.pkl')
        if flag == True:
            pretrained_dict = torch.load('./pkl/sever.pkl')

            for k1, v1 in net.named_parameters():
                for k2, v2 in pretrained_dict.items():
                    str_glo = 'globalnet'
                    posnum = k1.find(str_glo)
                    if posnum != -1:
                        if k1 == k2:
                            v1 = v2
                            print("loading-sever-" + k1 + ",success")

        opt = torch.optim.Adam(net.parameters(), lr=1e-6)
        opt_fixed = torch.optim.Adam(netfixed.parameters(), lr=1e-6)
        # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, net.parameters()), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)

        criterion_CE = criterion.crossentry()
        criterion_dice = criterion.DiceMeanLoss1()
        criterion_kl = torch.nn.KLDivLoss()
        criterion_mse = torch.nn.MSELoss()


        stps=5
        for epoch in range(stps):
            for step, (image3_norm, label, name) in enumerate(dataloder):
                image3 = image3_norm.to(device).float()
                label = label.to(device).float()
                b, c, d, w, h = image3.shape

                with torch.no_grad():
                    fixed_segresult, fixed_glo_botton, fixed_loc_botton = netfixed(image3)

                segresult, glo_botton, loc_botton = net(image3)

                loss_seg_net = criterion_dice(segresult, label) + criterion_CE(segresult, label)
                loss_seg_fix2net = criterion_CE(segresult, fixed_segresult)

                loss_corre = criterion_mse(fixed_glo_botton,glo_botton)+criterion_mse(fixed_loc_botton,loc_botton)

                kl_glo = F.softmax(glo_botton, dim=-1)
                kl_loc = F.softmax(loc_botton, dim=-1)
                kl_glo_log = F.log_softmax(glo_botton, dim=-1)
                kl_loc_log = F.log_softmax(loc_botton, dim=-1)
                loss_mid_net = (criterion_kl(kl_glo_log, kl_loc) + criterion_kl(kl_loc_log, kl_glo)) / 2

                loss = loss_seg_net+loss_seg_fix2net*0.2+loss_corre+loss_mid_net

                opt.zero_grad()
                loss.backward()
                opt.step()

                torch.save(net.state_dict(), './pkl/Iter' + str(j) + 'Client' + str(i) + '.pkl')
                torch.save(net.state_dict(), './pkl/Client' + str(i) + '.pkl')

                for k1,v1 in netfixed.named_parameters():
                    # v1 = 0.9*v1+ 0.1 *net[k1].data
                    for k2,v2 in net.named_parameters():
                        if k1 == k2:
                            v1 = 0.5 * v1 + 0.5 * v2
                        # str_loc = 'localnet'
                        # posnum = k1.find(str_loc)
                        # if posnum != -1:
                        #     if k1 == k2:
                        #         v2 = 0.1 * v1 + 0.9 * v2

                fixed_segresult1, fixed_glo_botton1, fixed_loc_botton1 = netfixed(image3)
                loss_fixed = criterion_dice(fixed_segresult1, label) + criterion_CE(fixed_segresult1, label)
                opt_fixed.zero_grad()
                loss_fixed.backward()
                opt_fixed.step()

                torch.save(netfixed.state_dict(), './pkl/fixed' + str(i) + '.pkl')
                torch.save(netfixed.state_dict(), './pkl/Iter' + str(j) + 'fixed' + str(i) + '.pkl')

                print('Iter:', j, 'Client:', i, 'EPOCH:', epoch, '|Step:', step, '|loss:',
                      loss.data.cpu().numpy(), '|loss_seg_net:', loss_seg_net.data.cpu().numpy(), '|loss_seg_fix2net:',
                      loss_seg_fix2net.data.cpu().numpy(), '|loss_mid_net:', loss_mid_net.data.cpu().numpy(),
                      '|loss_corre:', loss_corre.data.cpu().numpy(), '|loss_fixed:',
                      loss_fixed.data.cpu().numpy())


                pt = image3[0, 0, :, :, :].data.cpu().numpy()
                out = sitk.GetImageFromArray(pt)
                sitk.WriteImage(out, './state/imge.nii')
                if i == 1:
                    mm = label[0, 1, :, :, :] * 1 + label[0, 2, :, :, :] * 2+ label[0, 3, :, :, :] * 3+ label[0, 4, :, :, :] * 4
                    pt = mm.data.cpu().numpy()
                    out = sitk.GetImageFromArray(pt)
                    sitk.WriteImage(out, './state/label.nii')

                    mm = segresult[0, 1, :, :, :] * 1 + segresult[0, 2, :, :, :] * 2+ segresult[0, 3, :, :, :] * 3+ segresult[0, 4, :, :, :] * 4
                    pt = mm.data.cpu().numpy()
                    out = sitk.GetImageFromArray(pt)
                    sitk.WriteImage(out, './state/segresult.nii')

                    mm = fixed_segresult[0, 1, :, :, :] * 1 + fixed_segresult[0, 2, :, :, ] * 2+ fixed_segresult[0, 3, :, :, :] * 3+ fixed_segresult[0, 4, :, :, :] * 4
                    pt = mm.data.cpu().numpy()
                    out = sitk.GetImageFromArray(pt)
                    sitk.WriteImage(out, './state/fixed_segresult.nii')
                else:
                    mm = label[0, 1, :, :, :] * 1 + label[0, 2, :, :, :] * 2 + label[0, 3, :, :, :] * 3
                    pt = mm.data.cpu().numpy()
                    out = sitk.GetImageFromArray(pt)
                    sitk.WriteImage(out, './state/label.nii')

                    mm = segresult[0, 1, :, :, :] * 1 + segresult[0, 2, :, :,
                                                               :] * 2 + segresult[0, 3, :, :, :] * 3
                    pt = mm.data.cpu().numpy()
                    out = sitk.GetImageFromArray(pt)
                    sitk.WriteImage(out, './state/segresult.nii')

                    mm = fixed_segresult[0, 1, :, :, :] * 1 + fixed_segresult[0, 2, :, :,
                                                               :] * 2 + fixed_segresult[0, 3, :, :, :] * 3
                    pt = mm.data.cpu().numpy()
                    out = sitk.GetImageFromArray(pt)
                    sitk.WriteImage(out, './state/fixed_segresult.nii')



##############
    C1 = torch.load('./pkl/Client1.pkl')
    C2 = torch.load('./pkl/Client2.pkl')
    C3 = torch.load('./pkl/Client3.pkl')
    C4 = torch.load('./pkl/Client4.pkl')
    C5 = torch.load('./pkl/Client5.pkl')
    C6 = torch.load('./pkl/Client6.pkl')

    C_avg = C1

    for k, v in C1.items():
        for k1, v1 in C5.items():
            if k == k1:
                layer_name = k
                layer_w1 = C1[k].data
                layer_w2 = C2[k].data
                layer_w3 = C3[k].data
                layer_w4 = C4[k].data
                layer_w5 = C5[k].data
                layer_w6 = C6[k].data

                posnum = layer_name.find('globalnet')
                if posnum != -1:
                    C_avg[k].data = (layer_w1 + layer_w2 + layer_w3 + layer_w4 + layer_w5 + layer_w6) / 6


    torch.save(C_avg, './pkl/sever.pkl')
    torch.save(C_avg, './pkl/Iter' + str(j) + 'sever.pkl')


