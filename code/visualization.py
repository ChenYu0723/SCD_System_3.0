# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plot_language = 'en'
# plot_language = 'cn'

if plot_language == 'en':
    label_true = 'true'
    label_pred = 'prediction'
    label_x = 'Timeslot, [15min]'
    label_y = '# Flow'

    label_train = 'train'
    label_test = 'test'
    label_epoch = 'Epoch'
else:
    label_true = '真实值'
    label_pred = '预测值'
    label_x = '时隙/15min'
    label_y = '人流量/人次'

    label_train = '训练过程'
    label_test = '测试过程'
    label_epoch = '迭代次数'


def plot_flow(station_ID, test_result, test_label, path):
    col_name = [112, 113, 114, 119, 124, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 234, 237, 238, 239, 240, 247, 248, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 312, 314, 325, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 413, 415, 416, 420, 421, 423, 426, 502, 503, 505, 507, 508, 509, 510, 511, 512, 513, 622, 623, 624, 625, 626, 628, 629, 633, 634, 635, 636, 637, 638, 639, 640, 642, 643, 644, 645, 646, 647, 648, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 738, 744, 745, 747, 749, 750, 751, 753, 820, 821, 822, 823, 824, 825, 827, 828, 830, 834, 837, 838, 840, 842, 843, 844, 845, 846, 847, 848, 849, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 937, 940, 941, 943, 944, 945, 946, 947, 948, 950, 951, 952, 1018, 1019, 1020, 1043, 1044, 1045, 1046, 1047, 1048, 1051, 1055, 1058, 1060, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1131, 1132, 1133, 1134, 1135, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1150, 1152, 1153, 1155, 1156, 1157, 1159, 1161, 1162, 1163, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1238, 1239, 1241, 1242, 1243, 1244, 1245, 1246, 1248, 1249, 1250, 1321, 1322, 1323, 1324, 1325, 1326, 1329, 1331, 1333, 1338, 1339, 1622, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057]
    col_ID = col_name.index(station_ID)
    # ==== all test result visualization
    fig = plt.figure(figsize=(10, 5))
#    ax1 = fig1.add_subplot(1,1,1)
    y_pred = test_result[100:, col_ID]
    y_true = test_label[100:, col_ID]

    plt.plot(y_true, ls='-', label=label_true)
    plt.plot(y_pred, ls='--', label=label_pred)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_x, fontsize=16)
    plt.ylabel(label_y, fontsize=16)
    plt.legend(loc='best', fontsize=12)  # loc=1
    plt.tight_layout()
    plt.savefig(path+'/station_flow_all_%s.jpg' % station_ID, dpi=300)
    plt.show()

    # ==== oneday test result visualization
    fig = plt.figure(figsize=(10, 5))
#    ax1 = fig1.add_subplot(1,1,1)
    y_pred = test_result[10:74, col_ID]
    y_true = test_label[10:74, col_ID]

    plt.plot(y_true, ls='-', label=label_true)
    plt.plot(y_pred, ls='--', label=label_pred)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_x, fontsize=16)
    plt.ylabel(label_y, fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(path+'/station_flow_oneday_%s.jpg' % station_ID, dpi=300)
    plt.show()


def plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, test_mape, path):
    # ==== train_rmse & test_rmse
    fig = plt.figure()
    plt.plot(train_rmse, 'b-', label=label_train)  # , marker='^', markersize=3
    plt.plot(test_rmse, 'r-', label=label_test)  # , marker='s', markersize=3
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_epoch, fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(path+'/train_test_rmse.jpg', dpi=300)
    plt.show()

    # ==== train_loss
    fig1 = plt.figure()
    plt.plot(train_loss, 'b-', label=label_train)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_epoch, fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(path+'/train_loss.jpg', dpi=300)
    plt.show()

    # ==== train_rmse
    # fig1 = plt.figure()
    # plt.plot(train_rmse, 'b-', label=label_train)
    # plt.xlabel('Epoch')
    # plt.ylabel('RMSE')
    # plt.legend(loc='best', fontsize=10)
    # plt.tight_layout()
    # plt.savefig(path+'/train_rmse.jpg', dpi=300)
    # plt.show()

    # ==== test accuracy
    fig1 = plt.figure()
    plt.plot(test_acc, 'b-', label=label_test)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_epoch, fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(path+'/test_acc.jpg', dpi=300)
    plt.show()

    # ==== test rmse
    fig1 = plt.figure()
    plt.plot(test_rmse, 'b-', label=label_test)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_epoch, fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    plt.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path+'/test_rmse.jpg', dpi=300)
    plt.show()

    # ==== test mae
    fig1 = plt.figure()
    plt.plot(test_mae, 'b-', label=label_test)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_epoch, fontsize=16)
    plt.ylabel('MAE', fontsize=16)
    plt.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path+'/test_mae.jpg', dpi=300)
    plt.show()

    # ==== test mape
    fig1 = plt.figure()
    plt.plot(test_mape, 'b-', label=label_test)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(label_epoch, fontsize=16)
    plt.ylabel('MAPE', fontsize=16)
    plt.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path+'/test_mape.jpg', dpi=300)
    plt.show()

