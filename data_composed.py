import torch
import math
import scipy.io as sio

def prepare_mnist_data(view):
    #view = args.view
    mat_file_name = f'mnist_pcomp_{chr(96 + view)}_v2.mat'
    mat_file = sio.loadmat(mat_file_name)

    xp_key = f'positive_train_data_{chr(96 + view)}'
    xn_key = f'negative_train_data_{chr(96 + view)}'
    xt_key = f'xt_{chr(96 + view)}'
    yt_key = f'yt'

    if xp_key in mat_file and xn_key in mat_file and xt_key in mat_file and yt_key in mat_file:
        xp = torch.from_numpy(mat_file[xp_key])
        xn = torch.from_numpy(mat_file[xn_key])
        xt = torch.from_numpy(mat_file[xt_key])
        yt = torch.from_numpy(mat_file[yt_key])
    else:
        raise KeyError(f"Keys {xp_key}, {xn_key}, {xt_key}, or {yt_key} not found in the .mat file")

    positive_train_data = xp.float()
    negative_train_data = xn.float()
    positive_test_data = xt[:len(xt) // 2].float()
    negative_test_data = xt[len(xt) // 2:].float()
    num_train = xp.shape[0] + xn.shape[0]
    num_test = xt.shape[0]
    dim = xp.shape[1]
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim

def prepare_kmnist_data(view):
    #view = args.view
    mat_file_name = f'kmnist_pcomp_{chr(96 + view)}_v2.mat'
    mat_file = sio.loadmat(mat_file_name)

    xp_key = f'positive_train_data_{chr(96 + view)}'
    xn_key = f'negative_train_data_{chr(96 + view)}'
    xt_key = f'xt_{chr(96 + view)}'
    yt_key = f'yt'

    if xp_key in mat_file and xn_key in mat_file and xt_key in mat_file and yt_key in mat_file:
        xp = torch.from_numpy(mat_file[xp_key])
        xn = torch.from_numpy(mat_file[xn_key])
        xt = torch.from_numpy(mat_file[xt_key])
        yt = torch.from_numpy(mat_file[yt_key])
    else:
        raise KeyError(f"Keys {xp_key}, {xn_key}, {xt_key}, or {yt_key} not found in the .mat file")

    positive_train_data = xp.float()
    negative_train_data = xn.float()
    positive_test_data = xt[:len(xt) // 2].float()
    negative_test_data = xt[len(xt) // 2:].float()
    num_train = xp.shape[0] + xn.shape[0]
    num_test = xt.shape[0]
    dim = xp.shape[1]
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim

def prepare_fashion_data(view):

    mat_file_name = f'fashion_pcomp_{chr(96 + view)}_v2.mat'
    mat_file = sio.loadmat(mat_file_name)

    xp_key = f'positive_train_data_{chr(96 + view)}'
    xn_key = f'negative_train_data_{chr(96 + view)}'
    xt_key = f'xt_{chr(96 + view)}'
    yt_key = f'yt'

    if xp_key in mat_file and xn_key in mat_file and xt_key in mat_file and yt_key in mat_file:
        xp = torch.from_numpy(mat_file[xp_key])
        xn = torch.from_numpy(mat_file[xn_key])
        xt = torch.from_numpy(mat_file[xt_key])
        yt = torch.from_numpy(mat_file[yt_key])
    else:
        raise KeyError(f"Keys {xp_key}, {xn_key}, {xt_key}, or {yt_key} not found in the .mat file")

    positive_train_data = xp.float()
    negative_train_data = xn.float()
    positive_test_data = xt[:len(xt) // 2].float()
    negative_test_data = xt[len(xt) // 2:].float()
    num_train = xp.shape[0] + xn.shape[0]
    num_test = xt.shape[0]
    dim = xp.shape[1]
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim

def prepare_cifar_data(view):

    mat_file_name = f'cifar10_pcomp_{chr(96 + view)}_v2.mat'
    mat_file = sio.loadmat(mat_file_name)

    xp_key = f'positive_train_data_{chr(96 + view)}'
    xn_key = f'negative_train_data_{chr(96 + view)}'
    xt_key = f'xt_{chr(96 + view)}'
    yt_key = f'yt'

    if xp_key in mat_file and xn_key in mat_file and xt_key in mat_file and yt_key in mat_file:
        xp = torch.from_numpy(mat_file[xp_key])
        xn = torch.from_numpy(mat_file[xn_key])
        xt = torch.from_numpy(mat_file[xt_key])
        yt = torch.from_numpy(mat_file[yt_key])
    else:
        raise KeyError(f"Keys {xp_key}, {xn_key}, {xt_key}, or {yt_key} not found in the .mat file")

    positive_train_data = xp.float()
    negative_train_data = xn.float()
    positive_test_data = xt[:len(xt) // 2].float()
    negative_test_data = xt[len(xt) // 2:].float()
    num_train = xp.shape[0] + xn.shape[0]
    num_test = xt.shape[0]
    dim = xp.shape[1]
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim

def prepare_bank_data(args):
    mat_file_name = f'banknote_authentication.mat'
    mat_file = sio.loadmat(mat_file_name)

    xp_key = f'xp'
    xn_key = f'xn'
    xt_key = f'xt'
    yt_key = f'yt'

    if xp_key in mat_file and xn_key in mat_file and xt_key in mat_file and yt_key in mat_file:
        xp = torch.from_numpy(mat_file[xp_key])
        xn = torch.from_numpy(mat_file[xn_key])
        xt = torch.from_numpy(mat_file[xt_key])
        yt = torch.from_numpy(mat_file[yt_key])
    else:
        raise KeyError(f"Keys {xp_key}, {xn_key}, {xt_key}, or {yt_key} not found in the .mat file")

    positive_train_data = xp.float()
    negative_train_data = xn.float()
    positive_test_data = xt[:len(xt) // 2].float()
    negative_test_data = xt[len(xt) // 2:].float()
    num_train = xp.shape[0] + xn.shape[0]
    num_test = xt.shape[0]
    dim = xp.shape[1]
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim

def prepare_uci_data(args):
    dataname= f'{args.ds}.mat'
    current_data = sio.loadmat(dataname)
    data = current_data['data']
    label = current_data['label']
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()
    labels = label.argmax(dim=1)
    labels[labels==10] = 0
    train_index = torch.arange(labels.shape[0])
    positive_index = torch.cat((torch.cat((torch.cat((torch.cat((train_index[labels==0],train_index[labels==2]),dim=0),train_index[labels==4]),dim=0),train_index[labels==6]),dim=0),train_index[labels==8]),dim=0)
    negative_index = torch.cat((torch.cat((torch.cat((torch.cat((train_index[labels==1],train_index[labels==3]),dim=0),train_index[labels==5]),dim=0),train_index[labels==7]),dim=0),train_index[labels==9]),dim=0)
    positive_data = data[positive_index,:]
    negative_data = data[negative_index,:]
    np = positive_data.shape[0]
    nn = negative_data.shape[0]
    positive_data = positive_data[torch.randperm(positive_data.shape[0])]
    negative_data = negative_data[torch.randperm(negative_data.shape[0])]
    train_p = int(np*0.8)
    train_n = int(nn*0.8)
    positive_train_data  = positive_data[:train_p,:]
    positive_test_data = positive_data[train_p:,:]
    negative_train_data = negative_data[:train_n,:]
    negative_test_data = negative_data[train_n:,:]
    num_train = train_p +train_n
    num_test = (np+nn)-num_train
    dim = positive_train_data.shape[1]
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test,dim

def synth_test_dataset(prior, positive_test_data, negative_test_data):
    num_p = positive_test_data.shape[0]
    num_n = negative_test_data.shape[0]
    if prior == 0.2:
        nn = num_n
        np = int(num_n*0.25)
    elif prior == 0.5:
        if num_p > num_n:
          nn = num_n
          np = num_n
        else:
          nn = num_p
          np = num_p
    elif prior == 0.8:
        np = num_p
        nn = int(num_p*0.25)
    else:
        np = num_p
        nn = num_n
    x = torch.cat((positive_test_data[:np, :], negative_test_data[:nn, :]), dim=0)
    y = torch.cat((torch.ones(np), -torch.ones(nn)), dim=0)
    return x, y

def synth_pcomp_train_dataset(args, prior, positive_train_data, negative_train_data):
    positive_train_data = positive_train_data[torch.randperm(positive_train_data.shape[0])]
    negative_train_data = negative_train_data[torch.randperm(negative_train_data.shape[0])]  # shuffle the given data

    pn_prior = prior * (1 - prior)
    pp_prior = prior * prior
    nn_prior = (1 - prior) * (1 - prior)
    n = args.n
    total_prior = pn_prior + pp_prior + nn_prior

    pn_number = math.floor(n * (pn_prior / total_prior))
    pp_number = math.floor(n * (pp_prior / total_prior))
    nn_number = n - pn_number - pp_number

    xpn_p = positive_train_data[:pn_number, :]
    xpn_n = negative_train_data[:pn_number, :]

    if (pn_number + 2 * pp_number) <= positive_train_data.shape[0]:
        xpp_p1 = positive_train_data[pn_number:(pn_number + pp_number), :]
        xpp_p2 = positive_train_data[(pn_number + pp_number):(pn_number + 2 * pp_number), :]
    else:
        raise ValueError("Positive Class Number Fail")

    if (pn_number + 2 * nn_number) <= negative_train_data.shape[0]:
        xnn_n1 = negative_train_data[pn_number:(pn_number + nn_number), :]
        xnn_n2 = negative_train_data[(pn_number + nn_number):(pn_number + 2 * nn_number), :]
    else:
        raise ValueError("Negative Class Number Fail")

    x1 = torch.cat((xpn_p, xpp_p1, xnn_n1), dim=0)
    x2 = torch.cat((xpn_n, xpp_p2, xnn_n2), dim=0)
    real_y1 = torch.cat((torch.ones(pn_number), torch.ones(pp_number), -torch.ones(nn_number)), dim=0)
    real_y2 = torch.cat((-torch.ones(pn_number), torch.ones(pp_number), -torch.ones(nn_number)), dim=0)
    given_y1 = torch.cat((torch.ones(pn_number), torch.ones(pp_number), torch.ones(nn_number)), dim=0)
    given_y2 = torch.cat((-torch.ones(pn_number), -torch.ones(pp_number), -torch.ones(nn_number)), dim=0)

    print(x1.shape, x2.shape, real_y1.shape, real_y2.shape, given_y1.shape, given_y2.shape)
    return x1, x2, real_y1, real_y2, given_y1, given_y2

def generate_pcomp_data(args):
    x1_list, x2_list, real_y1_list, real_y2_list, given_y1_list, given_y2_list, xt_list, yt_list = [], [], [], [], [], [], [], []
    dim = None
    if args.uci == 0:# image datasets: mnist, fashion, kmnist...
        #dataset: MNIST
        if isinstance(args.m, int):
            views = range(1, args.m + 1)
        else:
            views = args.m
        for view in views:
            #view = view + 1
            if args.ds == 'mnist':
                positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim = prepare_mnist_data(view)
            elif args.ds == 'kmnist':
                positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim = prepare_kmnist_data(view)
            elif args.ds == 'fashion':
                positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim = prepare_fashion_data(view)
            elif args.ds == 'cifar':
                positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim = prepare_cifar_data(view)

            print(f"#original train_v{view}:", num_train, f"#original test_v{view}:", num_test)
            print(f"#all train positive_v{view}:", positive_train_data.shape, f"#all train negative_v{view}:",
                  negative_train_data.shape)

            x1, x2, real_y1, real_y2, given_y1, given_y2 = synth_pcomp_train_dataset(
                args, args.prior, positive_train_data, negative_train_data)
            xt, yt = synth_test_dataset(args.prior, positive_test_data, negative_test_data)

            print(f"#test positive_v{view}:", (yt == 1).sum(), f"#test negative_v{view}:", (yt == -1).sum())
            print(f"test shape_v{view}:", xt.shape)

            x1_list.append(x1)
            x2_list.append(x2)
            real_y1_list.append(real_y1)
            real_y2_list.append(real_y2)
            given_y1_list.append(given_y1)
            given_y2_list.append(given_y2)
            xt_list.append(xt)
            yt_list.append(yt)


    elif args.uci == 1:  #upload uci multi-class datasets (.mat, .arff): usps, pendigits,opdigits,semion...
        if args.ds == 'bank':
            positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim = prepare_bank_data(
                args)
        elif args.ds == 'cnae9':
            positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test, dim = prepare_uci_data(
                args)
        print(f"#original train:", num_train, f"#original test:", num_test)
        print(f"#all train positive:", positive_train_data.shape, f"#all train negative:",
              negative_train_data.shape)

        x1, x2, real_y1, real_y2, given_y1, given_y2 = synth_pcomp_train_dataset(
            args, args.prior, positive_train_data, negative_train_data)
        xt, yt = synth_test_dataset(args.prior, positive_test_data, negative_test_data)

        print(f"#test positive:", (yt == 1).sum(), f"#test negative:", (yt == -1).sum())
        print(f"test shape:", xt.shape)

        x1_list.append(x1)
        x2_list.append(x2)
        real_y1_list.append(real_y1)
        real_y2_list.append(real_y2)
        given_y1_list.append(given_y1)
        given_y2_list.append(given_y2)
        xt_list.append(xt)
    return x1_list, x2_list, real_y1_list, real_y2_list, given_y1_list, given_y2_list, xt_list, yt, dim



