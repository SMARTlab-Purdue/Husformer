from src.eval_metrics import *
from src.utils import *
from torch.utils.data import DataLoader
from torch import nn
def eval(hyp_params, test_loader):
    model = load_model(hyp_params, name=hyp_params.name)
    model.eval()
    loader = test_loader 
    total_loss = 0.0
    criterion = getattr(nn, hyp_params.criterion)()
    results = []
    truths = []
    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, m1,m2,m3,m4 = batch_X
            eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    m1,m2,m3,m4,eval_attr = m1.cuda(), m2.cuda(), m3.cuda(),m4.cuda(), eval_attr.cuda()
            batch_size = m1.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, _ = net(m1,m2,m3,m4)
            total_loss += criterion(preds, eval_attr).item() * batch_size
            results.append(preds)
            truths.append(eval_attr) 
    avg_loss = total_loss / hyp_params.n_test 
    results = torch.cat(results)
    truths = torch.cat(truths)
    eval_hus(results, truths, True)
    return avg_loss, results, truths




