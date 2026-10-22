"""19_classifier.py — Supervised linear-probe/MLP on frozen embeddings to compare
GLOBAL vs PATCH representations for 3-group discrimination (ASC 160 / CAL 96 / Control).
70/15/15 unit-grouped split (leakage-safe, augmented controls kept with parent unit).
Logs loss/accuracy/macro-recall per epoch and test confusion matrices."""
import numpy as np, pandas as pd, torch, torch.nn as nn, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
ROOT="/home/claude/proj"; OUT=f"{ROOT}/outputs/discrim"; os.makedirs(OUT,exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
order=pd.read_csv(f"{ROOT}/outputs/embeddings/order.csv"); id2row={i:k for k,i in enumerate(order.image_id)}
ci=pd.read_csv(f"{ROOT}/outputs/embeddings/class_index.csv")
G=np.load(f"{ROOT}/outputs/embeddings/global.npy"); Pp=np.load(f"{ROOT}/outputs/embeddings/patch_pooled.npy")
rows=ci.image_id.map(id2row).values
classes=["ASC 160","CAL 96","Control"]; y=ci.group3.map({c:i for i,c in enumerate(classes)}).values
split=ci.split.values

def run(feat, name):
    X=feat[rows]
    tr,va,te=(split=="train"),(split=="val"),(split=="test")
    sc=StandardScaler().fit(X[tr])
    Xtr,Xva,Xte=[torch.tensor(sc.transform(X[m]),dtype=torch.float32) for m in (tr,va,te)]
    ytr,yva,yte=[torch.tensor(y[m]) for m in (tr,va,te)]
    cw=torch.tensor(1/np.bincount(y[tr]),dtype=torch.float32); cw=cw/cw.mean()
    net=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,3))
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.CrossEntropyLoss(weight=cw)
    hist=[]
    for ep in range(80):
        net.train(); opt.zero_grad(); out=net(Xtr); l=lossf(out,ytr); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl=lossf(net(Xva),yva).item(); tl=l.item()
            vp=net(Xva).argmax(1).numpy()
            va_acc=accuracy_score(yva,vp); va_rec=recall_score(yva,vp,average="macro",zero_division=0)
            tp=net(Xtr).argmax(1).numpy(); tr_acc=accuracy_score(ytr,tp)
        hist.append(dict(epoch=ep,train_loss=tl,val_loss=vl,train_acc=tr_acc,val_acc=va_acc,val_recall=va_rec))
    net.eval()
    with torch.no_grad(): pred=net(Xte).argmax(1).numpy()
    cm=confusion_matrix(yte,pred)
    met=dict(representation=name,
        test_accuracy=float(accuracy_score(yte,pred)),
        test_macro_recall=float(recall_score(yte,pred,average="macro",zero_division=0)),
        test_macro_precision=float(precision_score(yte,pred,average="macro",zero_division=0)),
        test_macro_f1=float(f1_score(yte,pred,average="macro",zero_division=0)),
        n_train=int(tr.sum()),n_val=int(va.sum()),n_test=int(te.sum()))
    pd.DataFrame(hist).to_csv(f"{OUT}/curves_{name}.csv",index=False)
    np.save(f"{OUT}/confusion_{name}.npy",cm)
    print(name,"test acc=%.3f recall=%.3f"%(met["test_accuracy"],met["test_macro_recall"]),"\ncm=\n",cm)
    return met

res=[run(G,"global"),run(Pp,"patch")]
json.dump({"classes":classes,"results":res},open(f"{OUT}/classifier_metrics.json","w"),indent=2)
print("saved discrim metrics")
