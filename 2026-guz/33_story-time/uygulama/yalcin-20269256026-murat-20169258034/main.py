"""Bölüm 33–34: Kararı açıkla, kaydı doğrula, itirazı değerlendir.
Eğitim amaçlı tarihsel veri deneyi. Gerçek kredi kararı sistemi değildir.
"""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import hashlib
import json
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

SOURCE_URL = 'https://archive.ics.uci.edu/static/public/573/south%2Bgerman%2Bcredit%2Bupdate.zip'
FEATURES = ['Vade (ay)', 'Tutar (dönüştürülmüş tarihsel değer)',
            'Hesap yok', 'Hesap: 0 DM altı', 'Hesap: 0–200 DM', 'Hesap: en az 200 DM / maaş']
ACCOUNT_LABELS = ['Hesap yok', '0 DM altı', '0–200 DM arası', 'En az 200 DM / en az 1 yıllık maaş aktarımı']

def load_data(folder=Path('.')):
    folder=Path(folder)
    file=folder/'SouthGermanCredit.asc'
    if not file.exists():
        with urlopen(SOURCE_URL,timeout=60) as response:
            archive=ZipFile(BytesIO(response.read()))
        for name in ['SouthGermanCredit.asc','codetable.txt','read_SouthGermanCredit.R']:
            if name in archive.namelist():
                (folder/name).write_bytes(archive.read(name))
    data=np.loadtxt(file,skiprows=1,dtype=int)
    assert data.shape==(1000,21)
    # Model yalnızca üç gerçek kavramı kullanır. Kategoriler sıralı sayı sayılmaz.
    X=np.column_stack([data[:,1],data[:,4]]+[(data[:,0]==v).astype(int) for v in range(1,5)])
    # Kaynak hedef: 0=bad, 1=good. Burada 1=olumsuz sonuç olarak dönüştürülür.
    y=1-data[:,-1]
    return data,X,y

def encode(duration,amount,account):
    return np.array([duration,amount]+[int(account==v) for v in range(1,5)],dtype=float)

def explain(model,row):
    """Eğitilmiş ağacın gerçekten izlediği kurallar; yaklaşık açıklama değil."""
    tree=model.tree_
    node=0
    rules=[]
    while tree.children_left[node]!=tree.children_right[node]:
        feature=int(tree.feature[node]); threshold=float(tree.threshold[node])
        go_left=bool(row[feature]<=threshold)
        if feature<2:
            text=f'{FEATURES[feature]}: {row[feature]:g} '+('≤' if go_left else '>')+f' {threshold:g}'
        else:
            text=f'{FEATURES[feature]}: '+('hayır' if go_left else 'evet')
        rules.append({'node':node,'feature':feature,'threshold':threshold,'left':go_left,'text':text})
        node=int(tree.children_left[node] if go_left else tree.children_right[node])
    score=float(model.predict_proba([row])[0,1])
    return {'rules':rules,'leaf':node,'score':score,'leaf_train_count':int(tree.n_node_samples[node])}

def evaluate(y,score,threshold=.5):
    prediction=np.asarray(score)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
    return {'threshold':threshold,'accuracy':float(accuracy_score(y,prediction)),
            'balanced_accuracy':float(balanced_accuracy_score(y,prediction)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def run(folder=Path('.')):
    folder=Path(folder)
    data,X,y=load_data(folder)
    train,test=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=42)
    model=DecisionTreeClassifier(max_depth=3,min_samples_leaf=30,random_state=42)
    model.fit(X[train],y[train])
    scores=model.predict_proba(X[test])[:,1]
    # Kaynak veri satırı 2 (başlık hariç); test kümesindedir. Kimlik bilgisi yoktur.
    # Sunum için sonuç değiştiren bir örnek seçildi; test başarısı bu seçimle ayarlanmadı.
    i=1
    assert i in test and data[i,1]==9
    correct=X[i].copy();wrong=correct.copy();wrong[0]=48
    assert model.predict_proba([wrong])[0,1]>=.5>model.predict_proba([correct])[0,1]
    t=model.tree_
    nodes=[]
    for n in range(t.node_count):
        nodes.append({'left':int(t.children_left[n]),'right':int(t.children_right[n]),
            'feature':int(t.feature[n]),'threshold':float(t.threshold[n]),
            'score':float(t.value[n][0,1]/t.value[n][0].sum()),'count':int(t.n_node_samples[n])})
    # Tarayıcıda çalışan aynı ağacın sonucu Python ile bütün gerçek satırlarda eşleşmeli.
    for row,expected in zip(X,model.predict_proba(X)[:,1]):
        n=0
        while nodes[n]['left']!=-1:
            q=nodes[n];n=q['left'] if row[q['feature']]<=q['threshold'] else q['right']
        assert abs(nodes[n]['score']-float(expected))<1e-12
    result={'data':{'name':'South German Credit','source':SOURCE_URL,'rows':1000,
        'predictor_concepts':3,'raw_predictors':20,'train':700,'test':300,
        'sha256':hashlib.sha256((folder/'SouthGermanCredit.asc').read_bytes()).hexdigest()},
        'versions':{'numpy':np.__version__,'scikit-learn':sklearn.__version__},
        'features':FEATURES,'account_labels':ACCOUNT_LABELS,'nodes':nodes,
        'metrics':evaluate(y[test],scores),
        'test':{'labels':y[test].tolist(),'scores':scores.tolist()},
        'case':{'source_row':2,'source_label':int(y[i]),'account':int(data[i,0]),
            'amount':int(data[i,4]),'correct_duration':9,'wrong_duration':48,
            'correct':explain(model,correct),'wrong':explain(model,wrong)},
        'tree_text':export_text(model,feature_names=FEATURES)}
    (folder/'sonuclar.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('700 eğitim / 300 test; üç gerçek kavram; derinlik en fazla 3.')
    print('Test sonuçları:',result['metrics'])
    for name,row in [('Canlandırma: yanlış vade 48 ay',wrong),('Kaynak kayda dönüş: 9 ay',correct)]:
        explanation=explain(model,row)
        print('\n'+name)
        for rule in explanation['rules']:print(' →',rule['text'])
        print('Yaprak olumsuz sonuç oranı:',round(explanation['score'],4))
        print('Örnek politika:', 'Ek inceleme' if explanation['score']>=.5 else 'Ön değerlendirmeyi geçer')
    print('\n',result['tree_text'])
    return model,result

if __name__=='__main__':
    import sys
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    run(Path(__file__).resolve().parent)
