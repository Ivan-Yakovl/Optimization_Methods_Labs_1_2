"""ЛР 1. Непрерывный план двухэтапного гильотинного раскроя листов."""
import itertools, json
import numpy as np
from common import *

def generate_data():
    rng = np.random.default_rng(10128)
    items = [dict(id=f'P{i+1}', width=w, height=h, demand=int(rng.integers(25, 46)))
             for i,(w,h) in enumerate([(500,500),(600,500),(750,500),(500,750),(600,1000)])]
    stocks = [dict(id=f'S{i+1}',width=w,height=h,available=a)
              for i,(w,h,a) in enumerate([(1000,1500,9),(1200,1500,8),(1500,1500,8),(1500,2000,40)])]
    return dict(seed=10128, items=items, stocks=stocks,
                generation='numpy.default_rng(10128); demand integers [25,46); geometry and stock fixed')

def load_data():
    d=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    require(len(d['items'])==5 and len(d['stocks'])==4,'Вариант 10 требует 5 деталей и 4 типа листов')
    for item in d['items']+d['stocks']:
        for key in ['width','height']:
            require(isinstance(item[key],int) and item[key]>0,f'{item["id"]}: {key} должен быть положительным целым числом мм')
    finite_array('Спрос',[i['demand'] for i in d['items']],(5,),True)
    finite_array('Запасы листов',[s['available'] for s in d['stocks']],(4,),True)
    return d

def enumerate_patterns(d):
    """Все сочетания однородных по высоте полос; ориентация фиксирована."""
    patterns=[]; items=d['items']
    for si,s in enumerate(d['stocks']):
        rows=[]
        for h in sorted({i['height'] for i in items}):
            idx=[k for k,i in enumerate(items) if i['height']==h]
            for nums in itertools.product(*[range(s['width']//items[k]['width']+1) for k in idx]):
                if not any(nums) or sum(n*items[k]['width'] for n,k in zip(nums,idx))>s['width']: continue
                counts=[0]*5
                for n,k in zip(nums,idx): counts[k]=n
                rows.append(dict(height=h,counts=counts))
        unique={}
        def visit(start, height, counts, layout):
            if any(counts):
                key=tuple(counts)
                if key not in unique: unique[key]=list(layout)
            for ri in range(start,len(rows)):
                row=rows[ri]
                if height+row['height']<=s['height']:
                    visit(ri,height+row['height'],[a+b for a,b in zip(counts,row['counts'])],layout+[row])
        visit(0,0,[0]*5,[])
        for counts,layout in sorted(unique.items()):
            patterns.append(dict(id=f'K{len(patterns)+1:03}', stock=si, counts=list(counts), rows=layout))
    return patterns

def build_model(d, patterns):
    A=np.array([p['counts'] for p in patterns],float).T
    S=np.array([[int(p['stock']==i) for p in patterns] for i in range(4)],float)
    cost=np.array([d['stocks'][p['stock']]['width']*d['stocks'][p['stock']]['height']/1e6 for p in patterns])
    demand=np.array([i['demand'] for i in d['items']],float)
    cap=np.array([s['available'] for s in d['stocks']],float)
    return cost,np.vstack([A,S]),np.r_[demand,[-np.inf]*4],np.r_[demand,cap],[np.inf]*len(patterns)

def verify(d, patterns, x, name, objective):
    x=solution_vector(x,len(patterns)); produced=np.zeros(5); used=np.zeros(4); area=0; rows=[]
    for p,v in zip(patterns,x):
        s=d['stocks'][p['stock']]; counts=np.zeros(5,int); height=0
        for strip in p['rows']:
            ns=finite_array(p['id']+' counts',strip['counts'],(5,),True)
            require(np.all(ns==np.rint(ns)),p['id']+': число деталей в полосе нецелое')
            require(np.isfinite(strip['height']) and strip['height']>0,p['id']+': неверная высота полосы')
            require(sum(n*i['width'] for n,i in zip(strip['counts'],d['items']))<=s['width'],p['id']+': переполнение полосы по ширине')
            require(all(n==0 or i['height']==strip['height'] for n,i in zip(strip['counts'],d['items'])),p['id']+': несовпадение высот деталей и полосы')
            height+=strip['height']; counts+=strip['counts']
        require(height<=s['height'] and list(counts)==p['counts'],p['id']+': неверная геометрия схемы')
        produced+=v*counts; used[p['stock']]+=v
        area+=v*s['width']*s['height']/1e6
        rows.append(boundrow(p['id'],v))
    for i,v in zip(d['items'],produced): rows.append(boundrow(i['id']+'_demand',v,i['demand'],i['demand']))
    for s,v in zip(d['stocks'],used): rows.append(boundrow(s['id']+'_stock',v,0,s['available']))
    return checks(name,rows,area,objective,dict(used_sheets=used.tolist(), produced=produced.tolist(), geometry_valid=True))

def main():
    import copy
    environment(); d=load_data()
    patterns=enumerate_patterns(d); dump(OUT/'patterns.json',patterns)
    csvwrite(OUT/'patterns.csv',[dict(pattern=p['id'],stock=d['stocks'][p['stock']]['id'],**{f'P{i+1}':v for i,v in enumerate(p['counts'])}) for p in patterns])
    base=solve(*build_model(d,patterns),'base'); shadow=np.array(base['row_dual'][5:])
    scarce=int(np.argmin(shadow))
    used=np.array([sum(x for p,x in zip(patterns,base['x']) if p['stock']==i) for i in range(4)])
    active=[i for i in range(4) if i!=scarce and abs(used[i]-d['stocks'][i]['available'])<=TOL]
    second=int(active[0] if active else next(i for i in range(4) if i!=scarce))
    scenarios={'base':d,'deficit':copy.deepcopy(d),'stress':copy.deepcopy(d)}
    scenarios['deficit']['stocks'][scarce]['available']*=.8
    scenarios['stress']['stocks'][scarce]['available']*=.6
    scenarios['stress']['stocks'][second]['available']*=.8
    summary=[]; allresults={}
    for name,sd in scenarios.items():
        dump(OUT/(name+'_data.json'),sd)
        r=base if name=='base' else solve(*build_model(sd,patterns),name)
        v=verify(sd,patterns,r['x'],name,r['objective'])
        useful=sum(i['width']*i['height']*i['demand']/1e6 for i in sd['items'])
        delta=np.array([s['available'] for s in sd['stocks']])-np.array([s['available'] for s in d['stocks']])
        row=dict(scenario=name,status=r['status'],area_m2=r['objective'],useful_m2=useful,waste_m2=r['objective']-useful,
                 sheets=sum(v['used_sheets']),objective_change=r['objective']-base['objective'],shadow_prediction=float(shadow@delta),seconds=r['seconds'])
        summary.append(row); allresults[name]=dict(result=r,verification=v,data=sd,summary=row)
        csvwrite(OUT/(name+'_solution.csv'),[dict(pattern=p['id'],stock=sd['stocks'][p['stock']]['id'],sheets=x,**{f'P{i+1}':n for i,n in enumerate(p['counts'])}) for p,x in zip(patterns,r['x'])])
    csvwrite(OUT/'scenarios.csv',summary)
    dual=[]
    for i,s in enumerate(d['stocks']):
        changed=copy.deepcopy(d); changed['stocks'][i]['available']+=.01
        rr=solve(*build_model(changed,patterns),f'shadow_{s["id"]}')
        verify(changed,patterns,rr['x'],f'shadow_{s["id"]}',rr['objective'])
        dump(OUT/f'shadow_{s["id"]}_data.json',changed)
        dual.append(dict(stock=s['id'],capacity=s['available'],used=allresults['base']['verification']['used_sheets'][i],
                         marginal_m2_per_sheet=shadow[i],predicted_change=shadow[i]*.01,actual_change=rr['objective']-base['objective']))
    csvwrite(OUT/'shadow_prices.csv',dual)
    dump(OUT/'summary.json',dict(patterns=len(patterns),scarce=scarce,second=second,scenarios=allresults,shadows=dual))
    fig,ax=plt.subplots(figsize=(8,4)); pos=np.arange(4); bw=.25
    for k,(name,z) in enumerate(allresults.items()): ax.bar(pos+k*bw,z['verification']['used_sheets'],bw,label=name)
    ax.set_xticks(pos+bw,[s['id'] for s in d['stocks']]); ax.set_ylabel('Листы, эквивалент шт.'); ax.legend(); savefig('stock_use.png')
    active=[(p,x) for p,x in zip(patterns,base['x']) if x>TOL]
    ncols=4; nrows=int(np.ceil(len(active)/ncols))
    fig,axs=plt.subplots(nrows,ncols,figsize=(10,3.8*nrows),squeeze=False)
    from matplotlib.patches import Rectangle
    colors=plt.cm.Set2(np.linspace(0,1,5))
    for ax,(p,x) in zip(axs.ravel(),active):
        s=d['stocks'][p['stock']]; y=0
        ax.add_patch(Rectangle((0,0),s['width'],s['height'],fill=False,lw=1.8))
        for strip in p['rows']:
            xx=0
            for i,n in enumerate(strip['counts']):
                it=d['items'][i]
                for _ in range(n):
                    ax.add_patch(Rectangle((xx,y),it['width'],it['height'],facecolor=colors[i],edgecolor='black',lw=.5))
                    ax.text(xx+it['width']/2,y+it['height']/2,it['id'],ha='center',va='center',fontsize=8); xx+=it['width']
            y+=strip['height']
        ax.set(xlim=(0,s['width']),ylim=(0,s['height']),title=f'{p["id"]} {s["id"]}\nx={x:.3f}'); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for ax in axs.ravel()[len(active):]: ax.axis('off')
    savefig('cutting_patterns.png')
    print('ЛР 1:',[(s['scenario'],round(s['area_m2'],6)) for s in summary])

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate-data',action='store_true',help='Восстановить исходный data.json по seed 10128')
    args=parser.parse_args()
    if args.regenerate_data: dump(ROOT/'data.json',generate_data())
    main()
    from reporting import build_report
    build_report()
