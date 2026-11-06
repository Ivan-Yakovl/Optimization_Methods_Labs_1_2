"""ЛР 2. Перевозки с ограничениями маршрутов и штрафуемым дефицитом."""
import copy, json, time
import numpy as np
from common import *

def generate_data():
    rng=np.random.default_rng(10128)
    return dict(seed=10128,warehouses=[f'W{i+1}' for i in range(4)],shops=[f'M{j+1}' for j in range(6)],
        supply=rng.integers(90,121,4).tolist(),demand=rng.integers(65,96,6).tolist(),
        cost=rng.integers(8,46,(4,6)).tolist(),capacity=rng.integers(25,61,(4,6)).tolist(),
        penalty=rng.integers(80,131,6).tolist(),
        generation='default_rng(10128), draws in order: supply [90,121), demand [65,96), cost [8,46), capacity [25,61), penalty [80,131)')

def build_model(d):
    A=np.zeros((10,30))
    for i in range(4): A[i,6*i:6*i+6]=1
    for j in range(6): A[4+j,j:24:6]=1; A[4+j,24+j]=1
    demand=np.array(d['demand'],float)
    return np.r_[np.ravel(d['cost']),d['penalty']],A,np.r_[[-np.inf]*4,demand],np.r_[d['supply'],demand],np.r_[np.ravel(d['capacity']),demand]

def load_data():
    d=json.loads((ROOT/'data.json').read_text(encoding='utf-8'))
    require(len(d['warehouses'])==4 and len(d['shops'])==6,'Вариант 10 требует 4 склада и 6 магазинов')
    for key,shape in [('supply',(4,)),('demand',(6,)),('cost',(4,6)),('capacity',(4,6)),('penalty',(6,))]:
        finite_array(key,d[key],shape,True)
    return d

def verify(d,x,name,objective):
    x=solution_vector(x,30)
    flow=np.array(x[:24]).reshape(4,6); deficit=np.array(x[24:]); rows=[]
    for i in range(4):
        rows.append(boundrow(f'supply_W{i+1}',flow[i].sum(),0,d['supply'][i]))
        for j in range(6): rows.append(boundrow(f'W{i+1}_M{j+1}',flow[i,j],0,d['capacity'][i][j]))
    for j in range(6):
        rows.append(boundrow(f'deficit_M{j+1}',deficit[j],0,d['demand'][j]))
        rows.append(boundrow(f'demand_M{j+1}',flow[:,j].sum()+deficit[j],d['demand'][j],d['demand'][j]))
    # Полная расширенная сеть. S — источник, D — фиктивный поставщик, T — сток.
    edges=[]
    for i in range(4): edges.append(dict(source='S',target=f'W{i+1}',flow=flow[i].sum(),capacity=d['supply'][i],cost=0))
    edges.append(dict(source='S',target='D',flow=deficit.sum(),capacity=sum(d['demand']),cost=0))
    for i in range(4):
        for j in range(6): edges.append(dict(source=f'W{i+1}',target=f'M{j+1}',flow=flow[i,j],capacity=d['capacity'][i][j],cost=d['cost'][i][j]))
    for j in range(6):
        edges.append(dict(source='D',target=f'M{j+1}',flow=deficit[j],capacity=d['demand'][j],cost=d['penalty'][j]))
        edges.append(dict(source=f'M{j+1}',target='T',flow=d['demand'][j],capacity=d['demand'][j],cost=0))
    balances=[]
    for node in ['S','D']+d['warehouses']+d['shops']+['T']:
        incoming=sum(e['flow'] for e in edges if e['target']==node); outgoing=sum(e['flow'] for e in edges if e['source']==node)
        expected=sum(d['demand']) if node=='S' else -sum(d['demand']) if node=='T' else 0
        balances.append(dict(node=node,inflow=incoming,outflow=outgoing,out_minus_in=outgoing-incoming,expected=expected,residual=outgoing-incoming-expected))
        rows.append(boundrow('balance_'+node,outgoing-incoming,expected,expected))
    for e in edges: rows.append(boundrow('arc_'+e['source']+'_'+e['target'],e['flow'],0,e['capacity']))
    transport=float(np.sum(flow*np.array(d['cost']))); penalty=float(deficit@np.array(d['penalty']))
    csvwrite(OUT/(name+'_network_arcs.csv'),edges); csvwrite(OUT/(name+'_node_balances.csv'),balances)
    return checks(name,rows,transport+penalty,objective,dict(transport_cost=transport,penalty_cost=penalty,total_deficit=float(deficit.sum()),deficit=deficit.tolist()))

def greedy(d):
    start=time.perf_counter(); x=np.zeros((4,6)); supply=np.array(d['supply'],float); rem=np.array(d['demand'],float)
    for i,j in sorted(np.ndindex(4,6),key=lambda ij:(d['cost'][ij[0]][ij[1]],ij)):
        if d['cost'][i][j]>=d['penalty'][j]: continue
        v=min(supply[i],rem[j],d['capacity'][i][j]); x[i,j]=v; supply[i]-=v; rem[j]-=v
    values=np.r_[x.ravel(),rem]
    obj=float(np.sum(x*np.array(d['cost']))+rem@d['penalty'])
    return values,obj,time.perf_counter()-start

def draw_network(d,result):
    import networkx as nx
    G=nx.DiGraph(); pos={}
    for i in range(4): pos[f'W{i+1}']=(0,3.5-i)
    pos['D']=(0,-.7)
    for j in range(6): pos[f'M{j+1}']=(4,4-j)
    pos['S']=(-1.8,1.6); pos['T']=(5.8,1.6)
    flow=np.array(result['x'][:24]).reshape(4,6); short=np.array(result['x'][24:])
    labels={}; solid=[]; empty=[]
    for i in range(4):
        for j in range(6):
            u=f'W{i+1}'; v=f'M{j+1}'; G.add_edge(u,v)
            if flow[i,j]>TOL: solid.append((u,v))
            else: empty.append((u,v))
            # Для каждой дуги показаны и нулевые значения.
            labels[(u,v)]=f'{flow[i,j]:g}/{d["capacity"][i][j]}'
    for j in range(6): G.add_edge('D',f'M{j+1}'); labels[('D',f'M{j+1}')]=f'{short[j]:g}/{d["demand"][j]}'
    for i in range(4):
        e=('S',f'W{i+1}'); G.add_edge(*e); solid.append(e)
        labels[e]=f'{flow[i].sum():g}/{d["supply"][i]}'
    G.add_edge('S','D'); solid.append(('S','D')); labels[('S','D')]=f'{short.sum():g}/{sum(d["demand"])}'
    for j in range(6):
        e=(f'M{j+1}','T'); G.add_edge(*e); solid.append(e)
        labels[e]=f'{d["demand"][j]}/{d["demand"][j]}'
    fig,ax=plt.subplots(figsize=(12,8))
    nx.draw_networkx_nodes(G,pos,node_color=['#e0ecf4' if n.startswith('W') else '#fee8c8' if n=='D' else '#eeeeee' if n in ['S','T'] else '#e5f5e0' for n in G],node_size=1000,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=10,ax=ax)
    nx.draw_networkx_edges(G,pos,edgelist=empty,edge_color='#cccccc',style='dotted',width=.6,arrowsize=12,ax=ax)
    nx.draw_networkx_edges(G,pos,edgelist=solid,edge_color='#377eb8',width=1.8,arrowsize=15,ax=ax)
    nx.draw_networkx_edges(G,pos,edgelist=[('D',f'M{j+1}') for j in range(6)],edge_color='#d95f02',style='dashed',arrowsize=15,ax=ax)
    for source,labelpos in zip(['W1','W2','W3','W4','D','S','sink'],[.16,.35,.55,.75,.9,.5,.5]):
        group={e:label for e,label in labels.items() if (e[1]=='T' if source=='sink' else e[0]==source and e[1]!='T')}
        nx.draw_networkx_edge_labels(G,pos,group,label_pos=labelpos,font_size=9,rotate=False,ax=ax,bbox=dict(alpha=.95,color='white',pad=.2))
    ax.text(2,-1.5,'Подпись дуги: поток / пропускная способность, палет/сут\nS — источник; T — сток; D — дефицит со штрафом',ha='center')
    ax.set_xlim(-2.4,6.4); ax.axis('off'); savefig('network.png')

def main():
    environment(); d=load_data()
    base=solve(*build_model(d),'base'); flow=np.array(base['x'][:24]).reshape(4,6)
    busiest=tuple(int(i) for i in np.unravel_index(np.argmax(flow),flow.shape))
    scenarios={k:copy.deepcopy(d) for k in ['base','deficit','arc_overload','arc_failure','stress']}
    for name in ['deficit','stress']: scenarios[name]['demand']=(np.array(d['demand'])*1.2).tolist()
    i,j=busiest; scenarios['arc_overload']['capacity'][i][j]*=.8
    for name in ['arc_failure','stress']: scenarios[name]['capacity'][i][j]=0
    summary=[]; allresults={}
    for name,sd in scenarios.items():
        dump(OUT/(name+'_data.json'),sd)
        r=base if name=='base' else solve(*build_model(sd),name)
        v=verify(sd,r['x'],name,r['objective']); gx,go,gt=greedy(sd); verify(sd,gx,name+'_greedy',go)
        csvwrite(OUT/(name+'_solution.csv'),[dict(warehouse=f'W{i+1}',shop=f'M{j+1}',flow=r['x'][6*i+j],capacity=sd['capacity'][i][j],cost=sd['cost'][i][j]) for i in range(4) for j in range(6)])
        csvwrite(OUT/(name+'_deficit.csv'),[dict(shop=f'M{j+1}',demand=sd['demand'][j],unserved=r['x'][24+j],penalty=sd['penalty'][j]) for j in range(6)])
        csvwrite(OUT/(name+'_greedy_solution.csv'),[dict(variable=k,value=z) for k,z in enumerate(gx)])
        row=dict(scenario=name,status=r['status'],cost=r['objective'],transport_cost=v['transport_cost'],penalty_cost=v['penalty_cost'],deficit=v['total_deficit'],damage=r['objective']-base['objective'],greedy_cost=go,greedy_gap_percent=100*(go/r['objective']-1),seconds=r['seconds'],greedy_seconds=gt)
        summary.append(row); allresults[name]=dict(result=r,verification=v,data=sd,summary=row)
    saturated=[]
    for i in range(4):
        for j in range(6):
            if abs(flow[i,j]-d['capacity'][i][j])<=TOL:
                marginal=base['reduced_cost'][6*i+j]
                sd=copy.deepcopy(d); sd['capacity'][i][j]+=.01
                rr=solve(*build_model(sd),f'arc_price_{i+1}_{j+1}')
                verify(sd,rr['x'],f'arc_price_{i+1}_{j+1}',rr['objective'])
                dump(OUT/f'arc_price_{i+1}_{j+1}_data.json',sd)
                saturated.append(dict(warehouse=f'W{i+1}',shop=f'M{j+1}',flow=flow[i,j],capacity=d['capacity'][i][j],marginal=marginal,predicted_change=.01*marginal,actual_change=rr['objective']-base['objective']))
    csvwrite(OUT/'scenarios.csv',summary); csvwrite(OUT/'bottlenecks.csv',saturated)
    csvwrite(OUT/'node_prices.csv',[dict(node=(d['warehouses']+d['shops'])[i],marginal=v) for i,v in enumerate(base['row_dual'])])
    dump(OUT/'summary.json',dict(busiest=busiest,scenarios=allresults,bottlenecks=saturated))
    draw_network(d,base)
    fig,ax=plt.subplots(figsize=(9,4)); pos=np.arange(len(summary))
    ax.bar(pos,[r['transport_cost'] for r in summary],label='Перевозка'); ax.bar(pos,[r['penalty_cost'] for r in summary],bottom=[r['transport_cost'] for r in summary],label='Штраф')
    ax.set_xticks(pos,[r['scenario'] for r in summary]); ax.set_ylabel('руб./сут'); ax.legend(); savefig('scenario_costs.png')
    print('ЛР 2:',[(s['scenario'],s['cost'],s['deficit']) for s in summary])

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate-data',action='store_true',help='Восстановить исходный data.json по seed 10128')
    args=parser.parse_args()
    if args.regenerate_data: dump(ROOT/'data.json',generate_data())
    main()
    from reporting import build_report
    build_report()
