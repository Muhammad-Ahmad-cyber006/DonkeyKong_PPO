import os
import time

import numpy as np

from dk_env import DonkeyKongEnv,N_ACTIONS,N_ROUNDS,OBS_DIM,nav_bot

MODEL_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')

# small fully connected net with tanh hidden layers and hand-written backprop
class MLP:
    def __init__(self,sizes,rng,out_gain):
        self.W,self.b=[],[]
        for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
            g=out_gain if i==len(sizes)-2 else np.sqrt(2)
            q,_=np.linalg.qr(rng.standard_normal((max(a,b),min(a,b))))
            w=(q if a>=b else q.T)[:a,:b]*g  # orthogonal init
            self.W.append(w.astype(np.float32))
            self.b.append(np.zeros(b,np.float32))

    def forward(self,x):
        self.acts=[x]
        for i,(w,b) in enumerate(zip(self.W,self.b)):
            x=x@w+b
            if i<len(self.W)-1:
                x=np.tanh(x)
            self.acts.append(x)
        return x

    def backward(self,d):
        gW,gb=[],[]
        for i in reversed(range(len(self.W))):
            gW.append(self.acts[i].T@d)
            gb.append(d.sum(0))
            if i:
                d=(d@self.W[i].T)*(1-self.acts[i]**2)
        return gW[::-1],gb[::-1]

# policy net (action probabilities) + value net (expected return) sharing one save file
class Agent:
    def __init__(self,hidden=128,seed=0):
        rng=np.random.default_rng(seed)
        self.pi=MLP([OBS_DIM,hidden,hidden,N_ACTIONS],rng,0.01)
        self.v=MLP([OBS_DIM,hidden,hidden,1],rng,1.0)
        self.hidden=hidden

    def params(self):
        return self.pi.W+self.pi.b+self.v.W+self.v.b

    def probs(self,obs):
        z=self.pi.forward(obs)
        z=z-z.max(1,keepdims=True)
        e=np.exp(z)
        return e/e.sum(1,keepdims=True)

    def act(self,obs,rng,greedy=False):
        p=self.probs(obs[None] if obs.ndim==1 else obs)
        if greedy:
            return p.argmax(1)
        c=p.cumsum(1)
        return (c<rng.random((len(p),1))).sum(1).clip(0,N_ACTIONS-1)

    def save(self,path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        np.savez(path,hidden=self.hidden,**{f'p{i}':p for i,p in enumerate(self.params())})

    @staticmethod
    def load(path):
        d=np.load(path)
        a=Agent(int(d['hidden']))
        for i,p in enumerate(a.params()):
            p[...]=d[f'p{i}']
        return a

# Adam optimiser with global gradient-norm clipping
class Adam:
    def __init__(self,params,lr):
        self.p,self.lr,self.t=params,lr,0
        self.m=[np.zeros_like(p) for p in params]
        self.v=[np.zeros_like(p) for p in params]

    def step(self,grads,max_norm=0.5):
        n=np.sqrt(sum((g**2).sum() for g in grads))
        if n>max_norm:
            grads=[g*max_norm/(n+1e-8) for g in grads]
        self.t+=1
        for p,g,m,v in zip(self.p,grads,self.m,self.v):
            m*=0.9
            m+=0.1*g
            v*=0.999
            v+=0.001*g*g
            p-=self.lr*(m/(1-0.9**self.t))/(np.sqrt(v/(1-0.999**self.t))+1e-8)

# one PPO update on a collected batch: clipped policy loss + value loss + entropy bonus.
# stops early when the policy moved too far (kl > 0.03). returns the last kl.
def ppo_update(agent,opt,B,clip=0.2,ent_coef=0.01,epochs=4,mb=512,rng=None):
    n=len(B['obs'])
    kl=0.0
    for _ in range(epochs):
        idx=rng.permutation(n)
        for st in range(0,n,mb):
            j=idx[st:st+mb]
            o,a,A=B['obs'][j],B['act'][j],B['adv'][j]
            A=(A-A.mean())/(A.std()+1e-8)
            p=agent.probs(o)
            lp=np.log(p+1e-10)
            ratio=np.exp(lp[np.arange(len(j)),a]-B['logp'][j])
            active=np.where(A>0,ratio<1+clip,ratio>1-clip)
            oh=np.eye(N_ACTIONS,dtype=np.float32)[a]
            dz=(-(A*ratio*active))[:,None]*(oh-p)  # policy-gradient term
            H=-(p*lp).sum(1,keepdims=True)
            dz+=ent_coef*p*(lp+H)  # entropy bonus
            gp=agent.pi.backward(dz/len(j))
            v=agent.v.forward(o)[:,0]
            gv=agent.v.backward(((v-B['ret'][j])/len(j))[:,None])
            opt.step(gp[0]+gp[1]+gv[0]+gv[1])
            kl=float(((ratio-1)-np.log(ratio)).mean())
        if kl>0.03:
            break
    return kl

# ---------------------------------------------------------------- evaluation
def play_game(policy,seed,start_round=0,render=False,env=None):
    # play one full game (up to 10 rounds, 5 lives) and return a stats dict
    env=env or DonkeyKongEnv(seed=seed)
    env.rng.seed(seed)
    obs=env.reset(start_round)
    att,clr,deaths=np.zeros(N_ROUNDS),np.zeros(N_ROUNDS),{}
    done,info=False,{}
    while not done:
        att_round=env.round
        obs,_,done,info=env.step(policy(obs,env))
        if 'victory' in info:
            clr[att_round]+=1
        if 'death' in info:
            deaths[info['death']]=deaths.get(info['death'],0)+1
    return dict(score=env.score,cleared=env.cleared,final_round=env.round+1,lives=env.lives,
                clr=clr,deaths=deaths)

# average stats over many full games with fixed seeds (same layouts/barrels for every policy)
def evaluate(policy,episodes=20,seed0=10_000):
    rs=[play_game(policy,seed0+i) for i in range(episodes)]
    sc=np.array([r['score'] for r in rs])
    cl=np.array([r['cleared'] for r in rs])
    per_round=np.sum([r['clr'] for r in rs],0)/episodes  # P(round k cleared)
    d={}
    for r in rs:
        for k,v in r['deaths'].items():
            d[k]=d.get(k,0)+v
    return dict(score_mean=sc.mean(),score_max=int(sc.max()),score_std=sc.std(),cleared_mean=cl.mean(),
                full_runs=float((cl==N_ROUNDS).mean()),per_round=per_round,deaths=d)

def agent_policy(agent,greedy=True,seed=0):
    rng=np.random.default_rng(seed)
    return lambda obs,env:int(agent.act(obs,rng,greedy)[0])

def random_policy(seed=0):
    rng=np.random.default_rng(seed)
    return lambda obs,env:int(rng.integers(N_ACTIONS))

def nav_policy():
    return lambda obs,env:nav_bot(env)  # scripted: follows the path hint, never dodges

def fmt(e):
    pr=' '.join(f'{x:.2f}' for x in e['per_round'])
    return (f"score {e['score_mean']:.0f} (max {e['score_max']}, sd {e['score_std']:.0f}) | rounds cleared "
            f"{e['cleared_mean']:.2f}/10 | full 10-round runs {e['full_runs']:.0%}\n"
            f"   P(clear round 1..10): {pr}\n   deaths: {e['deaths']}")

# ------------------------------------------------------------------ training
def train(total_steps=3_000_000,n_envs=8,n_steps=256,lr=3e-4,gamma=0.995,lam=0.95,seed=0,
          out='ppo_dk',eval_every=25,resume=None,log=print):
    # collect n_steps from each of n_envs games, compute GAE advantages, run a PPO update, repeat.
    # saves the best-evaluated weights to <out>_best.npz and the latest to <out>_last.npz.
    rng=np.random.default_rng(seed)
    agent=Agent.load(resume) if resume else Agent(seed=seed)
    opt=Adam(agent.params(),lr)
    envs=[DonkeyKongEnv(seed=seed*1000+i) for i in range(n_envs)]
    start=lambda:0 if rng.random()<0.5 else int(rng.integers(N_ROUNDS))  # mix of full runs + every difficulty
    obs=np.stack([e.reset(start()) for e in envs])
    os.makedirs(MODEL_DIR,exist_ok=True)
    n_up=total_steps//(n_envs*n_steps)
    best,t0,eps=-1,time.time(),[]
    for up in range(1,n_up+1):
        opt.lr=lr*(1-(up-1)/n_up)  # linear learning-rate decay
        ent=0.01*(1-(up-1)/n_up)+0.002  # entropy bonus fades out
        T=n_steps
        O=np.zeros((T,n_envs,OBS_DIM),np.float32)
        A=np.zeros((T,n_envs),np.int64)
        LP=np.zeros((T,n_envs),np.float32)
        R=np.zeros((T,n_envs),np.float32)
        D=np.zeros((T,n_envs),np.float32)
        V=np.zeros((T+1,n_envs),np.float32)
        for t in range(T):
            p=agent.probs(obs)
            a=(p.cumsum(1)<rng.random((n_envs,1))).sum(1).clip(0,N_ACTIONS-1)
            O[t],A[t]=obs,a
            LP[t]=np.log(p[np.arange(n_envs),a]+1e-10)
            V[t]=agent.v.forward(obs)[:,0]
            for i,e in enumerate(envs):
                o,r,d,info=e.step(int(a[i]))
                R[t,i],D[t,i]=r,d
                if d:
                    eps.append((e.score,e.cleared))
                    o=e.reset(start())
                obs[i]=o
        V[T]=agent.v.forward(obs)[:,0]
        # GAE: walk backwards, blending 1-step errors; nt cuts the chain at episode ends
        adv=np.zeros((T,n_envs),np.float32)
        g=0
        for t in reversed(range(T)):
            nt=1-D[t]
            delta=R[t]+gamma*V[t+1]*nt-V[t]
            g=delta+gamma*lam*nt*g
            adv[t]=g
        B=dict(obs=O.reshape(-1,OBS_DIM),act=A.reshape(-1),logp=LP.reshape(-1),
               adv=adv.reshape(-1),ret=(adv+V[:-1]).reshape(-1))
        kl=ppo_update(agent,opt,B,ent_coef=ent,rng=rng)
        steps=up*n_envs*n_steps
        tr_score=np.mean([x[0] for x in eps[-30:]]) if eps else 0  # recent training episodes
        tr_clear=np.mean([x[1] for x in eps[-30:]]) if eps else 0
        if up%eval_every==0 or up==n_up:
            ev=evaluate(agent_policy(agent),episodes=10)
            if ev['score_mean']>best:
                best=ev['score_mean']
                agent.save(os.path.join(MODEL_DIR,f'{out}_best.npz'))
            log(f"[{time.time()-t0:5.0f}s] upd {up}/{n_up} steps {steps} | train ep score {tr_score:.0f} "
                f"cleared {tr_clear:.2f} | EVAL 1->10: score {ev['score_mean']:.0f} cleared {ev['cleared_mean']:.2f} "
                f"(best {best:.0f}) | kl {kl:.4f}")
        agent.save(os.path.join(MODEL_DIR,f'{out}_last.npz'))
    return agent

if __name__=='__main__':  # bugfix: the file had no runnable entry point
    train()