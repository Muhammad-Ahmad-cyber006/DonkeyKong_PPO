# Donkey Kong PPO agent. No human input: the AI plays everything.
#
#   python main.py train [--steps 4000000]     train with PPO (saves models/ppo_dk_best.npz)
#   python main.py test  [--episodes 30]       evaluate trained agent vs baselines over 10 rounds
#   python main.py watch [--games 1]           watch the trained agent play in real time
DESC='Donkey Kong PPO agent: train, test or watch.'
import argparse
import os

import ppo
from dk_env import DonkeyKongEnv,N_ROUNDS

MODEL=os.path.join(ppo.MODEL_DIR,'ppo_dk_best.npz')

# load a trained agent or exit with a hint if the file is missing
def load(path):
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found. Run: python main.py train')
    return ppo.Agent.load(path)

def main():
    ap=argparse.ArgumentParser(description=DESC)
    ap.add_argument('mode',choices=['train','test','watch'])
    ap.add_argument('--steps',type=int,default=4_000_000)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--episodes',type=int,default=30)
    ap.add_argument('--model',default=MODEL)
    ap.add_argument('--resume',default=None,help='continue training from a .npz')
    ap.add_argument('--stochastic',action='store_true',help='sample actions instead of argmax')
    ap.add_argument('--games',type=int,default=1)
    ap.add_argument('--fps',type=int,default=60,help='watch speed (0 = as fast as possible)')
    a=ap.parse_args()

    if a.mode=='train':
        ppo.train(a.steps,seed=a.seed,resume=a.resume)
        print('saved',MODEL)
    elif a.mode=='test':
        agent=load(a.model)
        print(f'Evaluating {a.episodes} full games (rounds 1->10, 5 lives, fresh seeds)\n')
        for name,pol in (('random',ppo.random_policy()),('scripted path-follower (no dodging)',ppo.nav_policy()),
                         ('PPO agent',ppo.agent_policy(agent,not a.stochastic))):
            print(f'== {name}\n   '+ppo.fmt(ppo.evaluate(pol,a.episodes))+'\n')
    else:
        agent=load(a.model)
        pol=ppo.agent_policy(agent,not a.stochastic)
        env=DonkeyKongEnv(seed=int.from_bytes(os.urandom(2),'little'),window=True)
        try:
            for g in range(a.games):
                obs,done,info=env.reset(),False,{}
                while not done:
                    act=pol(obs,env)
                    env.fps=a.fps
                    obs,_,done,info=env.step(act)
                print(f"game {g+1}: score {info['score']}  rounds cleared {info['cleared']}/{N_ROUNDS}")
                for _ in range(120):
                    env.render(None,a.fps or 60)
        except KeyboardInterrupt:
            pass
        env.close()

if __name__=='__main__':
    main()