import random
import heapq
import os
from bisect import bisect

import numpy as np
import pygame
from pygame import Rect

HERE=os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------------- geometry
W,H=1120,930
SW,SH=W//32,H//32  # section width / height
SLOPE=SH//8
start_y=H-2*SH
row2_y=start_y-4*SH
row3_y=row2_y-7*SLOPE-3*SH
row4_y=row3_y-4*SH
row5_y=row4_y-7*SLOPE-3*SH
row6_y=row5_y-4*SH
BRIDGES=[(1,start_y,15),(16,start_y-SLOPE,3),(19,start_y-2*SLOPE,3),
         (22,start_y-3*SLOPE,3),(25,start_y-4*SLOPE,3),(28,start_y-5*SLOPE,3),
         (25,row2_y,3),(22,row2_y-SLOPE,3),(19,row2_y-2*SLOPE,3),
         (16,row2_y-3*SLOPE,3),(13,row2_y-4*SLOPE,3),(10,row2_y-5*SLOPE,3),
         (7,row2_y-6*SLOPE,3),(4,row2_y-7*SLOPE,3),(2,row2_y-8*SLOPE,2),
         (4,row3_y,3),(7,row3_y-SLOPE,3),(10,row3_y-2*SLOPE,3),
         (13,row3_y-3*SLOPE,3),(16,row3_y-4*SLOPE,3),(19,row3_y-5*SLOPE,3),
         (22,row3_y-6*SLOPE,3),(25,row3_y-7*SLOPE,3),(28,row3_y-8*SLOPE,2),
         (25,row4_y,3),(22,row4_y-SLOPE,3),(19,row4_y-2*SLOPE,3),
         (16,row4_y-3*SLOPE,3),(13,row4_y-4*SLOPE,3),(10,row4_y-5*SLOPE,3),
         (7,row4_y-6*SLOPE,3),(4,row4_y-7*SLOPE,3),(2,row4_y-8*SLOPE,2),
         (4,row5_y,3),(7,row5_y-SLOPE,3),(10,row5_y-2*SLOPE,3),
         (13,row5_y-3*SLOPE,3),(16,row5_y-4*SLOPE,3),(19,row5_y-5*SLOPE,3),
         (22,row5_y-6*SLOPE,3),(25,row5_y-7*SLOPE,3),(28,row5_y-8*SLOPE,2),
         (25,row6_y,3),(22,row6_y-SLOPE,3),(19,row6_y-2*SLOPE,3),
         (16,row6_y-3*SLOPE,3),(2,row6_y-4*SLOPE,14),(13,row6_y-4*SH,6),
         (10,row6_y-3*SH,3)]
LADDERS=[(12,row2_y+6*SLOPE,2),(12,row2_y+26*SLOPE,2),(25,row2_y+11*SLOPE,4),
         (6,row3_y+11*SLOPE,3),(14,row3_y+8*SLOPE,4),(10,row4_y+6*SLOPE,1),
         (10,row4_y+24*SLOPE,2),(16,row4_y+6*SLOPE,5),(25,row4_y+9*SLOPE,4),
         (6,row5_y+11*SLOPE,3),(11,row5_y+8*SLOPE,4),(23,row5_y+4*SLOPE,1),
         (23,row5_y+24*SLOPE,2),(25,row6_y+9*SLOPE,4),(13,row6_y+5*SLOPE,2),
         (13,row6_y+25*SLOPE,2),(18,row6_y-27*SLOPE,4),(12,row6_y-17*SLOPE,2),
         (10,row6_y-17*SLOPE,2),(12,-5,13),(10,-5,13)]
HAMMERS=[(4,row6_y-4*SLOPE+SH),(4,row4_y-8*SLOPE+SH)]  # (col, top y)
TARGET=Rect(13*SW,row6_y-4*SH,3*SW,1)
# platforms: 2px-high top surface per bridge piece, bucketed per 35px column
PX0=[b[0]*SW for b in BRIDGES]
PLEN=[b[2] for b in BRIDGES]
PY=[b[1] for b in BRIDGES]
_ROW_BREAKS=[(row6_y+row5_y-8*SLOPE)//2,(row5_y+row4_y-8*SLOPE)//2,
             (row4_y+row3_y-8*SLOPE)//2,(row3_y+row2_y-8*SLOPE)//2,
             (row2_y+start_y-5*SLOPE)//2]

def row_of(y):
    # girder row of a y coordinate: 1 = bottom girder ... 6 = top girder
    return 6-bisect(_ROW_BREAKS,y)

# girders slope down to the left on rows 1,3,5 and to the right on rows 2,4,6
PDIR=[1 if row_of(y)%2==0 else -1 for y in PY]
COLS=[[] for _ in range(32)]
for _i,_b in enumerate(BRIDGES):
    for _c in range(_b[0],min(_b[0]+_b[2],32)):
        COLS[_c].append(_i)
CLIMB=[l for l in LADDERS if l[2]>=3]  # shorter ones are decoration
P_OFF=[sum(PLEN[:i]) for i in range(len(BRIDGES))]  # first nav-graph node of each girder
OIL=Rect(4*SW,int(H-4.5*SH),2*SW,int(2.5*SH))
MARIO_W,MARIO_H=2*SW,int(2.5*SH)
SPAWN=(250,H-130)
FALL_LIMIT=150  # px; longer fall is fatal
HAMMER_TIME=450  # frames the hammer lasts
JUMP_V=5.5  # jump launch speed (px/frame); gravity 0.25 -> ~60px high

def ground(x0,x1,y0,y1):
    # index of the highest platform whose 2px top strip overlaps the box, else -1
    best,by=-1,1<<30
    for c in range(max(x0//SW,0),min((x1-1)//SW,31)+1):
        for i in COLS[c]:
            y=PY[i]
            if y0<y+2 and y<y1 and y<by:
                best,by=i,y
    return best

# ------------------------------------------- navigation graph (reward shaping)
class Layout:
    # one ladder configuration: which ladders exist, plus the path-to-goal data built for it
    def __init__(self,removed,bodies,chains,dist,hint,ok):
        self.removed=self.bodies=self.chains=removed,bodies,chains
        self.dist,self.hint,self.ok=dist,hint,ok
        self.d0=max(d for d in dist if d<1e8)  # longest finite distance (normaliser)
        self.cost=dist[P_OFF[0]+(SPAWN[0]-PX0[0])//SW]  # path length from spawn to goal

def _build_nav(removed=()):
    # build the graph of walkable spots (girder cells + ladder rungs) without the ladders in
    # removed (indices into CLIMB), then run Dijkstra from the goal to get distance + a move hint
    keep=[i for i in range(len(CLIMB)) if i not in removed]
    bodies=[Rect(x*SW,y-SH,SW,int(0.6*n*SH+SH)) for x,y,n in (CLIMB[i] for i in keep)]
    xy=[]
    for i in range(len(BRIDGES)):  # one node per girder cell
        xy+=[(PX0[i]+c*SW+SW/2,PY[i]) for c in range(PLEN[i])]
    adj=[[] for _ in range(len(xy)+4000)]

    def link(u,v,w):
        adj[u].append((v,w))
        adj[v].append((u,w))

    off=P_OFF
    for i in range(len(BRIDGES)):
        for c in range(PLEN[i]-1):  # walk along a girder
            link(off[i]+c,off[i]+c+1,SW/3)
        for j in range(len(BRIDGES)):  # girders that touch end to start
            if i!=j and PX0[i]+PLEN[i]*SW==PX0[j] and abs(PY[i]-PY[j])<=2*SH:
                link(off[i]+PLEN[i]-1,off[j],SW/3+(20 if abs(PY[i]-PY[j])>=12 else 0))
    chains={}  # ladder idx -> (first node, y_top, rungs)
    for j,body in enumerate(bodies):
        lx=body.x+SW//2
        cand=[i for i in range(len(BRIDGES)) if PX0[i]<=lx<PX0[i]+PLEN[i]*SW
              and body.top-10<=PY[i]<=body.bottom+SH]
        if len(cand)<2:
            continue
        yt,yb=min(PY[i] for i in cand),max(PY[i] for i in cand)
        n=round((yb-yt)/6)+1
        first=len(xy)
        xy+=[(lx,yt+6*k) for k in range(n)]
        for k in range(n-1):  # climb along the rungs
            link(first+k,first+k+1,3)
        for i in cand:  # join rungs to the girders they touch
            link(off[i]+(lx-PX0[i])//SW,first+round((PY[i]-yt)/6),1)
        chains[j]=(first,yt,n)
    adj=adj[:len(xy)]
    # distance (in frames) to goal: multi-source Dijkstra from target girder columns
    goal=[i for i in range(len(BRIDGES)) if PY[i]==TARGET.y and PX0[i]==TARGET.x][0]
    dist=[1e9]*len(xy)
    pq=[]
    for c in range(3):
        dist[off[goal]+c]=0
        pq.append((0,off[goal]+c))
    heapq.heapify(pq)
    while pq:
        d,u=heapq.heappop(pq)
        if d>dist[u]:
            continue
        for v,w in adj[u]:
            if d+w<dist[v]:
                dist[v]=d+w
                heapq.heappush(pq,(d+w,v))
    ok=dist[off[0]+(SPAWN[0]-PX0[0])//SW]<1e8  # can Mario still reach the goal from spawn?
    fin=max(d for d in dist if d<1e8)
    dist=[min(d,fin*1.2) for d in dist]  # unreachable spots get a large finite cost
    is_chain=[False]*len(xy)
    for f,_,n in chains.values():
        for k in range(n):
            is_chain[f+k]=True

    def best(u):  # neighbour closest to the goal
        v=min(adj[u],key=lambda e:dist[e[0]],default=(u,0))[0]
        return v if dist[v]<dist[u] else u

    sg=lambda a:(a>1)-(a<-1)
    sgy=lambda a:(a>4)-(a<-4)  # girder slope steps (3px) are not "up"
    hint=[]
    for u in range(len(xy)):  # (dx, dy, jump) toward goal
        v=best(u)
        dx,dy=sg(xy[v][0]-xy[u][0]),sgy(xy[v][1]-xy[u][1])
        if dy==0 and is_chain[v]:
            w=best(v)
            dy=sgy(xy[w][1]-xy[v][1])
        jump=int(not is_chain[u] and not is_chain[v] and xy[u][1]-xy[v][1]>=12)
        hint.append((dx,dy,jump) if v!=u else (0,0,0))
    return Layout(tuple(removed),bodies,chains,dist,hint,ok)

FULL=_build_nav()  # the untouched level
# every single-ladder removal that stays solvable but forces a longer route
LAYOUTS=[lay for lay in (_build_nav((i,)) for i in range(len(CLIMB))) if lay.ok and lay.cost>FULL.cost+1]

ACTIONS=[(0,0,0,0),(-1,0,0,0),(1,0,0,0),(0,0,1,0),(0,0,0,1),
         (0,1,0,0),(-1,1,0,0),(1,1,0,0)]  # dx, jump, up, down
ACTION_NAMES=['noop','left','right','up','down','jump','jump-left','jump-right']
N_ACTIONS=len(ACTIONS)
N_ROUNDS=10
OBS_DIM=59

def round_params(r):
    # difficulty for round r (0..9); every number gets harder with r
    return dict(spawn=max(200,340-15*r),  # frames between barrels
                b_speed=3+r//5,  # barrel speed px/frame
                b_ladder=0.10+0.02*r,  # chance a barrel takes a ladder
                f_speed=2+(r>=6),  # fireball speed
                f_cap=2+r//4,  # max fireballs alive
                f_ladder=0.20+0.02*r,  # chance a fireball climbs
                tick=max(40,60-2*r),  # frames per -100 bonus
                hammers=1 if r<3 else 0)  # hammers only help early on

def pick_layout(rng,r):
    # random ladder layout: one ladder on the shortest path is gone, so the route changes.
    # 10% of attempts keep the full level so the main route is not forgotten.
    return FULL if rng.random()<0.1 or not LAYOUTS else rng.choice(LAYOUTS)

# ----------------------------------------------------------------- entities
class Barrel:
    def __init__(self,speed=3):
        self.r=Rect(0,0,50,50)
        self.r.center=(270,270)
        self.yc,self.dir,self.falling,self.fy0=0,1,False,0
        self.check_lad=self.oil=False
        self.pos=self.cnt=0
        self.vx=0
        self.speed=speed  # px/frame while rolling

    @property
    def hit(self):
        # deadly core of the barrel: smaller than the sprite, so skimming over the top is safe
        return self.r.inflate(-16,-22)

class Flame:
    def __init__(self,speed):
        self.r=Rect(0,0,int(1.5*SW),2*SH)
        self.r.center=(5*SW,H-4*SH)
        self.pos,self.cnt,self.x_count,self.x_max=1,0,0,4
        self.x_change,self.y_change=speed,0
        self.climbing=self.check_lad=False
        self.cy0=0

class Mario:
    def __init__(self):
        self.r=Rect(0,0,MARIO_W,MARIO_H)
        self.r.center=SPAWN
        self.yc,self.xc,self.dir=0.0,0,1
        self.landed=self.climbing=self.over_barrel=False
        self.hammer=0  # frames of hammer left
        self.hpos=1
        self.peak=self.r.bottom
        self.anim=0
        self.hit=self.hbox=Rect(0,0,0,0)
        self.calc_hitbox()

    def calc_hitbox(self):
        r=self.r
        if not self.hammer or self.hpos==1:
            self.hit=Rect(r.x+15,r.y+5,r.w-30,r.h-10)
            self.hbox=Rect(self.hit.x,self.hit.y-SH,self.hit.w,SH) if self.hammer else Rect(0,0,0,0)
        elif self.dir==1:
            self.hit=Rect(r.x,r.y+5,r.w-30,r.h-10)
            self.hbox=Rect(self.hit.right,r.y+5,self.hit.w,r.h-10)
        else:
            self.hit=Rect(r.x+40,r.y+5,r.w-30,r.h-10)
            self.hbox=Rect(self.hit.x-self.hit.w,r.y+5,self.hit.w,r.h-10)

# --------------------------------------------------------------- environment
class DonkeyKongEnv:
    # Gym-style: reset() -> obs, step(a) -> obs, reward, done, info.
    def __init__(self,frame_skip=4,lives=5,seed=None,window=False):
        self.frame_skip,self.start_lives=frame_skip,lives
        self.rng=random.Random(seed)
        self.window,self.fps=window,60
        self.screen=None
        self.high_score=0
        self.bgs={}  # cached background per ladder layout
        self._side_img=None  # pre-rotated barrel-stack sprite

    # ---- game flow
    def reset(self,start_round=0):
        self.round,self.lives,self.score,self.cleared=start_round,self.start_lives,0,0
        self.done=False
        self.last_death=None
        self._new_attempt()
        return self._obs()

    def _new_attempt(self):
        p=self.p=round_params(self.round)
        self.L=pick_layout(self.rng,self.round)  # random ladders -> a different route every attempt
        self.m=Mario()
        self.barrels,self.flames=[],[]
        self.hammers=[Rect(c*SW,y,2*SW,2*SH) for c,y in HAMMERS[:p['hammers']]]
        self.bonus,self.t=6000,0
        self.bcount,self.btime=p['spawn']//2,p['spawn']
        self.first_flame=False
        self.node,self.lad=None,-1
        self.phi=None
        self.can,self.down=False,False

    def _die(self,why):
        self.lives-=1
        self.last_death=why
        if self.lives<=0:
            self.done=True
        else:
            self._new_attempt()

    # ---- helpers
    def _check_climb(self):
        m=self.m
        can=down=False
        under=Rect(m.r.x,m.r.y+2*SH,m.r.w,m.r.h)
        for j,body in enumerate(self.L.bodies):
            if not can and m.hit.colliderect(body):
                can,self.lad=True,j
            if under.colliderect(body):
                down=True
        if (not can and (not down or m.yc<0)) or (m.landed and can and m.yc>0 and not down):
            m.climbing=False
        self.can,self.down=can,down

    def _barrel_step(self,b):
        p,r=self.p,b.r
        if not b.falling and b.yc<8:
            b.yc+=2
        g=ground(r.x,r.right,r.bottom-4,r.bottom+3+int(b.yc))  # swept: no tunnelling
        if g>=0 and (not b.falling or r.bottom>b.fy0+20):
            r.bottom,b.yc,b.falling,b.dir=PY[g],0,False,PDIR[g]
        if not b.oil and r.colliderect(OIL):
            b.oil=True
            if self.rng.randint(0,4)==4 and len(self.flames)<p['f_cap']:
                self.flames.append(Flame(p['f_speed']))
        b.vx=0 if b.falling else b.dir*b.speed
        r.move_ip(b.vx,int(b.yc))
        # ladders: roll the dice once per ladder crossed
        below=Rect(r.x,r.y+SH,r.w,SH)
        touching=(not b.falling) and any(below.colliderect(l) for l in self.L.bodies)
        if touching and not b.check_lad and self.rng.random()<p['b_ladder']:
            b.falling,b.yc,b.fy0=True,4,r.bottom
        b.check_lad=touching
        b.cnt+=1
        if b.cnt>=16:
            b.cnt,b.pos=0,(b.pos+(1 if b.vx>0 else -1))%4
        return r.top<=H

    def _flame_step(self,f):
        p,r=self.p,f.r
        if f.y_change<3 and not f.climbing:
            f.y_change+=0.25
        if ground(r.x,r.right,r.y,r.bottom)>=0 and not (f.climbing and r.bottom>f.cy0-40):
            f.climbing,f.y_change=False,-4
        f.cnt+=1
        if f.cnt>=16:
            f.cnt,f.pos=0,-f.pos
            if f.x_count<f.x_max:
                f.x_count+=1
            else:  # rows 1,3,5 drift right, 2,4,6 drift left
                f.x_count=0
                odd=row_of(r.bottom)%2==1
                f.x_max=self.rng.randint(3,6) if (f.x_change>0)==odd else self.rng.randint(6,10)
                f.x_change*=-1
        r.move_ip(f.x_change,int(f.y_change))
        r.clamp_ip(Rect(0,-10**5,W,2*10**5))
        touching=(not f.climbing) and any(r.colliderect(l) for l in self.L.bodies)
        if touching and not f.check_lad and self.rng.random()<p['f_ladder']:
            f.climbing,f.y_change,f.cy0=True,-4,r.bottom
        f.check_lad=touching
        return 0<=r.top<=H

    # ---- one 60fps physics frame. returns the event that ended it ('barrel','flame','fall','time','victory') or None
    def _frame(self,act):
        dx,jump,up,down=ACTIONS[act]
        p,m=self.p,self.m
        self.t+=1
        if self.t%self.p['tick']==0:
            self.bonus-=100
            if self.bonus<=0:
                return 'time'
        self._check_climb()
        if self.bcount<p['spawn']:
            self.bcount+=1
        else:
            self.bcount=self.rng.randint(0,120)  # random gap until the next barrel
            self.btime=p['spawn']-self.bcount
            self.barrels.append(Barrel(p['b_speed']))
            if not self.first_flame:
                self.first_flame=True
                self.flames.append(Flame(p['f_speed']))
        # --- Mario input (held keys) + physics
        if m.hammer:
            jump=up=down=0  # no jumping/climbing with the hammer
        if up and self.can:
            m.yc,m.climbing=-2,True
        elif down and self.down:
            m.yc,m.climbing=2,True
        elif m.climbing:
            m.yc=0
            if m.landed:
                m.climbing=False
        m.xc=0 if m.climbing else dx
        if dx and not m.climbing:
            m.dir=dx
        if jump and m.landed and not m.climbing:
            m.landed,m.yc=False,-JUMP_V
        g=ground(m.r.left,m.r.right,m.r.bottom-20,m.r.bottom)
        m.landed=g>=0
        fell=False
        if m.landed and not m.climbing:
            m.r.bottom=PY[g]+1
            if m.yc>0:
                m.yc=0
            fell=m.r.bottom-m.peak>FALL_LIMIT
            cx=m.r.centerx
            gc=ground(cx,cx+1,m.r.bottom-20,m.r.bottom+4)  # girder under his centre
            gc=g if gc<0 else gc
            self.node=P_OFF[gc]+min(max((cx-PX0[gc])//SW,0),PLEN[gc]-1)
        if not m.landed and not m.climbing:
            m.yc+=0.25
        m.peak=m.r.bottom if (m.landed or m.climbing) else min(m.peak,m.r.bottom)
        m.r.move_ip(m.xc*3,int(m.yc))
        m.r.clamp_ip(Rect(0,-10**5,W,2*10**5))
        if m.climbing and self.lad in self.L.chains:
            f0,yt,n=self.L.chains[self.lad]
            self.node=f0+min(max(round((m.r.bottom-1-yt)/6),0),n-1)
        if m.xc or (m.climbing and m.yc):
            m.anim+=1
        if m.hammer:
            m.hpos=(m.hammer//30)%2
            m.hammer-=1
        m.calc_hitbox()
        for h in self.hammers[:]:
            if h.colliderect(m.hit):
                self.hammers.remove(h)
                m.hammer=HAMMER_TIME
        # --- enemies
        self.barrels=[b for b in self.barrels if self._barrel_step(b)]
        self.flames=[f for f in self.flames if self._flame_step(f)]
        # --- hammer smashes (before deaths: swinging hammer protects the front)
        if m.hammer:
            for b in self.barrels[:]:
                if b.r.colliderect(m.hbox):
                    self.barrels.remove(b)
                    self.score+=500
            for f in self.flames[:]:
                if f.r.colliderect(m.hbox):
                    self.flames.remove(f)
                    self.score+=500
        # --- jump-over bonus (only when really airborne, not on a ladder)
        under=Rect(m.r.x,m.r.y+2*SH,m.r.w,m.r.h)
        if m.landed or m.climbing:
            m.over_barrel=False
        elif not m.over_barrel:
            for b in self.barrels:
                if under.colliderect(b.r) and not b.hit.colliderect(m.hit):
                    m.over_barrel=True
                    self.score+=100
                    break
        # --- deaths (fresh hitbox, after everything has moved)
        if any(b.hit.colliderect(m.hit) for b in self.barrels):
            return 'barrel'
        if any(f.r.colliderect(m.hit) for f in self.flames):
            return 'flame'
        if fell or m.r.top>H:
            return 'fall'
        if m.landed and m.r.colliderect(TARGET.inflate(0,40)) and \
                Rect(m.r.left,m.r.bottom-20,m.r.w,20).colliderect(TARGET):
            return 'victory'
        return None

    # ---- gym API
    def step(self,a):
        assert not self.done,'call reset()'
        score0,reward,info=self.score,0.0,{}
        for _ in range(self.frame_skip):
            ev=self._frame(a)
            if self.window:
                self.render(a,self.fps)
            if ev=='victory':
                self.score+=max(self.bonus,0)
                self.cleared+=1
                reward+=5.0
                info['victory']=True
                if self.round+1>=N_ROUNDS:
                    self.done=True
                else:
                    self.round+=1
                    self._new_attempt()
                break
            if ev:
                reward-=5.0
                info['death']=ev
                self._die(ev)
                break
        if not self.done and 'death' not in info and 'victory' not in info and self.node is not None:
            phi=self.L.dist[self.node]
            if self.phi is not None:
                reward+=8.0*(self.phi-phi)/self.L.d0  # potential-based progress shaping
            self.phi=phi
        reward+=(self.score-score0)/1000.0-0.002
        self.high_score=max(self.high_score,self.score)
        info.update(score=self.score,round=self.round,lives=self.lives,cleared=self.cleared)
        return self._obs(),reward,self.done,info

    # ---- observation (59 floats)
    def _obs(self):
        m=self.m
        cx,cy=m.r.center
        node=self.node if self.node is not None else 0
        node=min(node,len(self.L.hint)-1)  # bugfix: node can outlive a stale layout
        hdx,hdy,hj=self.L.hint[node]
        o=[cx/W,cy/H,m.yc/8,m.landed,m.climbing,self.can,self.down,m.hammer/HAMMER_TIME,
           m.dir,self.L.dist[node]/self.L.d0,hdx,hdy,hj]
        cl=lambda v:max(-2.0,min(2.0,v))
        for ent,k in ((sorted(self.barrels,key=lambda b:(b.r.centerx-cx)**2+(b.r.centery-cy)**2),5),
                      (sorted(self.flames,key=lambda f:(f.r.centerx-cx)**2+(f.r.centery-cy)**2),3)):
            for i in range(k):
                if i<len(ent):
                    e=ent[i]
                    vx=e.vx/6 if isinstance(e,Barrel) else e.x_change/4
                    fl=e.falling if isinstance(e,Barrel) else e.climbing
                    o+=[cl((e.r.centerx-cx)/250),cl((e.r.centery-cy)/150),vx,float(fl),1.0]
                else:
                    o+=[0.0]*5
        if self.hammers:
            h=min(self.hammers,key=lambda h:(h.centerx-cx)**2+(h.centery-cy)**2)
            o+=[cl((h.centerx-cx)/400),cl((h.centery-cy)/300),1.0]
        else:
            o+=[0.0]*3
        o+=[self.bonus/6000,self.round/9,self.lives/self.start_lives]
        return np.array(o,dtype=np.float32)

    # ------------------------------------------------------------ rendering
    def _init_display(self):
        os.environ.setdefault('SDL_VIDEO_CENTERED','1')
        pygame.init()
        info=pygame.display.Info()
        self.scale=min(1.0,(info.current_h-90)/H,(info.current_w-40)/W)
        self.win=pygame.display.set_mode((int(W*self.scale),int(H*self.scale)))
        pygame.display.set_caption('Donkey Kong - PPO agent')
        self.surf=pygame.Surface((W,H))
        self.clock=pygame.time.Clock()
        self.font,self.font2=(pygame.font.Font('freesansbold.ttf',n) for n in (50,30))
        ld=lambda f,w,h:pygame.transform.scale(
            pygame.image.load(os.path.join(HERE,'assets/images',f)).convert_alpha(),(int(w),int(h)))
        self._side_img=pygame.transform.rotate(ld('barrels/barrel2.png',SW*2,SH*2.5),90)  # rotate once
        self.img=dict(
            barrel=ld('barrels/barrel.png',SW*1.5,SH*2),flames=ld('fire.png',SW*2,SH),
            dk=[ld(f'dk/dk{i}.png',SW*5,SH*5) for i in (1,2,3)],
            peach=[ld(f'peach/peach{i}.png',2*SW,3*SH) for i in (1,2)],
            fire=[ld('fireball.png',1.5*SW,2*SH),ld('fireball2.png',1.5*SW,2*SH)],
            hammer=ld('hammer.png',2*SW,2*SH),
            stand=ld('mario/standing.png',2*SW,2.5*SH),jump=ld('mario/jumping.png',2*SW,2.5*SH),
            run=ld('mario/running.png',2*SW,2.5*SH),
            climb=[ld('mario/climbing1.png',2*SW,2.5*SH),ld('mario/climbing2.png',2*SW,2.5*SH)],
            hjump=ld('mario/hammer_jump.png',2.5*SW,2.5*SH),
            hover=ld('mario/hammer_overhead.png',2.5*SW,3.5*SH))

    def _draw_static(self,surf):
        col=(225,51,129)
        for x,y,n in BRIDGES:
            for i in range(n):
                l=x*SW+SW*i
                m_,r=l+SW*0.5,l+SW
                pygame.draw.line(surf,col,(l,y),(r,y),7)
                pygame.draw.line(surf,col,(l,y+SH),(r,y+SH),7)
                pygame.draw.line(surf,col,(l,y+SH),(m_,y),7)
                pygame.draw.line(surf,col,(m_,y),(r,y+SH),7)
        gone={CLIMB[i] for i in self.L.removed}  # removed ladders are not drawn
        for x,y,n in (l for l in LADDERS if l not in gone):
            for i in range(n):
                t=y+0.6*SH*i
                b=t+0.6*SH
                pygame.draw.line(surf,'light blue',(x*SW,t),(x*SW,b),3)
                pygame.draw.line(surf,'light blue',(x*SW+SW,t),(x*SW+SW,b),3)
                pygame.draw.line(surf,'light blue',(x*SW,t+0.3*SH),(x*SW+SW,t+0.3*SH),3)
        for px,py in ((1.2,5.4),(2.5,5.4),(2.5,7.7),(1.2,7.7)):
            surf.blit(self._side_img,(SW*px,SH*py))
        xc,yc=OIL.x,OIL.y
        pygame.draw.rect(surf,'blue',OIL)
        pygame.draw.rect(surf,'blue',[xc-0.1*SW,yc,2.2*SW,.2*SH])
        pygame.draw.rect(surf,'blue',[xc-0.1*SW,yc+2.3*SH,2.2*SW,.2*SH])
        pygame.draw.rect(surf,'light blue',[xc,yc+.5*SH,2*SW,.2*SH])
        pygame.draw.rect(surf,'light blue',[xc,yc+1.7*SH,2*SW,.2*SH])
        surf.blit(self.font2.render('OIL',True,'light blue'),(xc+.4*SW,yc+.7*SH))

    def render(self,action=None,fps=60):
        if self.screen is None:
            self._init_display()
            self.screen=True
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                raise KeyboardInterrupt
        sf,im,m,p=self.surf,self.img,self.m,self.p
        if self.L.removed not in self.bgs:  # draw each layout's background once
            self.bgs[self.L.removed]=pygame.Surface((W,H))
            self._draw_static(self.bgs[self.L.removed])
        sf.blit(self.bgs[self.L.removed],(0,0))
        flick=(self.t//15)%2
        sf.blit(pygame.transform.flip(im['flames'],bool(flick),False),(OIL.x,OIL.y-SH))
        sf.blit(im['peach'][self.bcount>=self.p['spawn']//2],(10*SW,self.row6_y-6*SH))
        ph=self.btime//4
        left=self.p['spawn']-self.bcount
        kong=im['dk'][1] if left>3*ph else im['dk'][0] if left>2*ph else im['dk'][2] if left>ph \
            else pygame.transform.flip(im['dk'][0],True,False)
        sf.blit(kong,(3.5*SW,self.row6_y-5.5*SH))
        for h in self.hammers:
            sf.blit(im['hammer'],h.topleft)
        for b in self.barrels:
            sf.blit(pygame.transform.rotate(im['barrel'],90*b.pos),b.r.topleft)
        for f in self.flames:
            fi=im['fire'][0 if f.pos==1 else 1]
            sf.blit(fi if f.x_change>0 else pygame.transform.flip(fi,True,False),f.r.topleft)
        if self.m.hammer:
            img=im['hjump'] if self.m.hpos==0 else im['hover']
        elif self.m.climbing:
            img=im['climb'][(self.m.anim//4)%2]
        elif not self.m.landed:
            img=im['jump']
        else:
            img=im['run'] if (self.m.xc and (self.m.anim//4)%2) else im['stand']
        if self.m.dir==-1:
            img=pygame.transform.flip(img,True,False)
        sf.blit(img,(self.m.r.left,self.m.r.top-(SH if self.m.hammer and self.m.hpos==1 else 0)))
        hud=lambda txt,x,y,f=None:sf.blit((f or self.font).render(txt,True,'white'),(x,y))
        hud(f'I•{self.score}',3*SW,2*SH)
        hud(f'TOP•{self.high_score}',14*SW,2*SH)
        hud('[  ][        ][  ]',20*SW,4*SH)
        hud('  M    BONUS     L ',20*SW+5,4*SH,self.font2)
        hud(f'  {self.lives}       {max(self.bonus,0)}        {self.round+1}  ',20*SW+5,5*SH,self.font2)
        if action is not None:
            hud(f'PPO action: {ACTION_NAMES[action]}   cleared {self.cleared}/10',3*SW,0,self.font2)
        self.win.blit(pygame.transform.smoothscale(sf,self.win.get_size()) if self.scale<1 else sf,(0,0))
        pygame.display.flip()
        if fps:
            self.clock.tick(fps)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen=None

# ------------------------------------------------------------------ self test
def nav_bot(env):
    # scripted bot that follows the navigation hint (never dodges); proves each layout is solvable
    o=env._obs()
    hdx,hdy,hj=o[10],o[11],o[12]
    if hdy<0:
        return 3
    if hdy>0:
        return 4
    if hj:
        return 7 if hdx>0 else 6
    return 2 if hdx>0 else 1 if hdx<0 else 0

def selftest():
    # 1) every ladder layout is solvable: the scripted bot finishes an enemy-free round on each one
    env=DonkeyKongEnv(seed=0)
    worst=0
    for lay in [FULL]+LAYOUTS:
        env.reset()
        env.L=lay
        env.barrels,env.flames,env.m.hammer=[],[],0
        env.p=dict(env.p,spawn=10**9,tick=10**9)  # bugfix: freeze the bonus clock so slow routes can't die of 'time'
        env.bcount=0
        for i in range(1500):
            env.step(nav_bot(env))
            if env.cleared:
                break
        assert env.cleared==1,f'nav bot failed layout without ladders {lay.removed} (node {env.node})'
        worst=max(worst,4*i)
    print(f'solvable: all {1+len(LAYOUTS)} layouts cleared, slowest {worst} frames')
    env.reset()  # fresh game for the checks below
    # 2) collisions kill Mario
    for kind in ('barrel','flame'):
        env.reset()
        for _ in range(40):
            env._frame(0)
        (env.barrels if kind=='barrel' else env.flames).append(Barrel() if kind=='barrel' else Flame(2))
        (env.barrels if kind=='barrel' else env.flames)[-1].r.center=env.m.r.center
        lives=env.lives
        env.step(0)
        assert env.lives==lives-1,f'{kind} did not kill Mario'
    # 3) falling off the world and long falls kill
    env.reset()
    env.m.r.top=H+5
    lives=env.lives
    env.step(0)
    assert env.lives==lives-1
    env.reset()
    env.m.r.bottom=row6_y
    env.m.peak=env.m.r.bottom-300
    env.m.r.x=500
    lives=env.lives
    for _ in range(80):
        env.step(0)
    assert env.lives<lives,'fatal fall not detected'
    # 4) barrels never tunnel through girders / fall out of the level early
    env.reset(0)
    env.p=dict(env.p,spawn=60)
    seen=set()
    for _ in range(6000):
        env.m.r.center=(SW,H+400)  # parked harmlessly
        env.m.yc=0
        env._frame(0)
        for b in env.barrels:
            seen.add(row_of(b.r.bottom-1))
    assert {6,5,4,3,2,1}<=seen,f'barrels only reached rows {sorted(seen)}'
    print('selftest OK: collisions, falls, barrel rows',sorted(seen))

if __name__=='__main__':
    selftest()