# RL

## 一、马尔可夫决策过程

### （一）随机过程

​	随机过程指的是随时间变化而演变的随机现象（例如天气随时间的变化、城市交通随时间的变化）。在随机过程

### （二）马尔科夫性质

​	当且仅当某时刻的状态仅仅取决于上一时刻的状态，则一个随机过程被称为具有马尔可夫性质。用公式表示为：
$$
P(S_{t+1}|S_t)=P(S_{t+1}|S_t,...,S_1)
$$

### （三）马尔可夫过程

​	马尔可夫过程是具有马尔可夫性质的随机过程，也成为马尔科夫链。



### （四）马尔可夫奖励过程



### （五）马尔可夫决策过程

$$
state\_value:v^{\pi}=E[G_t|S=s]
$$



## 二、值迭代与策略迭代

​	**值迭代** 和 **策略迭代** 用于解决 **最优策略求解的问题**，即给定 **MDP模型** 的状态、动作、转移概率和奖励函数，求解 **最优动作** 以及 **最大长期回报**

### （一）值迭代

The algorithm
$$
v_{k+1}=f(v_k)=max_\pi(r_\pi + \gamma P_\pi v_k)
$$
can be decomposed to two steps

**Step 1 ：policy update**
$$
\pi_{k+1}=argmax_{\pi}(r_\pi + \gamma P_\pi v_k)
$$
**Step 2 ：value update**
$$
v_{k+1}=r_{\pi+1} + \gamma P_{\pi+1} v_k
$$
![image-20241014150248322](D:\ZJUT\移动计算\强化学习\img\image-20241014150248322.png)



## 三、蒙特卡洛方法



## 四、随机近似和随机梯度下降

### （一）Robbins-Monro algorithm

​	当我们不清楚一个函数 `g(x)` 的表达式时，如何去求解 `g(x)=0` 呢，事实上 `Robbins-Monro` 可以用来解决这一问题。流程如下：
$$
w_{k+1}=w_{k}-a_{k} \tilde{g}\left(w_{k}, \eta_{k}\right)
$$
其中
$$
a_{k}是一个正系数
\\\\ 
\tilde{g}\left(w_{k}, \eta_{k}\right)=g(w_{k})+\eta_{k}为噪声,\eta_{k}为噪声
$$


理论部分
$$
In the Robbins-Monrm algorithm, if
\\\\
1) \ 1 <c_{1}\leq\nabla g(w)\leq c_{2} for \ all \ w
\\\\
导数大于零且有界，说明此函数递增
\\\\
2) \ \sum_{k=1}^{\infty} a_{k}=\infty \ and \ \ \sum_{k=1}^{\infty} a_{k}^2 < \infty
\\\\
3) \ E(\eta_{k} \ | \ H_{k})=0 \ and \ E(\eta_{k}^2 \ | \ H_{k})<\infty
$$
对式二的深入：

![image-20241015102959254](D:\ZJUT\移动计算\强化学习\img\image-20241015102959254.png)

如果前者有界，那么会对初始值的选取有限制

**RM简单实现**

```python
def f(x):
    return np.tanh(x - 1)

def RM(w1, iteration, eta=1):
    w_previous = np.float64(w1)  # 强制转换为高精度浮点数
    for k in range(1, iteration + 1):
        noise = np.random.uniform(-1, 1)
        a_k = np.float64(eta / k)  # 确保步长是高精度浮点数
        w_next = w_previous - a_k * (f(w_previous) + noise)

        w_previous = w_next
        
    return w_next
```



### （二）Stochastic gradient descent

#### 1、GD

#### 2、SGD

The **aim** of the SGD is to minimize
$$
J(w)=E[f(w,X)]
$$
The problem can be converted to **a root-finding problem**
$$
\nabla J(w)=E[\nabla f(w,X)]=0
$$
Let
$$
g(w)=\nabla J(w)
$$


What we can measure is
$$
\tilde{g}(w,\eta)=\nabla_{w}f(w,x)\\\\=
E[\nabla_{w}f(w,X)] + \nabla_{w}f(w,x)-E[\nabla_{w}f(w,X)]
\\\\where
\\\\g(w)=E[\nabla_{w}f(w,X)] \ \ and \ \ \ \eta = \nabla_{w}f(w,x)-E[\nabla_{w}f(w,X)]
$$


## 五、时序差分算法

The TD algorithm can be annotated as 
$$
v_{t+1}(s_t)=v_t(s_t)-\alpha_{t}[v_t(s_t)-[r_{t+1}+\gamma v_{t}(s_{t+1})]]
$$
Here,
$$
\overline{v}_{t}=r_{t+1}+\gamma v(s_{t+1})
$$
is called the TD target.
$$
\delta_{t}=v(s_t)-[r_{t+1} + \gamma v(s_{t+1})]=v(s_t)-\overline{v}_{t}
$$
is called the TD error.



**First, why is \delta_{t} called the TD target ?**
$$
v_{t+1}(s_t)=v_t(s_t)-\alpha_{t}[v_t(s_t)-\overline{v}_{t}]
\\\\
\Longrightarrow v_{t+1}(s_t)-\overline{v}_{t}=v_t(s_t)-\overline{v}_{t}-\alpha_{t}[v_t(s_t)-\overline{v}_{t}]\\\\
\Longrightarrow v_{t+1}(s_t)-\overline{v}_{t}=[1-\alpha_{t}][v_t(s_t)-\overline{v}_{t}]
\\\\
\Longrightarrow |v_{t+1}(s_t)-\overline{v}_{t}|=|1-\alpha_{t}||v_t(s_t)-\overline{v}_{t}|
$$
Since
$$
0<1-\alpha_{t}(s_t)<1
$$
Therefore,
$$
|v_{t+1}(s_t)-\overline{v}_{t}|<|v_t(s_t)-\overline{v}_{t}|
$$


**Second, what is the interpretation of the TD error ?**
$$
\delta_{t}=v(s_t)-[r_{t+1} + \gamma v(s_{t+1})]
$$
It is a **difference** between two consequent time steps.
$$
\delta_{t,\pi}=v_\pi(s_t)-[r_{t+1} + \gamma v(s_{t+1})]
$$
Note that
$$
E[\delta_{\pi,t}|S_t=s_t]=v_\pi(s_t)-E[R_{t+1}+\gamma v_{\pi}(S_{t+1}) | S_t=s_t]=0
$$




**What does this TD algorithm do mathematically ?**

It solves the Bellman equation of a given policy Π.



