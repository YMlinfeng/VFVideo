import math
from scipy.spatial.transform import Rotation as R
import numpy as np

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, x0=0.0, t0=None,  dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        if t0 is not None:
            self.t_prev = float(t0)
        self.init_flag = False

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


    def filter(self, x, t=None, reset=False, prev=None):
        """Compute the filtered signal."""
        if reset:
            self.init_flag = False
        if not self.init_flag:
            self.init_flag = True
            if prev is None:
                self.x_prev = x
                self.dx_prev = 0
                self.t_prev = t     
                return x, (self.x_prev, self.t_prev, self.dx_prev)
            else:
                self.x_prev, self.t_prev, self.dx_prev = prev

        if t is not None:
            t_e = t - self.t_prev
        else:
            t_e = 1/20 #20fps

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        # print("dx_hat : ", self.d_cutoff, a_d, dx, self.dx_prev, dx_hat)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        # print("x_hat : ", self.min_cutoff, cutoff, a, x, self.x_prev, x_hat)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat, (self.x_prev, self.t_prev, self.dx_prev)

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)

# compute angle 
def compute_angle(q1, q2):
    cos_theta = np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))
    theta = math.acos(clamp(cos_theta, -1, 1))
    return theta

# q1 --- t * theta --- r --- (1-t) * theta --- q2
# 求一个向量 r = t * q1 + (1-t) * q2 注意：非实际数学定义，实际为球面线性插值
# 参考四元数球面插值slerp计算方法，当夹角较小时可直接退化为nlerp
# 参考：https://zhuanlan.zhihu.com/p/538653027
def slerp_q(q1, q2, t):
    cos_theta = clamp(np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2)), -1, 1)
    abs_cos_theta = abs(cos_theta)
    theta = math.acos(abs_cos_theta)

    # 退化至lerp
    if abs_cos_theta >= 1.0:
        q = (1-t) * q1 + t * q2
        return q / np.linalg.norm(q)

    a_t = math.sin((1-t)*theta) / math.sin(theta)
    b_t = math.sin(t*theta) / math.sin(theta)
    b_t = b_t if cos_theta>0 else -b_t
    q = a_t * q1 + b_t * q2
    return q / np.linalg.norm(q)

def exponential_smoothing_q(a, x, x_prev):
    return slerp_q(x_prev, x, a)

# 13+75*3 - 13+75*9 process_zeggs txform
class OneEuroFilterQuaternion:
    def __init__(self, x0=None, t0=None,  dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        if t0 is not None:
            self.t_prev = float(t0)
        self.init_flag = False
    
    def filter(self, x,  t=None, reset=False, prev=None):  #(x,y,z,w)
        """Compute the filtered signal."""
        if reset:
            self.init_flag = False
        if not self.init_flag:
            self.init_flag = True
            if prev is None:
                self.x_prev = x
                self.dx_prev = 0.
                self.t_prev = t
                return x, (self.x_prev, self.t_prev, self.dx_prev)
            else: 
                self.x_prev, self.t_prev, self.dx_prev = prev
                

        if t is not None:
            t_e = t - self.t_prev
        else:
            t_e = 1./20. #20fps

        # if np.random.randn()>0.5:
        #     pass
        # else:
        #     pass
        #     #x = -x

        #like unroll function
        #四元数完全相反的/W完全相反的，在这个非数值滤波中，需要unroll吗？-- 【不需要】，因为取了dot 绝对值算theta?
        #取相反是简单数值滤波中需要
        #这其实也不是右臂twist的原因
        # zy, 0516
        # d0 = np.sum(y[i] * y[i - 1], axis=-1)
        # d1 = np.sum(-y[i] * y[i - 1], axis=-1)

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = compute_angle(self.x_prev, x) / t_e # 角速度
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev) # 平滑角速度
        # print("dx_hat : ", self.d_cutoff, a_d, dx, self.dx_prev, dx_hat)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing_q(a, x, self.x_prev) # 平滑四元数
        # print("x_hat : ", self.min_cutoff, cutoff, a, x, self.x_prev, x_hat)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat, (self.x_prev, self.t_prev, self.dx_prev)

"""
# test code 
if __name__ == '__main__':
    x0 = np.array([1, 0, 0, 0])
    xt = np.array([0, 1, 0, 0])
    x = np.zeros((101, 4))
    t = np.zeros((101, 1))
    # generate x0----xt, 100
    # generate noise
    noise = np.random.normal(0, 0.001, 100)
    print("noise : ", noise)

    for i in range(101):
        n = 0 if i == 0 else noise[i-1]
        ratio = float(i)/100. + n
        xi = slerp_q(x0, xt, ratio)
        x[i] = xi
        t[i] = i * 0.05
    print("noise x", x)

    x_res = np.zeros_like(x)
    filterQuat = OneEuroFilterQuaternion()
    for i in range(101):
        x_res[i] = filterQuat.filter(x[i], t[i])
        
    print("res x", x_res)
"""
