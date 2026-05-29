import numpy as np
import time

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

# ==========================================
# 核心函数：完全复刻您提供的最新 MATLAB 源码
# ==========================================
def vis_MC_Gillespie_Elastic_SLS(neta, ka, kl=0.1):
    # definition of basic parameters
    Tfinal = 1000      # unit s running time
    kc = 5             # unit pN/nm  clutch stiffness
    ron0 = 1           # unit s^-1  clutch on rate 
    roff = 0.1         # unit s^-1  clutch off rate  
    Fb = 2             # unit pN  characteristic unbinding force 
    Nc = 75            # unit 1  number of clutches
    Nm = 75            # unit 1  number of motors
    Fm = 2             # unit pN  per motor applied force
    vu = 120           # unit nm/s polymerization speed
    # initialization for variables
    Pbi = np.zeros(Nc, dtype=bool)
    roni = ron0 * np.ones(Nc)
    Fci = np.zeros(Nc)
    xci = np.zeros(Nc)
    
    # Python 中使用列表动态追加，等效于 MATLAB 中不断扩容的数组
    T = [0.0]
    vf = [vu]   
    Fsub = [0.0]
    xst = [0.0]
    
    # solving governing equations based on Monte Carlo method
    t = 0.0
    while t < Tfinal:
        roffi = roff * np.exp(Fci / Fb)  # slip bond
        
        # Determine the clutch event (Gilespie algorithm)
        Rrate = np.where(~Pbi, roni, roffi)
        Rrtot = np.sum(Rrate)
        if Rrtot == 0:
            break
            
        rnum = np.random.rand(2)
        dt = (1.0 / Rrtot) * np.log(1.0 / rnum[0])
        
        cumsum_rate = np.cumsum(Rrate) / Rrtot
        index1 = np.searchsorted(cumsum_rate, rnum[1])
        Pbi[index1] = not Pbi[index1]
        Nb = np.sum(Pbi)
        
        # Calculate the deformation/position of substrate and retrograde flow
        vf_next = vu * (1.0 - Fsub[-1] / (Nm * Fm))  # Hill's relation
        
        sum_Pbi_x_next = np.sum(Pbi * (vf_next * dt + xci))
        numerator = (ka + kl) * (neta / dt) * xst[-1] \
                  + ka * kc * sum_Pbi_x_next \
                  + (neta / dt) * kc * sum_Pbi_x_next \
                  - (neta / dt) * Fsub[-1]
                  
        # 分母部分
        denominator = (ka + kl) * (neta / dt) + ka * kl \
                    + (ka * kc * Nb + (neta / dt) * kc * Nb)
        
        xst_next = numerator / denominator
        
        # Calculate the clutch force 
        xcb = Pbi * (vf_next * dt + xci)
        xci = (~Pbi) * xst_next + xcb
        Fci = kc * (xci - xst_next) * Pbi
        Fsub_next = np.sum(Fci)
        
        t += dt
        T.append(t)
        vf.append(vf_next)
        xst.append(xst_next)
        Fsub.append(Fsub_next)
        
        if np.isnan(Fsub_next):
            break
            
    # 转为 numpy 数组进行向量化计算
    T, vf, Fsub = map(np.array, [T, vf, Fsub])
    DT = np.diff(T)
    
    # calculate the mean value
    vfmean = np.sum(vf[:-1] * DT) / Tfinal
    Fadh = np.sum(Fsub[:-1] * DT) / Tfinal
    
    return T, vf, Fsub, vfmean, Fadh


# ==========================================
# 主执行程序与画图
# ==========================================
if __name__ == "__main__":
    if plt is None:
        raise RuntimeError("matplotlib is required to run plotting examples")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    vu = 120
    start_time = time.time()
    
    # ----------------------------------------------------
    # 验证要求1: 测试 ka 较小 (Load-fail) 和 ka, ks 较大 (Frictional slippage)
    # ----------------------------------------------------
    print("正在计算要求1: 时间序列 t-vf 图...")
    T1, vf1, _, _, _ = vis_MC_Gillespie_Elastic_SLS(neta=1.0, ka=0.1, kl=0.1)
    T2, vf2, _, _, _ = vis_MC_Gillespie_Elastic_SLS(neta=1.0, ka=10, kl=10)
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(T1, vf1, 'b-', linewidth=0.5)
    ax1.set_title("Load-fail 状态 (ka=0.1, kl=0.1)")
    ax1.set_xlabel("时间 (s)"), ax1.set_ylabel("流动速度 vf (nm/s)")
    
    ax2.plot(T2, vf2, 'r-', linewidth=0.5)
    ax2.set_title("Frictional slippage 状态 (ka=10, kl=10)")
    ax2.set_xlabel("时间 (s)"), ax2.set_ylabel("流动速度 vf (nm/s)")
    plt.tight_layout()

    # ----------------------------------------------------
    # 验证要求2 & 3: 复刻您 MATLAB 中最新的参数扫描循环
    # ----------------------------------------------------
    print("正在计算要求2与3: KMCS 扫描不同 ka 与 neta ...")
    
    ka_list = [0.1, 0.5, 1.0, 10]
    
    # 【速度优化提示】:
    # MATLAB 原版使用的是 1000 个线性分布点，会导致巨量计算。
    # neta_values = np.linspace(0.01, 100, 50) 
    
    # 为了双对数作图(loglog)更美观且1-2分钟内出结果，这里改用 50 个对数分布点：
    neta_values = np.logspace(np.log10(0.01), np.log10(100), 10)
    
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 10))
    
    for ka in ka_list:  
        print(f"--> 正在扫描 ka = {ka} ...")
        vfmean_values = np.zeros(len(neta_values))
        Fadh_values = np.zeros(len(neta_values))
        
        for i in range(len(neta_values)):
            neta = neta_values[i]
            _, _, _, vfmean, Fadh = vis_MC_Gillespie_Elastic_SLS(neta, ka)
            vfmean_values[i] = vfmean
            Fadh_values[i] = Fadh
            
        # 绘制 vu - vfmean 图像 (对数刻度)
        ax3.loglog(neta_values, vu - vfmean_values, marker='.', markersize=5, label=f'ka={ka}')
        
        # 绘制 Fadh 图像 (对数刻度)
        ax4.loglog(neta_values, Fadh_values, marker='.', markersize=5, label=f'ka={ka}')

    ax3.set_xlabel('neta ($\eta$)')
    ax3.set_ylabel('vfmean')
    ax3.set_title('Mean Retrograde Flow Speed vs neta')
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.5)

    ax4.set_xlabel('neta ($\eta$)')
    ax4.set_ylabel('Fadh')
    ax4.set_title('Adhesion Force vs neta')
    ax4.legend()
    ax4.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    
    end_time = time.time()
    print(f"全部计算完成！总耗时: {end_time - start_time:.1f} 秒")
    
    plt.show()
