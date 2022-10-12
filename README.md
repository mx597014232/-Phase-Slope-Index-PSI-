# -Phase-Slope-Index-PSI-
相位斜率指数(Phase Slope Index, PSI)的python版本

论文原文与官方matlab代码：
http://doc.ml.tu-berlin.de/causality/

example 在PSI.py里，直接运行就有结果了。

if __name__ == "__main__":
    signals = np.sin(range(10001))
    data_ = np.array([signals[1:], signals[0:-1]]).T
    seg_leng = 100
    ep_leng = 200
    freqbins = None

    [psi, stdpsi, psisum, stdpsisum] = data2psi(data=data_,
                                                segleng=100,
                                                epleng=200,
                                                freqbins=None)
    # # 注意，由data2psi计算的psi与论文中的{\psi}相比，它没有标准化。
    # # 最终的psi结果是以下给出的标准化版本：
    psi = psi/(stdpsi + np.finfo(float).eps)

    # 计算某个频率范围的psi
    freqs = np.array([np.arange(5, 11)])
    [psi, stdpsi, psisum, stdpsisum] = data2psi(data_, seg_leng, ep_leng, freqs)
    psi / (stdpsi + np.finfo(float).eps)

    # # 还可以计算多个频率范围之间的PSI指数，例如。
    freqs = np.array([np.arange(5, 11), np.arange(6, 12), np.arange(7, 13)])
    [psi, stdpsi, psisum, stdpsisum] = data2psi(data_, seg_leng, ep_leng, freqs)
    # # # psi有3个维度 M x M x K ，最后一个维度对应K个频率范围：
    psi = psi/(stdpsi + np.finfo(float).eps)

由于matlab 和python的计算精度有差异
因此结果可能在小数点后若干位会和matlab略有不同，一般不影响整体结果

如果论文使用了此代码，最好声明PSI的计算用的是matlab代码还是python代码
这样如果结果有差异，可以对比，方便排查问题。

如果发现我的代码有问题，还请多多指教！
