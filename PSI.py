# -*- coding: utf-8 -*-
# Time : 2022年10月12日
# name : 麦逊

# import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def data2psi(data, segleng, epleng=None, freqbins=None):
    """
    :param data: N x M矩阵, M 通道数，N 时间点
    :param segleng: bins中一小段信号的长度，（频率分辨率由它决定）
    :param epleng: bins中epoch的长度。这仅用于估计PSI的标准偏差。设置 epleng=None 可以不估计标准偏差（更快）。
    :param freqbin: K x Q矩阵。每行包含计算PSI的频率（在bins中）。
        （freqbins包括最后一个频率（f+delta f），即本文中的频带f针对第 k 个频率范围 出，即f=freqsins（k，1:end 1）。
        设置 freqbins = []，PSI，则计算全频段对应的psi。
    :return:
    psi：非标准化的psi值。对于M个输入信号通道道，PSI是 M x M 矩阵（如果freqbins只有一个 频率范围 或者为 []全频段）
        或者是 M x M x K 矩阵（如果frexbins有 K 个 频率范围（K>1））。
        psi（i，j）是从通道i到通道j的（非标准化）的相位斜率指数，（如果psi（i、j）为正值，那么通道i是信号发送方，j是信号接收方。）
    stdpsi: PSI的估计标准偏差。本文中的PSI以  PSI/（stdpsi+eps）表示（包括eps，以避免对角线元素出现 0/0 的情况）
    psisum: psisum = sum(psi,2) 是每个通道的psi值之和。
    stdpsisom: 是psisom的估计标准偏差。（无法根据 psi 和 stdpsi 计算 stdpsisum ，因此需要额外输出）
    """
    ndat, nchan = data.shape  # nchan通道数， ndat时间点
    method = 'jackknife'
    segshift = segleng / 2
    epjack = epleng
    if epleng is None:  # 如果没有指定epoch 的长度，则使用整段信号
        method = 'none'
        epleng = ndat
    if freqbins is None:  # 如果没有给定频率范围，则频率范围最大到segleng的一半
        maxfreqbin = np.floor(segleng / 2) + 1
        freqbins = np.array([np.arange(1, maxfreqbin+1)])
    else:  # 如果给定了频率范围，则根据给定的范围确定频率范围计算
        maxfreqbin = np.max(freqbins)

    nepoch = np.floor(ndat / epleng)  # epoch的数量

    if epjack > 0:
      nepochjack = np.floor(ndat/epjack).astype('int')
    else:
      nepochjack = 2

    class Para:
        def __init__(self, segeve=None, subave=None, detrend=None, proj=None, mywindow=None):
            self.segave, self.subave = segeve, subave
            self.detrend, self.proj = detrend, proj
            self.mywindow = mywindow
    para = Para(segeve=1, subave=0)

    cs, nave = data2cs_event(data, segleng, segshift, epleng, maxfreqbin, para)

    nm, nf = freqbins.shape

    psall = np.zeros((nchan, nchan, nm))
    pssumall = np.zeros((nchan, nm))
    for ii in range(0, nm):
        start, end = int(freqbins[ii, 0]) - 1, int(freqbins[ii, -1])
        psall[:, :, ii] = cs2ps(cs[:, :, start:end])
        pssumall[:, ii] = np.sum(psall[:, :, ii], 1)
    psisum = np.squeeze(pssumall)

    csall = cs
    psloc = np.zeros((nchan, nchan, nepochjack, nm))
    pssumloc = np.zeros((nchan, nepochjack, nm))

    if method == 'jackknife':
        if epjack > 0:
            for i in range(0, nepochjack):
                dataloc = data[i*epjack: (i+1)*epjack, :]
                csloc, _ = data2cs_event(dataloc, segleng, segshift, epleng, maxfreqbin, para)
                cs = (nepochjack * csall - csloc) / (nepochjack + 1)
                for ii in range(0, nm):
                    start, end = int(freqbins[ii, 0]) - 1, int(freqbins[ii, -1])
                    psloc[:, :, i, ii] = cs2ps(cs[:, :, start:end])
                    pssumloc[:, i, ii] = np.sum(psloc[:, :, i, ii], 1)
        psi = np.squeeze(psall)
        std_psi = np.squeeze(np.std(psloc, ddof=1, axis=2)) * np.sqrt(nepochjack)
        std_psisum = np.squeeze(np.std(pssumloc, ddof=1, axis=1)) * np.sqrt(nepochjack)
    else:  # method=None
        psi = psall
        std_psi = 0
        std_psisum = 0

    return psi, std_psi, psisum, std_psisum


def data2cs_event(data, segleng, segshift, epleng, maxfreqbin, para):
    """
    根据事件相关测量的数据计算交叉谱cs
    :param data: ndat x nchan矩阵
    :param segleng: bins中一小段信号的长度，（频率分辨率由它决定）
    :param segshift: 相邻的子信号段发生时移的量，
    例如，segshift = segleng / 2则会生成重叠一半的相邻子信号段
    :param epleng: 每个epoch的长度
    :param maxfreqbin: bins的最大频率
    :param para:
    segave = 0 -> 没有跨段平均
    segave != 0 -> 分段平均值（默认值为0）
    subave = 1 -> 减去各个epochs的平均值，
    subave != 1 -> 不减去各个epochs的平均值（默认值为1）
    重要提示：如果您只有一个epoch（例如，对于连续的一段信号数据）,那么：
    para.subave=0 -> 分段平均 （默认值为0）
    para.proj 必须是通道空间中的一组向量，如果存在，则输出的结果包含该通道信号的的一次傅里叶变换
    :return:
    cs: nchan x nchan x maxfreqbin x nseg 的矩阵cs（ M ：M，，f，i）包含频率 f 和信号段 i 的交叉谱
    nave: 平均值的数量
    """
    maxfreqbin = np.min([maxfreqbin, np.floor(segleng / 2) + 1]).astype('int')
    subave, segave = 1, 0
    mydetrend, proj = 0, np.array([[]])
    if para.segave is not None:
        segave = para.segave
    if para.detrend is not None:
        mydetrend = para.detrend
    if para.proj is not None:
        proj = para.proj
    if para.subave is not None:
        subave = para.subave
    ndum, npat = proj.shape
    ndat, nchan = data.shape
    if npat > 0:
        data = data * proj
        nchan = npat
    nep = np.floor(ndat / epleng).astype('int')  # epoch的数量
    nseg = np.floor((epleng - segleng) / segshift).astype('int') + 1  # seqments的数量
    if segave == 0:
        cs = np.zeros((nchan, nchan, maxfreqbin, nseg)).astype('complex128')
        av = np.zeros((nchan, maxfreqbin, nseg)).astype('complex128')
    else:
        cs = np.zeros((nchan, nchan, maxfreqbin)).astype('complex128')
        av = np.zeros((nchan, maxfreqbin)).astype('complex128')
    if npat > 0:
        if segave == 0:
            cs = np.zeros((nchan, nchan, maxfreqbin, nep, nseg)).astype('complex128')
            av = np.zeros((nchan, maxfreqbin, nep, nseg)).astype('complex128')
        else:
            cs = np.zeros((nchan, nchan, maxfreqbin, nep)).astype('complex128')
            av = np.zeros((nchan, maxfreqbin, nep)).astype('complex128')

    window = signal.windows.hann(segleng+2)[1:-1]  # hann窗口,舍弃两端值为0的窗口
    mywindow = np.repeat(window[:, np.newaxis], nchan, axis=1)
    if para.mywindow is not None:
        mywindow = np.repeat(para.mywindow[:, np.newaxis], nchan, axis=1)

    nave = 0

    for j in range(0, nep):
        dataep = data[j*epleng: (j+1)*epleng, :]
        for i in range(0, nseg):   # 所有 segments 的平均值
            start, end = int(i*segshift), int(i*segshift+segleng)
            dataloc = dataep[start:end, :]
            if mydetrend == 1:
                detrend_data_window = signal.detrend(dataloc, type == 'linear') * mywindow
                datalocfft = np.fft.fft(detrend_data_window.T).T  # detrend(x,n) 去除 n 次多项式趋势
            else:
                datalocfft = np.fft.fft((dataloc * mywindow).T).T  # python的fft处理和matlab稍有不同，矩阵需要转置处理

            for f in range(0, maxfreqbin):
                if npat == 0:
                    if segave == 0:
                        cs[:, :, f, i] = cs[:, :, f, i] + \
                                         np.conjugate(np.outer(datalocfft[f, :].T.conjugate(), datalocfft[f, :]))
                        av[:, f, i] = av[:, f, i] + np.conjugate(datalocfft[f, :].T.conjugate())  # 计算外积(注意要共轭转置)
                    else:
                        cs[:, :, f] = cs[:, :, f] + \
                                      np.conjugate(np.outer(datalocfft[f, :].T.conjugate(), datalocfft[f, :]))
                        av[:, f] = av[:, f] + np.conjugate(datalocfft[f, :].T.conjugate())  # 计算外积(注意要共轭转置)
                else:
                    if segave == 0:
                        cs[:, :, f, j, i] = np.conjugate(np.outer(datalocfft[f, :].T.conjugate(), datalocfft[f, :]))
                        av[:, f, j, i] = np.conjugate(datalocfft[f, :].T.conjugate())  # 计算外积(注意要共轭转置)
                    else:
                        cs[:, :, f, j] = cs[:, :, f, j] + \
                                         np.conjugate(np.outer(datalocfft[f, :].T.conjugate(), datalocfft[f, :]))
                        av[:, f, j] = av[:, f, j] + np.conjugate(datalocfft[f, :].T.conjugate())  # 计算外积(注意要共轭转置)
        nave = nave + 1  # 计算 average用到的n

    if segave == 0:
        cs = cs / nave
        av = av / nave
    else:
        nave = nave * nseg
        cs = cs / nave
        av = av / nave

    for f in range(0, maxfreqbin):
        if subave == 1:
            if npat == 0:
                if segave == 0:
                    for i in range(0, nseg):
                        cs[:, :, f, i] = cs[:, :, f, i] - np.outer(av[:, f, i], av[:, f, i].T.conjugate())
                else:
                    cs[:, :, f] = cs[:, :, f] - np.outer(av[:, f], av[:, f].T.conjugate())
            else:
                if segave == 0:
                    for i in range(0, nseg):
                        for j in range(1, nep + 1):
                            cs[:, :, f, j, i] = cs[:, :, f, j, i] - \
                                                np.outer(av[:, f, j, i], av[:, f, j, i].T.conjugate())
                else:
                    for j in range(0, nep):
                        cs[:, :, f, j] = cs[:, :, f, j] - np.outer(av[:, f, j], av[:, f, j].T.conjugate())

    return cs, nave


def cs2ps(cs):
    df = 1
    n_chan, n_chan, nf = cs.shape
    pp = np.copy(cs)  # 重新开辟存储空间
    for f in range(0, nf):
        pp[:, :, f] = cs[:, :, f] / np.sqrt(np.outer(np.diag(cs[:, :, f]), np.diag(cs[:, :, f]).T.conjugate()))
    ps = np.sum(np.imag(np.conj(pp[:, :, 0: nf - df]) * pp[:, :, df: nf]), 2)  # 差分,虚部
    return ps


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