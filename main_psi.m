

% 从通道1到通道2的流量的简单示例。
n=10001;
% x=randn(n,1);
x = sin([0:10000])';
data=[x(2:n),x(1:n-1)]; 

% PSI计算参数
segleng=100;epleng=200;

% % PSI的计算。最后一个参数为空，这意味着PSI是在所有频率上计算的
[psi, stdpsi, psisum, stdpsisum]=data2psi(data,segleng,epleng,[]);

% % 注意，由data2psi计算的psi与本文中的{\psi}相对应，即它没有标准化。
% % 最终版本是由以下给出的规范化版本：
psi./(stdpsi+eps)


% 关于psisum和stdpisum，请参阅本文

% 
% % To calculate psi in a band set, e.g., 计算带组中的psi
freqs = [5:10];
[psi, stdpsi, psisum, stdpsisum]=data2psi(data,segleng,epleng,freqs);
% %with result:
psi./(stdpsi+eps)
% % 在本例中，由于矩阵元素psi（1,2）为正，因此估计流量从通道1流向通道2。
% 
% % 您还可以一次计算多个频带，例如。 
% freqs=[[5:10];[6:11];[7:12]];
% [psi, stdpsi, psisum, stdpsisum]=data2psi(data,segleng,epleng,freqs);
% %psi有3个索引，最后一个索引是指以频率表示的行：
% psi./(stdpsi+eps)