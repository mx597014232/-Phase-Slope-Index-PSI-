

% ��ͨ��1��ͨ��2�������ļ�ʾ����
n=10001;
% x=randn(n,1);
x = sin([0:10000])';
data=[x(2:n),x(1:n-1)]; 

% PSI�������
segleng=100;epleng=200;

% % PSI�ļ��㡣���һ������Ϊ�գ�����ζ��PSI��������Ƶ���ϼ����
[psi, stdpsi, psisum, stdpsisum]=data2psi(data,segleng,epleng,[]);

% % ע�⣬��data2psi�����psi�뱾���е�{\psi}���Ӧ������û�б�׼����
% % ���հ汾�������¸����Ĺ淶���汾��
psi./(stdpsi+eps)


% ����psisum��stdpisum������ı���

% 
% % To calculate psi in a band set, e.g., ��������е�psi
freqs = [5:10];
[psi, stdpsi, psisum, stdpsisum]=data2psi(data,segleng,epleng,freqs);
% %with result:
psi./(stdpsi+eps)
% % �ڱ����У����ھ���Ԫ��psi��1,2��Ϊ������˹���������ͨ��1����ͨ��2��
% 
% % ��������һ�μ�����Ƶ�������硣 
% freqs=[[5:10];[6:11];[7:12]];
% [psi, stdpsi, psisum, stdpsisum]=data2psi(data,segleng,epleng,freqs);
% %psi��3�����������һ��������ָ��Ƶ�ʱ�ʾ���У�
% psi./(stdpsi+eps)