function x = coeff(node,order)
%���������ڼ���������Ĳ��ϵ��
%nodeΪ�������ڼ�����λ����������[-2,-1,0]��
%orderΪ�������Ľ���(Ĭ��Ϊ1)��
if nargin==1
    order=1;
end
m=length(node);
factor=ones(m,1);pownode=ones(m,1);b=zeros(m,1);b(order+1)=1;A=zeros(m,m);
for i=2:m
    factor(i)=(i-1)*factor(i-1);
end
for i=1:m
    a=node(i);
    for j=2:m
        pownode(j)=a*pownode(j-1);
    end
    A(:,i)=pownode./factor;
end
[L,U]=lu(A);x=U\(L\b);
end

