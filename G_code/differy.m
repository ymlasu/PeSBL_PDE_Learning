function du = differy(u,formwork,order)
%���������ڼ������y����Ĳ��
%uΪ���������formworkΪ���ģ�壬orderΪ��ֽ���(Ĭ��Ϊ1)
%����ֵΪ��uͬ��С������(�����д���߽�����)
if nargin==2
    order=1;
end
coeff_u=coeff(formwork,order);
m=formwork(1);n=formwork(end);
leng=length(formwork);du=u.*0;
for i=1:leng
    du(1-m:end-n,:)=du(1-m:end-n,:)+coeff_u(i).*u(1-m+formwork(i):end-n+formwork(i),:);
end

end
