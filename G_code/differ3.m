function du = differ3(u,formwork,direct,order)
%���������ڼ�����ά����Ĳ��
%uΪ���������formworkΪ���ģ�壬directΪ��ַ���orderΪ��ֽ���(Ĭ��Ϊ1)
%directȡֵ��1,2,3.��ע��x�����Ӧ��ȡֵ�ܿ���Ϊ3��
%����ֵΪ��uͬ��С������(�����д���߽�����)
if nargin==3
    order=1;
end
coeff_u=coeff(formwork,order);
m=formwork(1);n=formwork(end);
leng=length(formwork);du=u.*0;
if direct==1
    for i=1:leng
        du(1-m:end-n,:,:)=du(1-m:end-n,:,:)+coeff_u(i).*u(1-m+formwork(i):end-n+formwork(i),:,:);
    end
elseif direct==2
    for i=1:leng
        du(:,1-m:end-n,:)=du(:,1-m:end-n,:)+coeff_u(i).*u(:,1-m+formwork(i):end-n+formwork(i),:);
    end
elseif direct==3
    for i=1:leng
        du(:,:,1-m:end-n)=du(:,:,1-m:end-n)+coeff_u(i).*u(:,:,1-m+formwork(i):end-n+formwork(i));
    end
else
    error('��ַ������');
end
end
