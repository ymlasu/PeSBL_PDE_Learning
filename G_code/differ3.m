function du = differ3(u,formwork,direct,order)
%本函数用于计算三维数组的差分
%u为差分向量，formwork为差分模板，direct为差分方向，order为差分阶数(默认为1)
%direct取值：1,2,3.（注意x方向对应的取值很可能为3）
%返回值为与u同大小的向量(请自行处理边界条件)
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
    error('差分方向错误！');
end
end
