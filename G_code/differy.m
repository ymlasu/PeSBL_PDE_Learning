function du = differy(u,formwork,order)
%本函数用于计算矩阵y方向的差分
%u为差分向量，formwork为差分模板，order为差分阶数(默认为1)
%返回值为与u同大小的向量(请自行处理边界条件)
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
