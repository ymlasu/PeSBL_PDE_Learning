function du = differfitx(u,formwork,dx,order1,order2)

m=formwork(1);n=formwork(end);
[a,b]=size(u);du=u;
for i=1:a
    for j=1-m:b-n
        ind=formwork+j;
        y=u(i,ind);
        fx=polyfit(formwork*dx,y,order1);
        du(i,j,1)=fx(end);
        for k=1:order2
            fx=fx(1:end-1).*[length(fx)-1:-1:1];
            du(i,j,k+1)=fx(end);
        end
    end
end
end
