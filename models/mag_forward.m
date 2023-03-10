Nx=3000

for i=1:3000
    filename1=strcat('density_',num2str(i),'.mat');
    filename3=strcat('magnetic_',num2str(i),'.mat');
    filename4=strcat('magnetic_anomaly_',num2str(i),'.mat');
    
    load(filename1)
    
    
    a=randint(1,1,[5 20]);
    for i=1:80
        for j=1:80
            for k=1:40
                mag(i,j,k)=density(i,j,k)/a;
            end
        end
    end
    
    
    
h=waitbar(0,'Optimization calculation?');
for m=1:40
    for n=1:40
    sum=0;
        for i=1:40
            for j=1:40
                for k=1:10
                    rs(i,j,k)=atan((m-(i+0.5))*10000*(n-(j+0.5))/(((m-(i+0.5))^2*10000+(n-(j+0.5))^2*10000+(k+0.5)^2*10000)^(0.5))/(k+0.5)/100)- ...
                        atan((m-(i-0.5))*10000*(n-(j+0.5))/(((m-(i-0.5))^2*10000+(n-(j+0.5))^2*10000+(k+0.5)^2*10000)^(0.5))/(k+0.5)/100)- ...
                        atan((m-(i+0.5))*10000*(n-(j-0.5))/(((m-(i+0.5))^2*10000+(n-(j-0.5))^2*10000+(k+0.5)^2*10000)^(0.5))/(k+0.5)/100)+ ...
                        atan((m-(i-0.5))*10000*(n-(j-0.5))/(((m-(i-0.5))^2*10000+(n-(j-0.5))^2*10000+(k+0.5)^2*10000)^(0.5))/(k+0.5)/100)- ...
                        atan((m-(i+0.5))*10000*(n-(j+0.5))/(((m-(i+0.5))^2*10000+(n-(j+0.5))^2*10000+(k-0.5)^2*10000)^(0.5))/(k-0.5)/100)+ ...
                        atan((m-(i-0.5))*10000*(n-(j+0.5))/(((m-(i-0.5))^2*10000+(n-(j+0.5))^2*10000+(k-0.5)^2*10000)^(0.5))/(k-0.5)/100)+ ...
                        atan((m-(i+0.5))*10000*(n-(j-0.5))/(((m-(i+0.5))^2*10000+(n-(j-0.5))^2*10000+(k-0.5)^2*10000)^(0.5))/(k-0.5)/100)- ...
                        atan((m-(i-0.5))*10000*(n-(j-0.5))/(((m-(i-0.5))^2*10000+(n-(j-0.5))^2*10000+(k-0.5)^2*10000)^(0.5))/(k-0.5)/100);
                        mag(i,j,k)=ab212(i*2,j*2,k*2)*rs(i,j,k)*40000*100;
                        sum=sum+mag(i,j,k);
                end
            end
        end
        ma(m,n)=-sum/1000;
    end
    waitbar(m/40)
end

save(filename3,'mag')
save(filename4,'ma')

end
