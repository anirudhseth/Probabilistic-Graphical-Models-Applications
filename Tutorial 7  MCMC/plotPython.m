T=200;
xi = 1:T;
yi = -25:0.25:25;
[xx,yy] = meshgrid(xi,yi);
den = zeros(size(xx));
% file='a50.txt';
% N=50;
% file='a100.txt';
% N=100;
% file='a200.txt';
% N=200;
% file='a500.txt';
% N=500;
% file='a1000.txt';
% N=1000;
% file='a1500.txt';
% N=1500;
% file='a2000.txt';
% N=2000;
file='a2100.txt';
N=2100;
figure
hold on;
[a,delimiterOut]=importdata(file);
for i = xi
   % for each time step perform a kernel density estimation
   den(:,i) = ksdensity(a(i,:), yi,'kernel','epanechnikov');
   [~, idx] = max(den(:,i));

   % estimate the mode of the density
    if mod(i,10)==0
    set(gcf,'color','w');
    plot3(repmat(xi(i),length(yi),1), yi', den(:,i));
    title(sprintf('Estimated Filtering distribution plot for N=%d',N))
    xlabel('Time(t)')
    ylabel('x_t')
    zlabel('p(x_t|y_{(1:T)})')
    end
end
view(3);
