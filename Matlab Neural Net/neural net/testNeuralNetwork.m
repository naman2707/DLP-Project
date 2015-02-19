function testNeuralNetwork
clear
sparsityParam = 0.005; 
lambda = 0;  
beta = 0; 

in = [ 0 ]';
out = [ .5 .5 ]';
theta = [ .3 .5 1 .5 .4 .3 ]';
fnn = factoryLayeredNetFunction([ 1, 1, 2 ],Sigmoid,Sigmoid,lambda,[],beta,sparsityParam);
%fnn(theta,data,data)
checkGradient( @(x) fnn(x, in, out), theta);
%checkGradient( [], [] )

end

