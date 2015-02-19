function fnn = factoryLayeredNetFunction(H,func,funclast,funcerror,lambda,l,beta,p)
    [fI,f0,~,nx] = factoryLayeredNet(H,func,funclast,funcerror,lambda,l,beta,p);
    fnn = @(W,X,Y) neuralFunction(W, X, Y, nx, fI, f0);
%     for i = 1:length(fI)
%         fI{i}
%     end
%     f0
end

function [ cost, dW, ddW ] = neuralFunction(W, X, Y, n, fI, f0)
    [WI, XI, cost] = forwardBackwardValidate(2, W, n, X, Y, fI, f0);
%     XI{1}.XI
%     XI{2}.XI
%     XI{3}.XI
    dW = WI{2}.XI;
    ddW = WI{3}.XI;
end

