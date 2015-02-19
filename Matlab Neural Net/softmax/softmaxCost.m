function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

theta = theta(:);
theta(end+1:end+numClasses) = zeros(numClasses,1);
% Refer to Neural Net/SoftMax and Neural Net/FunctKLDivergence
% fnn = factoryLayeredNetFunction([ inputSize, numClasses],[],SoftMax,@FunctKLDivergence,lambda/2,2,0,.05);
% [ cost, dW, ddW ] = fnn(theta,data(1:inputSize,:),groundTruth);
% thetagrad = dW(1:numClasses*inputSize);
maxInput = 5000; % for memory problems
[fI,f0,~,nx] = factoryLayeredNet([ inputSize, numClasses],[],SoftMax,@FunctKLDivergence,lambda/2,2,0,.05);
thetagrad = thetagrad(:);
for i = 1:ceil(size(data,2)/maxInput)
    index = (i-1)*maxInput+1:min(i*maxInput,size(data,2));
    data0 = data(:,index);
    groundTruth0 = groundTruth(:,index);
    [WI, ~, cost] = forwardBackwardValidate(1, theta, nx, data0, groundTruth0, fI, f0);
    thetagrad = thetagrad + length(index)*WI{2}.XI(1:numClasses*inputSize)/size(data,2);
end










% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

