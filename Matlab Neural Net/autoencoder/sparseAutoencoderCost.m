function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

% W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
% 
% % Cost and gradient variables (your code needs to compute these values). 
% % Here, we initialize them to zeros. 
% cost = 0;
% W1grad = zeros(size(W1)); 
% W2grad = zeros(size(W2));
% b1grad = zeros(size(b1)); 
% b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

maxInput = 1000; % for memory problems

cost = 0;
grad = zeros(size(theta));
[fI,f0,~,nx] = factoryLayeredNet([visibleSize,hiddenSize,visibleSize],[],[],[],lambda/2,2,beta,sparsityParam);
% for i = 1:length(fI)
%     fI{i}
% end
for i = 1:ceil(size(data,2)/maxInput)
    index = (i-1)*maxInput+1:min(size(data,2),i*maxInput);
    [WI, ~, cost0] = forwardBackwardValidate(1, theta, nx, data(:,index), data(:,index), fI, f0);
    grad0 = WI{2}.XI;
    cost = cost + length(index)*cost0/size(data,2);
    grad = grad + length(index)*grad0/size(data,2);
end


% WI = cell(2,1);
% WI{1} = W1;
% WI{2} = W2;
% BI = cell(2,1);
% BI{1} = b1;
% BI{2} = b2;
% X = data;
% YE = X;
% f = @(x) sigmoid(x);
% df = @(x) sigmoidDeriv(x);
% [dWI, dBI, cost] = forwardBackBiasFull(WI, BI, X, YE, f, df, lambda, sparsityParam, beta);
% W1grad = dWI{1}; 
% W2grad = dWI{2};
% b1grad = dBI{1}; 
% b2grad = dBI{2};

% Y01 = W1 * data + repmat(b1,size(data,2));
% X2 = sigmoid(Y01);
% Y02 = W2 * X2 + repmat(b2,size(X2,2));
% X3 = sigmoid(Y02);
% E = X3 - YE;
% cost = sum(E(:).^2./2) ./ size(E,2);
% 
% [W2grad, b2grad, dX2, ~] = backBiasFull(W2, b2, X2, Y02, X3, E, df, lambda, sparsityParam, beta);
% [W1grad, b1grad, ~, ~] = backBiasFull(W1, b1, data, Y01, X2, dX2, df, lambda, sparsityParam, beta);


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

% grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sigmd = sigmoidDeriv(x)
    sigm = sigmoid(x);
    sigmd = sigm .* (1 - sigm);
end


function [W, X] = unbias(W, B, X)
    X = [X; ones(1,size(X,2))];
    W = [W, B];
end

function [W, B, X] = bias(W, X)
    X = X(1:end-1,:);
    B = W(:,end);
    W = W(:,1:end-1);
end

function [dWI, dBI, cost] = forwardBackBiasFull(WI, BI, X, YE, f, df, lambda, sparsityParam, beta)
    n = length(WI);
    XI = cell(n+1,1);
    Y0I = cell(n,1);
    XI{1} = X;
    for i = 1:n
        [Y0I{i}, XI{i+1}] = forwardBias(WI{i}, BI{i}, XI{i}, f);
    end
    dXI = cell(n+1,1);
    dXI{end} = XI{end} - YE;
    cost = sum(dXI{end}(:).^2./2) ./ size(dXI{end},2);
    dWI = cell(n,1);
    dBI = cell(n,1);
    for i = n:-1:1
        [dWI{i}, dBI{i}, dXI{i}, ~, cost] = backBiasFull(WI{i}, BI{i}, XI{i}, Y0I{i}, XI{i+1}, dXI{i+1}, df, lambda, sparsityParam, (i~=n)*beta, cost);
    end
end

function [Y0, Y] = forwardBias(W, B, X, f)
    [W, X] = unbias(W, B, X);
    [Y0, Y] = forward(W, X, f);
end

function [dW, dB, dX, dY0, cost] = backBiasFull(W, B, X, Y0, Y, dY, df, lambda, sparsityParam, beta, cost)
    pi = sum(Y,2) ./ size(Y,2);
    dYsparse = -sparsityParam ./ pi + (1 - sparsityParam) ./ (1 - pi);
    dY = dY + beta .* repmat(dYsparse,1,size(Y,2));
    Ysparse = (sparsityParam .* log(sparsityParam ./ pi) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - pi)));
    cost = cost + beta * sum(Ysparse(:));
    [dW, dB, dX, dY0] = backBias(W, B, X, Y0, dY, df);
    dW = dW + lambda .* W;
    cost = cost + lambda * sum(W(:).^2) ./ 2;
end

function [dW, dB, dX, dY0] = backBias(W, B, X, Y0, dY, df)
    [W, X] = unbias(W, B, X);
    [dW, dX, dY0] = back(W, X, Y0, dY, df);
    [dW, dB, dX] = bias(dW, dX);
end

function [Y0, Y] = forward(W, X, f)
    Y0 = forwardXW(W, X);
    Y = forwardY0(Y0, f);
end

function [dW, dX, dY0] = back(W, X, Y0, dY, df)
    dY0 = backY0(Y0, dY, df);
    [dW, dX] = backXW(W, X, dY0);
end

function Y = forwardY0(Y0, f)
    Y = f(Y0);
end

function dY0 = backY0(Y0, dY, df)
    dY0 = dY .* df(Y0);
end

function Y0 = forwardXW(W, X)
    Y0 = W * X;
end

function [dW, dX] = backXW(W, X, dY0)
    dW = dY0 * X' ./ size(X,2);
    dX = W' * dY0;
end

