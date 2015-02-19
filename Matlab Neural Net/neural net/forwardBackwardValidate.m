function [WI, XI, cost] = forwardBackwardValidate(d, W, n, X, Y, fI, f0)
    cost = 0;
    X = [ X ; zeros(n-size(X,1), size(X,2)) ];
    WI = cell(d+1);
    WI{1} = DataHandle(W);
    XI = cell(d+1);
    XI{1} = DataHandle(X);
    for i = 1:d
        XI{i+1} = DataHandle(zeros(size(X)));
        WI{i+1} = DataHandle(zeros(size(W)));
    end
    
    [WI, XI, cost] = forf(0, WI, XI, fI, cost);
    [XI, cost] = f0.f(0, XI, Y, cost);
    for i = 1:d
        [XI, cost] = f0.f(i, XI, Y, cost);
        [WI, XI, cost] = forf(i, WI, XI, fI, cost);
    end
end

function [WI, XI, cost] = forf(d, WI, XI, fI, cost)
    iter = 1:length(fI);
    if d > 0
        iter = fliplr(iter);
    end
    for i = iter
        [WI, XI, cost] = fI{i}.f(d, WI, XI, cost);
    end
end