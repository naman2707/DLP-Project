classdef FunctKLDivergence
    
    properties
        nx
        ix
    end
    
    methods
        function obj = FunctKLDivergence(nx,ix)
            obj.nx = nx;
            obj.ix = ix;
        end
        function [XI, cost] = f(obj, d, XI, Y, cost)
            X = XI{1}.read(obj.ix, obj.nx);
            n = size(X,2);
            if d == 0
                cost = cost + sum(sum(Y.*log((Y+(Y==0))./X),1),2)/n;
            elseif d == 1
                XI{d+1}.write(-Y./X / n, obj.ix);
            elseif d == 2
                XI{d+1}.write(Y./X.^2 / n, obj.ix);
            end
        end
    end
    
end