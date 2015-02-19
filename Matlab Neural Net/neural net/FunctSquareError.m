classdef FunctSquareError
    
    properties
        nx
        ix
    end
    
    methods
        function obj = FunctSquareError(nx,ix)
            obj.nx = nx;
            obj.ix = ix;
        end
        function [XI, cost] = f(obj, d, XI, Y, cost)
            X = XI{1}.read(obj.ix, obj.nx);
            n = size(X,2);
            Z = X-Y;
            if d == 0
                cost = cost + Z(:)' * Z(:) / (2 * n);
            elseif d == 1
                XI{d+1}.write(Z / n, obj.ix);
            elseif d == 2
                XI{d+1}.write(ones(size(X)) / n, obj.ix);
            end
        end
    end
    
end